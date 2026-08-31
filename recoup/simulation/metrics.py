"""Turning a run into numbers, including the ones that are inconvenient.

Three decisions here are the point of the module.

Recovery is reported incrementally, against the arm's own untouched holdout,
rather than as gross recovered value. Gross recovery counts every customer who
would have paid anyway and is the single easiest way to overstate a recovery
system. Both figures are computed so the gap between them is visible.

Cost per rupee is divided by *incremental* rupees, not gross. Dividing real spend
by a number that includes money you did not earn flatters the ratio precisely in
proportion to how much you are overstating.

The incremental estimate carries a bootstrap interval. Payment amounts are heavy
tailed, so a handful of large cases dominate any value-weighted figure — during
development a 20% holdout of 400 cases produced a *negative* apparent lift purely
from which cases landed in which arm. Reporting a point estimate without an
interval would have made that noise look like a finding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from recoup.domain.case import Arm, CaseState, RecoveryCase
from recoup.domain.money import Money
from recoup.simulation.orchestrator import RunResult

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260821


def _ratio(numerator: Money, denominator: Money) -> Decimal:
    if denominator.paise == 0:
        return Decimal("0")
    return Decimal(numerator.paise) / Decimal(denominator.paise)


@dataclass(frozen=True, slots=True)
class ArmMetrics:
    key: str
    label: str
    cases: int
    treatment_cases: int
    control_cases: int
    at_risk: Money
    recovered_gross: Money
    recovered_incremental: Money
    incremental_low: Money
    incremental_high: Money
    control_rate: Decimal
    treatment_rate: Decimal
    attempts: int
    contacts: int
    escalations: int
    spend: Money
    wasted_spend: Money
    wasted_contacts: int
    halted: bool

    @property
    def gross_rate(self) -> Decimal:
        return _ratio(self.recovered_gross, self.at_risk)

    @property
    def lift(self) -> Decimal:
        """Percentage points of recovery attributable to the agent."""
        return self.treatment_rate - self.control_rate

    @property
    def cost_per_incremental_rupee(self) -> Decimal | None:
        if self.recovered_incremental.paise <= 0:
            return None
        return Decimal(self.spend.paise) / Decimal(self.recovered_incremental.paise)

    @property
    def interval_excludes_zero(self) -> bool:
        """Whether the run can distinguish this arm's effect from nothing at all."""
        return self.incremental_low.paise > 0


def _recovered(cases: Sequence[RecoveryCase]) -> list[RecoveryCase]:
    return [case for case in cases if case.state is CaseState.RECOVERED]


def _value(cases: Sequence[RecoveryCase]) -> Money:
    total = Money.zero()
    for case in cases:
        total = total + case.amount_at_risk
    return total


def _incremental(
    treatment: Sequence[RecoveryCase], control: Sequence[RecoveryCase]
) -> tuple[Money, Decimal, Decimal]:
    """Treatment recovery less what the holdout says would have happened anyway."""
    treatment_at_risk = _value(treatment)
    treatment_recovered = _value(_recovered(treatment))
    control_at_risk = _value(control)
    control_recovered = _value(_recovered(control))

    control_rate = _ratio(control_recovered, control_at_risk)
    treatment_rate = _ratio(treatment_recovered, treatment_at_risk)
    counterfactual = treatment_at_risk.scale(control_rate)
    return treatment_recovered - counterfactual, control_rate, treatment_rate


def _pairs(cases: Sequence[RecoveryCase]) -> list[tuple[int, int]]:
    """(at risk, recovered) in paise per case — the only inputs the statistic needs."""
    return [
        (
            case.amount_at_risk.paise,
            case.amount_at_risk.paise if case.state is CaseState.RECOVERED else 0,
        )
        for case in cases
    ]


def _bootstrap_incremental(
    treatment: Sequence[RecoveryCase],
    control: Sequence[RecoveryCase],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Money, Money]:
    """Percentile interval on the incremental estimate.

    Cases are resampled with replacement within each arm, which is the right unit:
    a case is what was randomised, so it is what the uncertainty is over. Returns a
    zero-width interval when either arm is empty, since there is nothing to
    resample and pretending otherwise would invent precision.

    Reduced to paise pairs before resampling rather than shuffling case objects.
    The statistic only depends on two sums per arm, and doing the arithmetic on
    integers turned a twenty-second run into a two-second one, which is what makes
    it affordable to raise the batch size until the interval is actually informative.
    """
    if not treatment or not control:
        return Money.zero(), Money.zero()

    rng = random.Random(seed)
    treatment_pairs = _pairs(treatment)
    control_pairs = _pairs(control)
    n_treatment = len(treatment_pairs)
    n_control = len(control_pairs)

    draws: list[int] = []
    for _ in range(samples):
        t_risk = t_recovered = 0
        for at_risk, recovered in rng.choices(treatment_pairs, k=n_treatment):
            t_risk += at_risk
            t_recovered += recovered
        c_risk = c_recovered = 0
        for at_risk, recovered in rng.choices(control_pairs, k=n_control):
            c_risk += at_risk
            c_recovered += recovered
        if c_risk == 0:
            continue
        # Integer arithmetic throughout, rounding the counterfactual the same way
        # Money.scale would, so the interval and the point estimate are commensurable.
        counterfactual = (t_risk * c_recovered + c_risk // 2) // c_risk
        draws.append(t_recovered - counterfactual)

    if not draws:
        return Money.zero(), Money.zero()

    draws.sort()
    low = draws[int(0.025 * len(draws))]
    high = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return Money("INR", low), Money("INR", high)


def summarise(result: RunResult, *, key: str, label: str) -> ArmMetrics:
    by_arm = result.by_arm
    treatment = by_arm[Arm.TREATMENT]
    control = by_arm[Arm.CONTROL]
    all_cases = result.cases

    # A fully held-out arm has no treatment group, so its own recovery *is* the
    # counterfactual and its incremental effect is zero by construction. Computing
    # it any other way would report the do-nothing baseline as an achievement.
    if not treatment:
        incremental = Money.zero()
        low = high = Money.zero()
        control_rate = _ratio(_value(_recovered(control)), _value(control))
        treatment_rate = control_rate
    else:
        incremental, control_rate, treatment_rate = _incremental(treatment, control)
        low, high = _bootstrap_incremental(treatment, control)

    spend = Money.zero()
    wasted_spend = Money.zero()
    wasted_contacts = 0
    for case in all_cases:
        cost = case.total_cost()
        spend = spend + cost
        if case.state is not CaseState.RECOVERED:
            wasted_spend = wasted_spend + cost
            wasted_contacts += len(case.contacts)

    return ArmMetrics(
        key=key,
        label=label,
        cases=len(all_cases),
        treatment_cases=len(treatment),
        control_cases=len(control),
        at_risk=_value(all_cases),
        recovered_gross=_value(_recovered(all_cases)),
        recovered_incremental=incremental,
        incremental_low=low,
        incremental_high=high,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        attempts=sum(case.attempts_used for case in all_cases),
        contacts=sum(len(case.contacts) for case in all_cases),
        escalations=sum(1 for case in all_cases if case.state is CaseState.ESCALATED),
        spend=spend,
        wasted_spend=wasted_spend,
        wasted_contacts=wasted_contacts,
        halted=result.halted,
    )


def _rupees(amount: Money) -> str:
    sign = "-" if amount.paise < 0 else ""
    return f"{sign}{abs(amount.paise) / 100:,.0f}"


def comparison_table(metrics: Sequence[ArmMetrics]) -> str:
    """Markdown table, ready to paste into the README.

    The holdout column is shown for a reason: arm assignment is a hash of the case
    id with a fixed salt, so the same cases are held out in every arm and the
    holdout rate should be identical down the column. That makes this a paired
    comparison rather than four independent experiments, and a reader can check the
    claim from the table itself.
    """
    header = (
        "| Arm | Cases | ₹ at risk | ₹ recovered (gross) | Gross rate | Holdout rate | "
        "Treated rate | ₹ incremental vs holdout | 95% interval | Attempts | "
        "Customer contacts | Contacts wasted | ₹ spent | Cost per ₹ incremental | Escalated |"
    )
    divider = "|" + "---|" * 15
    rows = [header, divider]
    for m in metrics:
        cost = m.cost_per_incremental_rupee
        rows.append(
            "| "
            + " | ".join(
                [
                    m.label,
                    f"{m.cases:,}",
                    _rupees(m.at_risk),
                    _rupees(m.recovered_gross),
                    f"{m.gross_rate:.1%}",
                    f"{m.control_rate:.1%}",
                    f"{m.treatment_rate:.1%}",
                    _rupees(m.recovered_incremental),
                    f"{_rupees(m.incremental_low)} to {_rupees(m.incremental_high)}",
                    f"{m.attempts:,}",
                    f"{m.contacts:,}",
                    f"{m.wasted_contacts:,}",
                    _rupees(m.spend),
                    f"₹{cost:.3f}" if cost is not None else "n/a",
                    f"{m.escalations:,}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def narrative(metrics: Sequence[ArmMetrics]) -> str:
    """Plain-language read of the table, including where it is inconclusive."""
    lines: list[str] = []
    by_key = {m.key: m for m in metrics}
    recoup = by_key.get("recoup")
    ladder = by_key.get("fixed_ladder")
    no_policy = by_key.get("no_policy")
    nothing = by_key.get("do_nothing")

    if nothing:
        lines.append(
            f"Left alone, {nothing.gross_rate:.1%} of the value at risk is recovered by "
            "customers with no intervention. Every arm below is measured against its own "
            "untouched holdout, so that share is excluded rather than claimed."
        )
    if recoup:
        verdict = (
            "the interval excludes zero"
            if recoup.interval_excludes_zero
            else "the interval includes zero, so this run cannot distinguish the effect from noise"
        )
        lines.append(
            f"Recoup recovers {_rupees(recoup.recovered_incremental)} incrementally "
            f"({recoup.lift:+.1%} on the holdout rate) and {verdict}."
        )
    if recoup and ladder:
        attempt_delta = ladder.attempts - recoup.attempts
        lines.append(
            f"The fixed ladder spends {ladder.attempts:,} attempts to recover "
            f"{_rupees(ladder.recovered_incremental)} incrementally; Recoup spends "
            f"{recoup.attempts:,} ({attempt_delta:+,} attempts) to recover "
            f"{_rupees(recoup.recovered_incremental)}. The ladder makes no customer "
            "contact at all — it only ever retries the same rail — so its advantage in "
            "contacts is a consequence of having one fewer capability, not of restraint."
        )
    if recoup and no_policy:
        lines.append(
            f"Removing the policy layer takes wasted customer contacts from "
            f"{recoup.wasted_contacts:,} to {no_policy.wasted_contacts:,} and attempts "
            f"from {recoup.attempts:,} to {no_policy.attempts:,}, for less recovered "
            f"money: {_rupees(no_policy.recovered_incremental)} against "
            f"{_rupees(recoup.recovered_incremental)}. That is what the expected-value "
            "gate, the frequency caps and the cause-aware timing are buying — more "
            "money from strictly fewer actions, which is the only version of this "
            "claim worth making."
        )
        lines.append(
            f"Escalations are near-identical ({no_policy.escalations:,} without the "
            f"policy layer, {recoup.escalations:,} with it) and that is expected: both "
            "arms share the diagnosis layer, and almost every escalation comes from a "
            "categorical prohibition or an unmapped error code rather than from a "
            "policy decision. It is not evidence either way."
        )
    if ladder and not ladder.interval_excludes_zero:
        lines.append(
            "Worth stating plainly: the fixed ladder's incremental effect is not "
            "distinguishable from zero at this sample size. Its gross recovery looks "
            f"substantial ({ladder.gross_rate:.1%}) and that is exactly the trap — most "
            "of it is customers who would have paid regardless, and the part that is not "
            "is smaller than the noise on the estimate."
        )
    return "\n\n".join(lines)
