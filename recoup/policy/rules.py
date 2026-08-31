"""Stopping rules, one predicate at a time.

Each rule is a small pure function from context to outcome, and every rule is
evaluated and logged on every decision — including the ones that pass. That
verbosity is deliberate: a trail showing only the rule that blocked an action
proves the block, while a trail showing all seven proves the check happened at
all. The second is what the track bar is asking for.

Rules are grouped by whether they depend on the proposed action. Global rules are
evaluated once per decision; per-action rules are re-evaluated for each candidate
action, because a silent rail retry and a customer-facing payment link have
genuinely different permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Mapping

from recoup.diagnosis import Diagnosis, RootCause
from recoup.domain.case import ActionKind, RecoveryCase
from recoup.domain.money import Money
from recoup.domain.taxonomy import NEVER_AUTO_RETRY
from recoup.policy.timing import (
    is_within_contact_hours,
    ist_stamp,
    latest,
    next_contact_window,
    next_salary_window,
    to_ist,
)


class Severity(str, Enum):
    """What a failed rule means for the case."""

    # Stop permanently. No timing or action change can rescue this case.
    HARD = "hard"
    # Not now. The action may be legitimate later.
    DEFER = "defer"
    # Hand to a human; the agent will not act.
    ESCALATE = "escalate"
    # Halt the whole batch, not just this case.
    HALT = "halt"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """One rule's verdict, and — for a per-action rule — which action it judged.

    `action` is stamped by `decide()` rather than by the rule functions themselves,
    so the seven per-action rules do not each have to remember to set it. It was
    absent from the first version, which made the audit log ambiguous in exactly
    the place the log is supposed to be useful: a reader could see that
    `quiet_hours` failed but not which of three candidate actions it blocked, and
    with seven rules times three actions there were twenty-one entries per decision
    and no way to group them.
    """

    rule: str
    passed: bool
    detail: str
    severity: Severity | None = None
    defer_until: datetime | None = None
    action: ActionKind | None = None

    @property
    def blocks(self) -> bool:
        return not self.passed


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Every bound in one place, so the demo can show them being enforced.

    All monetary and probabilistic values are Decimal or Money — never float —
    for the reasons documented in recoup.domain.money.
    """

    max_attempts: int = 3
    max_contacts_per_week: int = 2
    insufficient_funds_cooloff: timedelta = timedelta(hours=72)
    # Minimum gap before the nth retry, indexed by attempts already used. This is
    # the one piece of retry timing that is not cause-specific, and it lives here
    # rather than in the orchestrator so that the loop contains no policy: the
    # orchestrator re-enters the decision immediately after a failure and the rule
    # tells it when to come back. Swapping this tuple and disabling the
    # cause-aware rules is exactly how the fixed-ladder baseline is built.
    retry_backoff: tuple[timedelta, ...] = (
        timedelta(hours=6),
        timedelta(hours=24),
        timedelta(hours=72),
    )
    # Net margin on recovered revenue, after gateway fees. Recovering ₹100 does
    # not put ₹100 in the merchant's pocket, and an EV gate that pretends
    # otherwise systematically over-acts.
    margin: Decimal = Decimal("0.85")
    retry_cost: Money = field(default_factory=lambda: Money.from_rupees("2.00"))
    contact_cost: Money = field(default_factory=lambda: Money.from_rupees("0.25"))
    # Priced-in cost of spending a customer's patience. Not a real invoice, which
    # is exactly why it must be stated explicitly rather than assumed to be zero:
    # setting it to zero is what produces systems that harass people profitably.
    annoyance_cost: Money = field(default_factory=lambda: Money.from_rupees("8.00"))
    # Minimum expected value required to act at all.
    ev_floor: Money = field(default_factory=lambda: Money.from_rupees("1.00"))
    # Minimum success probability required before *contacting* a customer,
    # independent of how much money is at stake.
    #
    # This exists because an expected-value gate alone cannot express a
    # customer-experience limit. Expected value scales with the ticket, so a
    # ₹250,000 invoice at a one-percent chance clears any sane rupee floor by a
    # wide margin — and messaging that person is still almost certainly a waste of
    # their attention. Left to arithmetic alone, the policy harasses exactly the
    # customers it can least afford to annoy. Silent retries are exempt: nobody's
    # attention is spent, so the economics genuinely are the whole story there.
    #
    # Set to 5% after measuring, not before. My first value was 10%, which sounded
    # defensible and was not: on the reference batch it gives up ₹363,662 of
    # incremental recovery to avoid 584 customer contacts, or ₹623 per contact
    # avoided. Eight rupees is what `annoyance_cost` above says a contact is worth,
    # so a 10% floor was overriding the economics by about eighty times while I was
    # describing it to myself as a mild safeguard. Either that cost is badly
    # understated or the floor was too high, and I am not entitled to both.
    #
    # 5% keeps the rule as what it should be — a backstop against contacts no
    # amount of ticket size can justify — and leaves the pricing to the EV gate.
    #
    # Two things the sweep later showed that this comment originally got wrong, both
    # in EVALUATION.md. Under the *heuristic* estimator a 5% floor is very nearly
    # inert — two blocks in 7,684 evaluations, and removing it changes neither the
    # money recovered nor the contact count — because those priors are optimistic in
    # every reliability band and almost never predict below 0.05. It only starts
    # binding once the probabilities are calibrated, at which point it blocks 585 and
    # costs ₹387 per contact avoided: still about forty-eight times `annoyance_cost`.
    # So the argument above, which I used to reject 10%, applies with less force but
    # not zero force to the 5% I chose. The number that is actually hard to defend is
    # ₹8.00, not this floor.
    min_contact_probability: Decimal = Decimal("0.05")
    # Batch-level guard: if this share of executed actions fails, stop everything.
    #
    # The threshold is high on purpose, and the first value I picked — 60% — was
    # wrong for an instructive reason. In payment recovery most individual attempts
    # fail by design; that is why the attempt cap is three and not one. A breaker
    # set at an intuitively "bad" failure rate therefore fires during completely
    # healthy operation, which is worse than having no breaker at all because it
    # trains everyone to ignore it. What this rule is actually for is catching
    # systemic breakage — dead credentials, a misrouted rail, a model returning
    # garbage — and that shows up as near-total failure, not as merely poor odds.
    circuit_breaker_failure_rate: Decimal = Decimal("0.92")
    circuit_breaker_min_sample: int = 40


@dataclass(frozen=True, slots=True)
class BatchStats:
    """Rolling batch health, for the circuit breaker."""

    actions_executed: int = 0
    actions_failed: int = 0

    @property
    def failure_rate(self) -> Decimal:
        if self.actions_executed == 0:
            return Decimal("0")
        return Decimal(self.actions_failed) / Decimal(self.actions_executed)


@dataclass(frozen=True, slots=True)
class DecisionContext:
    case: RecoveryCase
    diagnosis: Diagnosis
    now: datetime
    config: PolicyConfig
    # Fallback probability, used when no per-action estimate is supplied.
    p_success: Decimal = Decimal("0")
    # Per-action calibrated probabilities. Recovery odds are not a property of
    # the case alone — a dead card retried on the same rail is near-hopeless
    # while the same card routed to UPI is close to a coin flip — so an
    # expected-value gate fed one scalar for every action would systematically
    # misprice the rail switch that is the whole point of the intervention.
    action_probabilities: Mapping[ActionKind, Decimal] = field(default_factory=dict)
    opted_out: bool = False
    # From Razorpay's downtime feed: when the issuer is expected back.
    downtime_until: datetime | None = None
    batch: BatchStats = field(default_factory=BatchStats)

    def p_for(self, action: ActionKind) -> Decimal:
        return self.action_probabilities.get(action, self.p_success)


# ---------------------------------------------------------------------------
# Global rules — independent of which action we are considering.
# ---------------------------------------------------------------------------


def rule_circuit_breaker(ctx: DecisionContext) -> RuleOutcome:
    """Stop the batch when actions are failing at scale.

    Guards against the scenario where something systemic is wrong — expired
    credentials, a misconfigured route, a bad model — and the agent would
    otherwise work through the entire batch making the same mistake.
    """
    stats, cfg = ctx.batch, ctx.config
    if stats.actions_executed < cfg.circuit_breaker_min_sample:
        return RuleOutcome(
            "circuit_breaker",
            True,
            f"sample too small to judge ({stats.actions_executed} < {cfg.circuit_breaker_min_sample})",
        )
    if stats.failure_rate >= cfg.circuit_breaker_failure_rate:
        return RuleOutcome(
            "circuit_breaker",
            False,
            f"batch failure rate {stats.failure_rate:.0%} >= "
            f"{cfg.circuit_breaker_failure_rate:.0%}; halting for human review",
            severity=Severity.HALT,
        )
    return RuleOutcome(
        "circuit_breaker", True, f"batch failure rate {stats.failure_rate:.0%} within tolerance"
    )


def rule_customer_opt_out(ctx: DecisionContext) -> RuleOutcome:
    if ctx.opted_out:
        return RuleOutcome(
            "customer_opt_out",
            False,
            "customer has opted out of recovery contact",
            severity=Severity.HARD,
        )
    return RuleOutcome("customer_opt_out", True, "no opt-out on record")


def rule_never_auto_retry(ctx: DecisionContext) -> RuleOutcome:
    """Categorical prohibitions the model cannot override.

    Risk declines are the important case. Retrying one is how a merchant converts
    a transaction the issuer already refused into a chargeback, so this rule sits
    above the expected-value calculation rather than inside it.

    The severity is ESCALATE rather than HARD, and the difference is the point.
    HARD closes the case silently, which is right for an opt-out — the customer
    asked not to be contacted, and that applies to humans too. A suspected-fraud
    decline is the opposite: the agent must not touch it, *and* somebody should
    look at it. Marking it HARD made these cases end as `suppressed`, sitting in
    the same bucket as opt-outs, which is how a real fraud pattern would get
    filed away as a compliance stop and never read.
    """
    reason = ctx.case.latest_event.error_reason
    if reason in NEVER_AUTO_RETRY:
        return RuleOutcome(
            "never_auto_retry",
            False,
            f"error_reason={reason} is on the categorical no-retry list",
            severity=Severity.ESCALATE,
        )
    if ctx.diagnosis.root_cause is RootCause.RISK_BLOCKED:
        return RuleOutcome(
            "never_auto_retry",
            False,
            "diagnosis is risk_blocked; retrying a risk decline invites a chargeback",
            severity=Severity.ESCALATE,
        )
    return RuleOutcome("never_auto_retry", True, f"error_reason={reason} is retryable")


def rule_confidence_floor(ctx: DecisionContext) -> RuleOutcome:
    if not ctx.diagnosis.is_confident:
        return RuleOutcome(
            "confidence_floor",
            False,
            f"diagnosis confidence {ctx.diagnosis.confidence} below floor; "
            "escalating rather than guessing at a money action",
            severity=Severity.ESCALATE,
        )
    return RuleOutcome(
        "confidence_floor", True, f"diagnosis confidence {ctx.diagnosis.confidence} is sufficient"
    )


def rule_attempt_cap(ctx: DecisionContext) -> RuleOutcome:
    used, cap = ctx.case.attempts_used, ctx.config.max_attempts
    if used >= cap:
        return RuleOutcome(
            "attempt_cap",
            False,
            f"{used} of {cap} attempts used; case is exhausted",
            severity=Severity.HARD,
        )
    return RuleOutcome("attempt_cap", True, f"{used} of {cap} attempts used")


GLOBAL_RULES = (
    # Order matters: the batch-level halt outranks anything case-specific, and a
    # categorical prohibition outranks a cap that time could clear.
    rule_circuit_breaker,
    rule_customer_opt_out,
    rule_never_auto_retry,
    rule_confidence_floor,
    rule_attempt_cap,
)


# ---------------------------------------------------------------------------
# Per-action rules — re-evaluated for each candidate action.
# ---------------------------------------------------------------------------


def rule_retry_spacing(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    """Hold the next attempt until the backoff for this attempt number has elapsed.

    Cause-independent floor on retry frequency, separate from `balance_cooloff`,
    which is cause-specific and longer. Keeping them apart matters because they
    answer different questions: this one asks whether we are hammering the rail,
    the other asks whether the money is likely to be there yet.
    """
    last = ctx.case.last_attempt_at()
    ladder = ctx.config.retry_backoff
    if last is None or not ladder:
        return RuleOutcome("retry_spacing", True, "no prior attempt to space from")

    gap = ladder[min(ctx.case.attempts_used - 1, len(ladder) - 1)]
    ready = last + gap
    if ctx.now < ready:
        return RuleOutcome(
            "retry_spacing",
            False,
            f"attempt {ctx.case.attempts_used + 1} not due until {ist_stamp(ready)} "
            f"({gap} after the last attempt)",
            severity=Severity.DEFER,
            defer_until=ready,
        )
    return RuleOutcome("retry_spacing", True, f"{gap} has elapsed since the last attempt")


def rule_contact_frequency(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    """Cap how often we reach a customer, counting only actions that reach them."""
    if not action.contacts_customer:
        return RuleOutcome(
            "contact_frequency", True, f"{action} is silent and consumes no contact budget"
        )
    week = timedelta(days=7)
    used = ctx.case.contacts_in_last(week, now=ctx.now)
    cap = ctx.config.max_contacts_per_week
    if used >= cap:
        oldest = min(ctx.case.contacts[-cap:]) if ctx.case.contacts else ctx.now
        return RuleOutcome(
            "contact_frequency",
            False,
            f"{used} of {cap} weekly contacts used",
            severity=Severity.DEFER,
            defer_until=oldest + week,
        )
    return RuleOutcome("contact_frequency", True, f"{used} of {cap} weekly contacts used")


def rule_quiet_hours(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    """No customer contact outside 09:00-19:00 IST. Silent retries are exempt."""
    if not action.contacts_customer:
        return RuleOutcome("quiet_hours", True, f"{action} does not contact the customer")

    if not is_within_contact_hours(ctx.now):
        window = next_contact_window(ctx.now)
        return RuleOutcome(
            "quiet_hours",
            False,
            f"{to_ist(ctx.now):%H:%M} IST is outside contact hours",
            severity=Severity.DEFER,
            defer_until=window,
        )
    return RuleOutcome("quiet_hours", True, f"{to_ist(ctx.now):%H:%M} IST is within contact hours")


def rule_balance_cooloff(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    """Hold balance-related retries for a cool-off, then land on the salary window.

    This is the rule that distinguishes the agent from a fixed +24/48/72h ladder.
    A retry two days after an insufficient-funds decline is a worse bet than the
    same retry timed to when money is likely to have arrived.
    """
    if ctx.diagnosis.root_cause is not RootCause.INSUFFICIENT_BALANCE:
        return RuleOutcome("balance_cooloff", True, "cause is not balance-related")

    last = ctx.case.last_attempt_at() or ctx.case.opened_at
    ready = last + ctx.config.insufficient_funds_cooloff
    target = next_salary_window(ctx.now, not_before=ready)

    if ctx.now < target:
        return RuleOutcome(
            "balance_cooloff",
            False,
            f"balance failure: waiting out cool-off then landing on the salary window "
            f"({ist_stamp(target)})",
            severity=Severity.DEFER,
            defer_until=target,
        )
    return RuleOutcome("balance_cooloff", True, "cool-off elapsed and inside the salary window")


def rule_issuer_downtime(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    """Do not spend an attempt into a known outage.

    An alternate rail is exempt, since routing around the outage is the point.
    """
    if ctx.downtime_until is None or ctx.now >= ctx.downtime_until:
        return RuleOutcome("issuer_downtime", True, "no active downtime for this issuer")
    if action is ActionKind.RETRY_ALTERNATE_RAIL:
        return RuleOutcome(
            "issuer_downtime", True, "downtime active but this action routes around it"
        )
    return RuleOutcome(
        "issuer_downtime",
        False,
        f"issuer downtime active until {ist_stamp(ctx.downtime_until)}",
        severity=Severity.DEFER,
        defer_until=ctx.downtime_until,
    )


def expected_value(ctx: DecisionContext, action: ActionKind) -> Money:
    """p(success | action) x amount x margin, less the costs of attempting.

    Annoyance cost is charged only for actions the customer sees, which is what
    makes a silent retry economically preferable to a message at equal odds.
    """
    cfg = ctx.config
    gross = ctx.case.amount_at_risk.scale(ctx.p_for(action) * cfg.margin)
    cost = cfg.retry_cost if not action.contacts_customer else cfg.contact_cost
    if action.contacts_customer:
        cost = cost + cfg.annoyance_cost
    return gross - cost


def rule_ev_floor(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    ev = expected_value(ctx, action)
    if ev < ctx.config.ev_floor:
        return RuleOutcome(
            "ev_floor",
            False,
            f"expected value {ev} for {action} is below the floor {ctx.config.ev_floor} "
            f"at p={ctx.p_for(action):.3f}",
            severity=Severity.HARD,
        )
    return RuleOutcome("ev_floor", True, f"expected value {ev} for {action} clears the floor")


def rule_contact_probability_floor(ctx: DecisionContext, action: ActionKind) -> RuleOutcome:
    """Refuse to spend a customer's attention on odds this poor, whatever the ticket.

    The one rule in the set whose behaviour depends directly on the estimator being
    *calibrated* rather than merely correctly ordered. Every other rule compares a
    probability against something that scales with it — an expected value, a rupee
    floor — so a uniformly optimistic estimate shifts both sides and mostly cancels.
    This one compares the probability against a constant, so an estimator that runs
    thirty percent high authorises contact on cases it should have declined, and no
    amount of good ranking rescues it.
    """
    if not action.contacts_customer:
        return RuleOutcome(
            "contact_probability_floor", True, f"{action} spends no customer attention"
        )
    probability = ctx.p_for(action)
    floor = ctx.config.min_contact_probability
    if probability < floor:
        return RuleOutcome(
            "contact_probability_floor",
            False,
            f"p={probability:.3f} for {action} is below the {floor:.0%} floor for "
            "customer contact; the ticket size does not make it worth their attention",
            severity=Severity.HARD,
        )
    return RuleOutcome(
        "contact_probability_floor", True, f"p={probability:.3f} clears the {floor:.0%} floor"
    )


PER_ACTION_RULES = (
    rule_retry_spacing,
    rule_contact_frequency,
    rule_quiet_hours,
    rule_balance_cooloff,
    rule_issuer_downtime,
    rule_contact_probability_floor,
    rule_ev_floor,
)


@dataclass(frozen=True, slots=True)
class RuleSet:
    """A named bundle of rules, so the baselines can be built by subtraction.

    The alternative was to neuter rules through configuration — an expected-value
    floor of minus infinity, a contact cap of a thousand — and that would have been
    a worse experiment. A baseline built that way still runs the machinery and
    still pays its costs, so a difference in results could always be attributed to
    the config rather than to the missing capability. Removing the rule outright is
    the comparison actually being claimed.
    """

    name: str
    global_rules: tuple[object, ...]
    per_action_rules: tuple[object, ...]


FULL_RULES = RuleSet("full", GLOBAL_RULES, PER_ACTION_RULES)

# What a system without a policy layer looks like: it still will not retry
# forever, because even the crudest dunning has a cap, and it still spaces
# attempts because the gateway would reject a tight loop. Everything that
# constitutes judgement — expected value, quiet hours, contact frequency,
# cause-aware timing, downtime awareness, categorical prohibitions — is absent.
MINIMAL_RULES = RuleSet("minimal", (rule_attempt_cap,), (rule_retry_spacing,))


def combined_defer_time(outcomes: list[RuleOutcome]) -> datetime | None:
    """The instant at which every deferral is satisfied."""
    return latest(*[o.defer_until for o in outcomes if o.blocks])
