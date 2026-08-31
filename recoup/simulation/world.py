"""Ground-truth world model for the synthetic batch.

This module is the honest centre of the evaluation, so it is worth being explicit
about what it is and is not.

It is a stated set of assumptions about how failed payments behave: how often a
retry works given a cause and a rail, how much correct timing helps, and — the
part most recovery demos omit — how often a customer simply pays on their own with
no intervention at all. That last parameter is why the reported figure can be
incremental rather than gross. A system measured only against "did the money
arrive after we acted" credits itself for every customer who would have paid
anyway.

It is not evidence. Nothing here is calibrated against real Razorpay data, and
the numbers are the author's estimates. Two things keep that from being fatal:
the parameters are visible and named rather than buried in a simulation loop, and
`EVALUATION.md` reports the policy's performance across perturbed versions of all
of them, so the comparison against baselines does not rest on any single guess.

The model never sees these parameters. It is trained on generated outcomes only,
exactly as it would be on production history.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from recoup.diagnosis import RootCause
from recoup.domain.case import ActionKind, RecoveryCase
from recoup.policy.timing import SALARY_DAYS, to_ist


def _stream(seed: int, *parts: str) -> random.Random:
    """A private RNG stream per (case, purpose).

    Deriving each stream from the case id rather than drawing from one shared
    generator means a case's outcomes do not shift when unrelated cases are added,
    removed or reordered. Without this, changing the batch size silently changes
    every case's result and no two runs are comparable.
    """
    material = ":".join((str(seed), *parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


# p(recovery | cause, action) when the action is well-timed and it is the first
# attempt. Ordered roughly by how much agency the customer retains.
#
# The levels are deliberately modest. My first pass used numbers in the 0.4-0.7
# range, which produced an overall recovery rate around two thirds — a figure no
# one who works on payments would believe for a moment, and one that would have
# discredited every other number in the submission. Published dunning results for
# failed card and mandate charges cluster far lower, so these are set to leave the
# headline in a range a practitioner would recognise. What the evaluation actually
# rests on is the *ratios* between rows, not their absolute level: that a dead
# instrument is hopeless on its own rail and viable on another, and that balance
# failures respond to timing. Those relationships survive the sensitivity sweep;
# the absolute level is an assumption and is labelled as one.
_BASE_SUCCESS: dict[tuple[RootCause, ActionKind], str] = {
    # Customer was present and willing; they just fumbled a step. Note this is the
    # probability *we* recover it — the ones who simply try again by themselves are
    # counted as self-recovery, not as anything the agent did.
    (RootCause.AUTH_FRICTION, ActionKind.RETRY_SAME_RAIL): "0.26",
    (RootCause.AUTH_FRICTION, ActionKind.SEND_PAYMENT_LINK): "0.21",
    # Money may have arrived since. Timing does most of the work here.
    (RootCause.INSUFFICIENT_BALANCE, ActionKind.RETRY_SAME_RAIL): "0.15",
    (RootCause.INSUFFICIENT_BALANCE, ActionKind.SEND_PAYMENT_LINK): "0.11",
    (RootCause.INSUFFICIENT_BALANCE, ActionKind.RESCHEDULE_MANDATE): "0.19",
    # The instrument is dead. Same rail cannot work; another might.
    (RootCause.INSTRUMENT_INVALID, ActionKind.RETRY_SAME_RAIL): "0.02",
    (RootCause.INSTRUMENT_INVALID, ActionKind.RETRY_ALTERNATE_RAIL): "0.23",
    (RootCause.INSTRUMENT_INVALID, ActionKind.SEND_PAYMENT_LINK): "0.15",
    # Transient. Re-routing is close to free money; waiting also works.
    (RootCause.GATEWAY_ROUTING, ActionKind.RETRY_SAME_RAIL): "0.31",
    (RootCause.GATEWAY_ROUTING, ActionKind.RETRY_ALTERNATE_RAIL): "0.42",
    (RootCause.ISSUER_OUTAGE, ActionKind.RETRY_SAME_RAIL): "0.28",
    (RootCause.ISSUER_OUTAGE, ActionKind.RETRY_ALTERNATE_RAIL): "0.39",
    # They walked away. Reaching out is the only lever.
    (RootCause.CUSTOMER_ABANDONED, ActionKind.SEND_PAYMENT_LINK): "0.12",
    (RootCause.CUSTOMER_ABANDONED, ActionKind.RETRY_SAME_RAIL): "0.04",
    (RootCause.MANDATE_INVALID, ActionKind.SEND_PAYMENT_LINK): "0.10",
}


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Every assumption, named and perturbable.

    `EVALUATION.md` re-runs the batch with each of these scaled up and down to
    show the policy's advantage is not an artefact of one lucky number.
    """

    base_success: dict[tuple[RootCause, ActionKind], str] = field(
        default_factory=lambda: dict(_BASE_SUCCESS)
    )
    # Multiplier applied to a balance-related retry that lands on the salary
    # window. The single most consequential assumption in the file, and the one
    # the policy's timing logic is built to exploit.
    salary_window_lift: Decimal = Decimal("1.75")
    # Penalty for retrying a balance failure well before payday.
    early_retry_penalty: Decimal = Decimal("0.55")
    # Each further attempt on the same case is worth less than the last.
    attempt_decay: Decimal = Decimal("0.62")
    # Retrying into a live outage mostly fails.
    downtime_penalty: Decimal = Decimal("0.15")
    # Share of failed payments the customer eventually completes with no prompting
    # at all, and how soon. Non-zero on purpose: this is the counterfactual the
    # control arm measures, and setting it to zero is how a recovery agent reports
    # a number it did not earn.
    #
    # Expressed as an overall rate plus a mean delay rather than a daily hazard.
    # A constant daily hazard was the first formulation and it was quietly
    # nonsensical — 3.5% a day compounds to roughly 80% over a 45-day horizon, so
    # almost every case recovered on its own and there was no headroom left for any
    # intervention to matter. Real unprompted recovery is front-loaded: a customer
    # who means to try again does so within days, and one who has not tried by next
    # week is not going to.
    self_recovery_rate: Decimal = Decimal("0.18")
    self_recovery_mean_days: Decimal = Decimal("3.5")
    # Customers who have opted out of recovery contact.
    opt_out_rate: Decimal = Decimal("0.04")

    def scaled(self, factor: Decimal) -> GroundTruth:
        """A perturbed copy, for the sensitivity analysis.

        Probabilities are clamped to [0.01, 0.99] so an aggressive perturbation
        cannot produce a nonsensical world.
        """

        def clamp(value: Decimal) -> Decimal:
            return max(Decimal("0.01"), min(Decimal("0.99"), value))

        return GroundTruth(
            base_success={
                key: str(clamp(Decimal(value) * factor))
                for key, value in self.base_success.items()
            },
            salary_window_lift=self.salary_window_lift,
            early_retry_penalty=self.early_retry_penalty,
            attempt_decay=self.attempt_decay,
            downtime_penalty=self.downtime_penalty,
            self_recovery_rate=clamp(self.self_recovery_rate * factor),
            self_recovery_mean_days=self.self_recovery_mean_days,
            opt_out_rate=self.opt_out_rate,
        )


class SimulatedWorld:
    """Decides what actually happens. Consulted by the mock gateway.

    Kept separate from the gateway so the gateway stays a dumb executor: it
    records an attempt and asks the world for the outcome. That boundary is what
    lets the same orchestrator drive either the mock or the live Razorpay
    adapter without the policy code knowing which.
    """

    def __init__(self, truth: GroundTruth | None = None, *, seed: int = 20260821) -> None:
        self.truth = truth or GroundTruth()
        self.seed = seed

    def success_probability(
        self,
        *,
        cause: RootCause,
        action: ActionKind,
        at: datetime,
        attempt_number: int,
        downtime_active: bool,
    ) -> Decimal:
        """True probability this attempt recovers the money.

        Never shown to the model or the policy engine — it exists so the
        simulation can sample outcomes, and so the evaluation can report how far
        the model's calibrated estimate sits from the truth.
        """
        base = self.truth.base_success.get((cause, action))
        if base is None:
            # Combination the diagnosis layer should not propose. Treated as
            # near-hopeless rather than impossible, so a bug surfaces as poor
            # performance instead of a crash.
            return Decimal("0.02")

        probability = Decimal(base)

        if cause is RootCause.INSUFFICIENT_BALANCE:
            if to_ist(at).day in SALARY_DAYS:
                probability *= self.truth.salary_window_lift
            else:
                probability *= self.truth.early_retry_penalty

        if downtime_active and action is not ActionKind.RETRY_ALTERNATE_RAIL:
            probability *= self.truth.downtime_penalty

        probability *= self.truth.attempt_decay ** (attempt_number - 1)
        return max(Decimal("0.001"), min(Decimal("0.99"), probability))

    def attempt_succeeds(
        self,
        case: RecoveryCase,
        *,
        cause: RootCause,
        action: ActionKind,
        at: datetime,
        downtime_active: bool,
    ) -> bool:
        probability = self.success_probability(
            cause=cause,
            action=action,
            at=at,
            attempt_number=case.attempts_used + 1,
            downtime_active=downtime_active,
        )
        rng = _stream(self.seed, "attempt", case.case_id, str(case.attempts_used))
        return rng.random() < float(probability)

    def self_recovery_at(self, case: RecoveryCase, *, horizon_days: int) -> datetime | None:
        """When, if ever, this customer pays without being prompted.

        Sampled per case from its own stream and independent of anything the
        agent does, which is what makes it a valid counterfactual. A case that
        self-recovers on day four would have done so whether it was in the
        treatment or the control arm.

        A draw landing past the horizon returns None rather than being clipped to
        the last day. The distinction is the honest one: we did not observe that
        recovery, so nothing should count it.
        """
        rng = _stream(self.seed, "self_recovery", case.case_id)
        if rng.random() >= float(self.truth.self_recovery_rate):
            return None
        delay = rng.expovariate(1.0 / float(self.truth.self_recovery_mean_days))
        if delay > horizon_days:
            return None
        # Land inside waking hours rather than at an arbitrary instant, so
        # ordering against scheduled actions is not degenerate.
        offset = timedelta(days=delay)
        moment = case.opened_at + offset
        return moment.replace(hour=rng.randrange(9, 21), minute=rng.randrange(60))

    def is_opted_out(self, customer_id: str) -> bool:
        rng = _stream(self.seed, "opt_out", customer_id)
        return rng.random() < float(self.truth.opt_out_rate)
