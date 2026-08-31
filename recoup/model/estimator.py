"""Recovery-probability estimation.

The division of labour this project argues for: a calibrated statistical estimate
decides *whether* an action is worth taking, and a language model — if used at
all — only decides how to word the message. Everything in this module is the
former.

Two estimators share one interface. `HeuristicEstimator` needs no dependencies
and encodes a competent engineer's priors, which makes it both a working default
and the honest "no model" baseline to beat. `CalibratedLogisticEstimator` (in
recoup.model.logistic) learns from generated history and is calibrated, so its
outputs can be multiplied by an amount without misstating expected value.

The interface returns a probability per action rather than per case, because the
whole value of the rail switch lives in that distinction.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Mapping, Protocol, runtime_checkable

from recoup.diagnosis import Diagnosis, RootCause
from recoup.domain.case import ActionKind, RecoveryCase
from recoup.policy.timing import SALARY_DAYS, to_ist


@runtime_checkable
class ProbabilityEstimator(Protocol):
    name: str

    def estimate(
        self,
        case: RecoveryCase,
        diagnosis: Diagnosis,
        *,
        at: datetime,
        downtime_active: bool,
    ) -> Mapping[ActionKind, Decimal]:
        """Probability each candidate action recovers the money."""
        ...


# Priors a competent payments engineer would hold without any data: a dead
# instrument cannot be retried onto itself, a gateway error is usually
# transient, and reaching out converts worse than a silent retry when the
# customer was already willing.
#
# Left deliberately a shade optimistic relative to the world model. A hand-set
# prior that happened to match the truth exactly would make the calibrated model
# look pointless, and would also be dishonest — nobody's guesses are that good.
# The gap is what the calibration in `logistic` exists to close, and the
# reliability curve in EVALUATION.md is where the difference shows up.
_PRIORS: dict[tuple[RootCause, ActionKind], str] = {
    (RootCause.AUTH_FRICTION, ActionKind.RETRY_SAME_RAIL): "0.32",
    (RootCause.AUTH_FRICTION, ActionKind.SEND_PAYMENT_LINK): "0.26",
    (RootCause.INSUFFICIENT_BALANCE, ActionKind.RETRY_SAME_RAIL): "0.18",
    (RootCause.INSUFFICIENT_BALANCE, ActionKind.SEND_PAYMENT_LINK): "0.14",
    (RootCause.INSUFFICIENT_BALANCE, ActionKind.RESCHEDULE_MANDATE): "0.22",
    (RootCause.INSTRUMENT_INVALID, ActionKind.RETRY_SAME_RAIL): "0.04",
    (RootCause.INSTRUMENT_INVALID, ActionKind.RETRY_ALTERNATE_RAIL): "0.27",
    (RootCause.INSTRUMENT_INVALID, ActionKind.SEND_PAYMENT_LINK): "0.18",
    (RootCause.GATEWAY_ROUTING, ActionKind.RETRY_SAME_RAIL): "0.36",
    (RootCause.GATEWAY_ROUTING, ActionKind.RETRY_ALTERNATE_RAIL): "0.48",
    (RootCause.ISSUER_OUTAGE, ActionKind.RETRY_SAME_RAIL): "0.33",
    (RootCause.ISSUER_OUTAGE, ActionKind.RETRY_ALTERNATE_RAIL): "0.44",
    (RootCause.CUSTOMER_ABANDONED, ActionKind.SEND_PAYMENT_LINK): "0.15",
    (RootCause.CUSTOMER_ABANDONED, ActionKind.RETRY_SAME_RAIL): "0.05",
    (RootCause.MANDATE_INVALID, ActionKind.SEND_PAYMENT_LINK): "0.13",
}

_DEFAULT_PRIOR = Decimal("0.08")


class HeuristicEstimator:
    """Hand-tuned priors with timing and attempt adjustments.

    Doubles as the no-model baseline in the evaluation. It is not a straw man —
    it knows about the salary cycle and about attempt decay — which is the point:
    a learned model should have to beat a competent heuristic, not an
    incompetent one.
    """

    name = "heuristic"

    def estimate(
        self,
        case: RecoveryCase,
        diagnosis: Diagnosis,
        *,
        at: datetime,
        downtime_active: bool,
    ) -> Mapping[ActionKind, Decimal]:
        out: dict[ActionKind, Decimal] = {}
        for action in diagnosis.candidate_actions:
            probability = Decimal(
                _PRIORS.get((diagnosis.root_cause, action), str(_DEFAULT_PRIOR))
            )

            if diagnosis.root_cause is RootCause.INSUFFICIENT_BALANCE:
                probability *= (
                    Decimal("1.6") if to_ist(at).day in SALARY_DAYS else Decimal("0.7")
                )

            if downtime_active and action is not ActionKind.RETRY_ALTERNATE_RAIL:
                probability *= Decimal("0.2")

            probability *= Decimal("0.65") ** case.attempts_used
            out[action] = max(Decimal("0.001"), min(Decimal("0.99"), probability))
        return out


class FixedScheduleEstimator:
    """Baseline that ignores cause entirely.

    Stands in for what most merchants actually do — retry on a fixed ladder and
    hope — by returning one flat probability for every action so the
    expected-value gate cannot distinguish between them. Paired with a
    fixed-schedule policy in the evaluation.
    """

    name = "fixed_schedule"

    def __init__(self, flat: Decimal = Decimal("0.30")) -> None:
        self.flat = flat

    def estimate(
        self,
        case: RecoveryCase,
        diagnosis: Diagnosis,
        *,
        at: datetime,
        downtime_active: bool,
    ) -> Mapping[ActionKind, Decimal]:
        return {action: self.flat for action in diagnosis.candidate_actions}
