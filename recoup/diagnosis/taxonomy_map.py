"""Failure taxonomy to named root cause.

The design claim this module makes: diagnosis should be a lookup, not an
inference. Razorpay already tells us `error_reason`, and a reason code maps onto a
cause deterministically — so sending that to a language model would add latency,
cost, and non-reproducibility while removing an audit-friendly explanation.

The model earns its place only on the tail: reason codes we do not recognise,
where the free-text description is the only signal. Even there it may abstain,
and abstention routes to a human rather than to a guess. An unrecognised failure
is exactly the case where an automated money action is least defensible.

Deterministic-first also has a testing consequence worth having: the batch
metrics reproduce exactly, because the hot path contains no sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from recoup.domain.case import ActionKind
from recoup.domain.events import FailureEvent
from recoup.domain.taxonomy import ErrorReason, ErrorSource, ErrorStep

# Below this, we do not act. Set deliberately high: the cost of escalating a
# recoverable case to a human is a few minutes of attention, while the cost of
# acting on a misdiagnosis is a wrong charge against a real customer.
CONFIDENCE_FLOOR = Decimal("0.60")


class RootCause(str, Enum):
    """Why the money did not move, in terms that imply an intervention."""

    # The account is empty now but may not be later. Timing is the lever.
    INSUFFICIENT_BALANCE = "insufficient_balance"
    # The customer was present and willing but fumbled a step. Retry in session.
    AUTH_FRICTION = "auth_friction"
    # The instrument itself cannot work. Only a rail switch helps.
    INSTRUMENT_INVALID = "instrument_invalid"
    # The issuer is down. Wait for the window, do not burn attempts.
    ISSUER_OUTAGE = "issuer_outage"
    # Our side of the rail failed. Re-route immediately.
    GATEWAY_ROUTING = "gateway_routing"
    # Nobody declined anything; the customer walked away.
    CUSTOMER_ABANDONED = "customer_abandoned"
    # Deliberately blocked. Never retry; retrying manufactures chargebacks.
    RISK_BLOCKED = "risk_blocked"
    # Mandate no longer valid. Needs re-authorisation, not a charge.
    MANDATE_INVALID = "mandate_invalid"

    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Diagnosis:
    root_cause: RootCause
    confidence: Decimal
    rationale: str
    # Ordered best-first. The policy engine walks this list and takes the first
    # action that survives every rule, so ordering encodes clinical preference
    # while the rules retain veto power.
    candidate_actions: tuple[ActionKind, ...] = field(default_factory=tuple)
    source: str = "taxonomy"

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_FLOOR

    @property
    def is_actionable(self) -> bool:
        return self.is_confident and bool(self.candidate_actions)


# Reason code -> (cause, confidence, ordered candidate actions).
#
# Confidence is 0.97 where the code names the cause outright and 0.80 where the
# code is unambiguous about the cause but the best action depends on context the
# policy engine holds (attempt history, downtime, the customer's contact budget).
_BY_REASON: dict[ErrorReason, tuple[RootCause, str, tuple[ActionKind, ...]]] = {
    ErrorReason.INSUFFICIENT_FUNDS: (
        RootCause.INSUFFICIENT_BALANCE,
        "0.97",
        # Wait for the balance, then ask. A silent retry is cheapest, and the
        # link is the fallback when retries are exhausted.
        (ActionKind.RETRY_SAME_RAIL, ActionKind.SEND_PAYMENT_LINK),
    ),
    ErrorReason.MANDATE_INSUFFICIENT_BALANCE: (
        RootCause.INSUFFICIENT_BALANCE,
        "0.97",
        (ActionKind.RESCHEDULE_MANDATE, ActionKind.SEND_PAYMENT_LINK),
    ),
    ErrorReason.INCORRECT_OTP: (
        RootCause.AUTH_FRICTION,
        "0.97",
        (ActionKind.RETRY_SAME_RAIL, ActionKind.SEND_PAYMENT_LINK),
    ),
    ErrorReason.OTP_NOT_ENTERED: (
        RootCause.AUTH_FRICTION,
        "0.90",
        (ActionKind.RETRY_SAME_RAIL, ActionKind.SEND_PAYMENT_LINK),
    ),
    ErrorReason.INVALID_CVV: (
        RootCause.AUTH_FRICTION,
        "0.90",
        # Needs the customer to re-enter details, so a silent retry cannot fix it.
        (ActionKind.SEND_PAYMENT_LINK,),
    ),
    ErrorReason.INVALID_VPA: (
        RootCause.AUTH_FRICTION,
        "0.90",
        (ActionKind.SEND_PAYMENT_LINK,),
    ),
    ErrorReason.EXPIRED_CARD: (
        RootCause.INSTRUMENT_INVALID,
        "0.97",
        (ActionKind.RETRY_ALTERNATE_RAIL, ActionKind.SEND_PAYMENT_LINK),
    ),
    ErrorReason.CARD_DECLINED: (
        RootCause.INSTRUMENT_INVALID,
        "0.80",
        # Same-rail retries on a decline burn the network's retry allowance.
        (ActionKind.RETRY_ALTERNATE_RAIL, ActionKind.SEND_PAYMENT_LINK),
    ),
    ErrorReason.INTERNATIONAL_NOT_ALLOWED: (
        RootCause.INSTRUMENT_INVALID,
        "0.97",
        (ActionKind.RETRY_ALTERNATE_RAIL,),
    ),
    ErrorReason.MANDATE_REVOKED: (
        RootCause.MANDATE_INVALID,
        "0.97",
        # No charge is legitimate against a revoked mandate; ask for a new one.
        (ActionKind.SEND_PAYMENT_LINK,),
    ),
    ErrorReason.ISSUER_DOWN: (
        RootCause.ISSUER_OUTAGE,
        "0.97",
        (ActionKind.RETRY_SAME_RAIL, ActionKind.RETRY_ALTERNATE_RAIL),
    ),
    ErrorReason.GATEWAY_TECHNICAL_ERROR: (
        RootCause.GATEWAY_ROUTING,
        "0.90",
        (ActionKind.RETRY_ALTERNATE_RAIL, ActionKind.RETRY_SAME_RAIL),
    ),
    ErrorReason.NETWORK_ERROR: (
        RootCause.GATEWAY_ROUTING,
        "0.80",
        (ActionKind.RETRY_SAME_RAIL, ActionKind.RETRY_ALTERNATE_RAIL),
    ),
    ErrorReason.PAYMENT_TIMEOUT: (
        RootCause.CUSTOMER_ABANDONED,
        "0.80",
        (ActionKind.SEND_PAYMENT_LINK,),
    ),
    ErrorReason.UPI_COLLECT_EXPIRED: (
        RootCause.CUSTOMER_ABANDONED,
        "0.90",
        (ActionKind.SEND_PAYMENT_LINK, ActionKind.RETRY_SAME_RAIL),
    ),
    ErrorReason.RISK_DECLINED: (
        RootCause.RISK_BLOCKED,
        "0.97",
        (),  # no action is appropriate
    ),
    ErrorReason.SUSPECTED_FRAUD: (
        RootCause.RISK_BLOCKED,
        "0.97",
        (),
    ),
}

# Fallback when the reason code is unrecognised but source and step still
# constrain the cause. Confidence caps at 0.65 — above the floor, but only just,
# because we are inferring from coarser signal than a named reason code.
_BY_SOURCE_STEP: dict[tuple[ErrorSource, ErrorStep], tuple[RootCause, str, tuple[ActionKind, ...]]] = {
    (ErrorSource.GATEWAY, ErrorStep.AUTHORIZATION): (
        RootCause.GATEWAY_ROUTING,
        "0.65",
        (ActionKind.RETRY_ALTERNATE_RAIL,),
    ),
    (ErrorSource.GATEWAY, ErrorStep.INITIATION): (
        RootCause.GATEWAY_ROUTING,
        "0.65",
        (ActionKind.RETRY_ALTERNATE_RAIL,),
    ),
    (ErrorSource.GATEWAY, ErrorStep.RESPONSE): (
        RootCause.GATEWAY_ROUTING,
        "0.65",
        (ActionKind.RETRY_SAME_RAIL,),
    ),
    (ErrorSource.CUSTOMER, ErrorStep.AUTHENTICATION): (
        RootCause.AUTH_FRICTION,
        "0.65",
        (ActionKind.SEND_PAYMENT_LINK,),
    ),
    (ErrorSource.BANK, ErrorStep.AUTHORIZATION): (
        # Could be balance or a decline; both want a delayed retry, and the
        # policy engine will pick timing. Deliberately not guessing which.
        RootCause.INSUFFICIENT_BALANCE,
        "0.62",
        (ActionKind.RETRY_SAME_RAIL,),
    ),
}


def diagnose(event: FailureEvent) -> Diagnosis:
    """Map a failure onto a root cause. Deterministic and side-effect free."""
    known = _BY_REASON.get(event.error_reason)
    if known is not None:
        cause, confidence, actions = known
        return Diagnosis(
            root_cause=cause,
            confidence=Decimal(confidence),
            rationale=(
                f"error_reason={event.error_reason} maps directly to {cause}; "
                f"source={event.error_source}, step={event.error_step}"
            ),
            candidate_actions=actions,
            source="taxonomy",
        )

    fallback = _BY_SOURCE_STEP.get((event.error_source, event.error_step))
    if fallback is not None:
        cause, confidence, actions = fallback
        return Diagnosis(
            root_cause=cause,
            confidence=Decimal(confidence),
            rationale=(
                f"error_reason={event.error_reason} is unrecognised; inferred {cause} "
                f"from source={event.error_source} and step={event.error_step}"
            ),
            candidate_actions=actions,
            source="taxonomy_fallback",
        )

    return Diagnosis(
        root_cause=RootCause.UNKNOWN,
        confidence=Decimal("0"),
        rationale=(
            f"no mapping for reason={event.error_reason}, source={event.error_source}, "
            f"step={event.error_step}; description={event.error_description[:120]!r}"
        ),
        candidate_actions=(),
        source="abstain",
    )
