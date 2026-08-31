"""The failure event: what arrives when money does not.

One event type covers both surfaces this project handles — a one-off payment and
a subscription mandate charge — because everything downstream (diagnosis, policy,
execution) treats them identically apart from which action verbs are available.
Modelling them as one type is what lets the second surface cost a day instead of
a week.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from recoup.domain.money import Money
from recoup.domain.taxonomy import ErrorReason, ErrorSource, ErrorStep, PaymentMethod


class Surface(str, Enum):
    """Which product surface the failure came from."""

    PAYMENT = "payment"
    SUBSCRIPTION_CHARGE = "subscription_charge"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """A single observed failure, normalised out of a webhook or a poll.

    Frozen because an event is a historical fact. Anything that changes as we
    work the problem belongs on RecoveryCase, not here.
    """

    event_id: str
    occurred_at: datetime
    surface: Surface
    entity_id: str
    customer_id: str
    amount: Money
    method: PaymentMethod
    error_source: ErrorSource
    error_step: ErrorStep
    error_reason: ErrorReason
    attempt_number: int = 1
    issuer: str | None = None
    subscription_id: str | None = None
    error_description: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError(
                "occurred_at must be timezone-aware. Quiet-hours and salary-cycle "
                "rules are evaluated in IST while the process may run in UTC, and a "
                "naive datetime here silently shifts both by 5h30m."
            )
        if self.attempt_number < 1:
            raise ValueError(f"attempt_number starts at 1, got {self.attempt_number}")

    @property
    def dedupe_key(self) -> str:
        """Stable identity for this failure, independent of delivery.

        Razorpay redelivers webhooks, and we also poll, so the same failure can
        legitimately arrive several times by different routes with different
        envelope ids. Identity therefore comes from what happened — entity,
        attempt, and reason — rather than from the transport's event id.
        """
        material = f"{self.surface}:{self.entity_id}:{self.attempt_number}:{self.error_reason}"
        return hashlib.sha256(material.encode()).hexdigest()[:32]

    @property
    def is_terminal_for_instrument(self) -> bool:
        """True when this instrument cannot succeed no matter how we time it."""
        from recoup.domain.taxonomy import REQUIRES_RAIL_SWITCH

        return self.error_reason in REQUIRES_RAIL_SWITCH

    @classmethod
    def from_razorpay(cls, payload: dict[str, Any], *, surface: Surface) -> FailureEvent:
        """Normalise a Razorpay payment entity into a FailureEvent.

        Deliberately tolerant. A missing or unrecognised field degrades to UNKNOWN
        and lets the policy engine refuse to act, which is a better failure mode
        than raising and dropping the event entirely — a dropped failure is
        revenue we never even knew was at risk.
        """
        error = payload.get("error") or {}
        occurred = payload.get("created_at")
        return cls(
            event_id=str(payload.get("id") or ""),
            occurred_at=(
                datetime.fromtimestamp(occurred, tz=timezone.utc)
                if isinstance(occurred, (int, float))
                else datetime.now(timezone.utc)
            ),
            surface=surface,
            entity_id=str(payload.get("id") or ""),
            customer_id=str(payload.get("customer_id") or payload.get("contact") or "unknown"),
            # Razorpay denominates in paise. This is the one place that
            # conversion happens, and it stays a conversion-free assignment.
            amount=Money.from_paise(
                int(payload.get("amount") or 0),
                str(payload.get("currency") or "INR"),
            ),
            method=PaymentMethod.parse(payload.get("method")),
            error_source=ErrorSource.parse(error.get("source")),
            error_step=ErrorStep.parse(error.get("step")),
            error_reason=ErrorReason.parse(error.get("reason")),
            attempt_number=int(payload.get("attempt_number") or 1),
            issuer=payload.get("bank") or payload.get("wallet") or None,
            subscription_id=payload.get("subscription_id"),
            error_description=str(error.get("description") or ""),
            raw=payload,
        )
