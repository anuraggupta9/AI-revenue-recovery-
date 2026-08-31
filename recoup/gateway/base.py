"""The gateway boundary.

Everything above this line is decision-making; everything below is executing a
decision against a payment provider. The interface is deliberately narrow — five
verbs — because a narrow boundary is what lets the mock and the live Razorpay
adapter be genuinely interchangeable, and therefore what makes the batch
evaluation meaningful as evidence about the real path.

Idempotency is part of the interface rather than an implementation detail. Every
executing method takes a key, and both adapters promise that the same key returns
the original result instead of moving money twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from recoup.domain.case import ActionKind, RecoveryCase
from recoup.domain.events import FailureEvent
from recoup.domain.money import Money


@dataclass(frozen=True, slots=True)
class ActionResult:
    """What happened when an action was executed."""

    succeeded: bool
    action: ActionKind
    idempotency_key: str
    at: datetime
    cost: Money = field(default_factory=lambda: Money.zero())
    gateway_ref: str = ""
    # Present when the attempt failed again, so the case can be re-diagnosed
    # against the new reason rather than the original one.
    new_failure: FailureEvent | None = None
    # True when this key had already been used and the stored result was returned.
    # Surfaced rather than hidden so the audit trail can show a replay was caught.
    replayed: bool = False
    detail: str = ""


@runtime_checkable
class PaymentGateway(Protocol):
    """Minimal surface the orchestrator depends on."""

    def execute(
        self,
        case: RecoveryCase,
        action: ActionKind,
        *,
        at: datetime,
        idempotency_key: str,
    ) -> ActionResult:
        """Perform one bounded money action.

        Implementations must return the stored result for a repeated
        idempotency_key rather than acting again.
        """
        ...

    def downtime_until(self, issuer: str | None, *, at: datetime) -> datetime | None:
        """When the issuer is expected back, or None if it is healthy."""
        ...
