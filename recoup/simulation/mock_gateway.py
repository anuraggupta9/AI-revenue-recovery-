"""Deterministic in-process gateway.

Two jobs. It executes actions against `SimulatedWorld` so a batch can be run and
measured without touching the network, and it enforces idempotency for real —
replaying a key returns the stored result rather than acting again, which is what
the duplicate-webhook demo relies on.

Nothing here decides whether an action *should* happen. That is the policy
engine's job, and keeping the gateway ignorant of it is what makes the mock a
faithful stand-in for the live adapter.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from recoup.diagnosis import RootCause, diagnose
from recoup.domain.case import ActionKind, RecoveryCase
from recoup.domain.downtime import DowntimeWindow
from recoup.domain.events import FailureEvent
from recoup.domain.money import Money
from recoup.domain.taxonomy import ErrorReason, ErrorSource, ErrorStep, PaymentMethod
from recoup.gateway.base import ActionResult
from recoup.simulation.world import SimulatedWorld

# What each action costs the merchant to attempt. Silent retries are cheap;
# reaching a customer costs a message fee.
_COSTS: dict[ActionKind, str] = {
    ActionKind.RETRY_SAME_RAIL: "2.00",
    ActionKind.RETRY_ALTERNATE_RAIL: "2.50",
    ActionKind.SEND_PAYMENT_LINK: "0.25",
    ActionKind.OFFER_DOWNSELL: "0.25",
    ActionKind.RESCHEDULE_MANDATE: "1.50",
}

# Rail an alternate-rail retry moves to. UPI is the realistic fallback in India
# for a failed card, and it is cheaper, which is why the policy engine prefers it
# when a card instrument is dead.
_ALTERNATE_RAIL: dict[PaymentMethod, PaymentMethod] = {
    PaymentMethod.CARD: PaymentMethod.UPI,
    PaymentMethod.UPI: PaymentMethod.CARD,
    PaymentMethod.NETBANKING: PaymentMethod.UPI,
    PaymentMethod.WALLET: PaymentMethod.UPI,
    PaymentMethod.EMI: PaymentMethod.CARD,
    PaymentMethod.PAYLATER: PaymentMethod.UPI,
    PaymentMethod.UNKNOWN: PaymentMethod.UPI,
}


class MockGateway:
    """Implements PaymentGateway against a simulated world."""

    def __init__(
        self,
        world: SimulatedWorld,
        *,
        downtime: list[DowntimeWindow] | None = None,
    ) -> None:
        self.world = world
        self.downtime = downtime or []
        # idempotency_key -> the result that key produced.
        self._seen: dict[str, ActionResult] = {}
        self.calls = 0

    def execute(
        self,
        case: RecoveryCase,
        action: ActionKind,
        *,
        at: datetime,
        idempotency_key: str,
    ) -> ActionResult:
        if idempotency_key in self._seen:
            # The defining behaviour: a replayed decision does not move money a
            # second time. Flagged so the caller can log that it was caught.
            stored = self._seen[idempotency_key]
            return ActionResult(
                succeeded=stored.succeeded,
                action=stored.action,
                idempotency_key=idempotency_key,
                at=stored.at,
                cost=Money.zero(),  # no second charge, so no second cost
                gateway_ref=stored.gateway_ref,
                new_failure=stored.new_failure,
                replayed=True,
                detail="idempotency key already used; returning the stored result",
            )

        self.calls += 1
        event = case.latest_event
        cause = diagnose(event).root_cause
        downtime_active = self.downtime_until(event.issuer, at=at) is not None

        succeeded = self.world.attempt_succeeds(
            case,
            cause=cause,
            action=action,
            at=at,
            downtime_active=downtime_active,
        )
        cost = Money.from_rupees(_COSTS.get(action, "0"))
        ref = "pay_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:14]

        result = ActionResult(
            succeeded=succeeded,
            action=action,
            idempotency_key=idempotency_key,
            at=at,
            cost=cost,
            gateway_ref=ref,
            new_failure=(
                None
                if succeeded
                else _next_failure(event, action=action, at=at, downtime_active=downtime_active)
            ),
            detail=f"{action} {'succeeded' if succeeded else 'failed'} via mock gateway",
        )
        self._seen[idempotency_key] = result
        return result

    def downtime_until(self, issuer: str | None, *, at: datetime) -> datetime | None:
        if issuer is None:
            return None
        for window in self.downtime:
            if window.issuer == issuer and window.covers(at):
                return window.ends_at
        return None


def _next_failure(
    previous: FailureEvent,
    *,
    action: ActionKind,
    at: datetime,
    downtime_active: bool,
) -> FailureEvent:
    """The failure produced by an unsuccessful retry.

    The reason can legitimately change between attempts, and modelling that is
    the point: a card retried onto UPI that fails again is not still an
    `expired_card` failure, so re-diagnosing on the new reason is what stops the
    agent from looping on a stale cause.
    """
    method = previous.method
    reason = previous.error_reason
    source = previous.error_source
    step = previous.error_step

    if action is ActionKind.RETRY_ALTERNATE_RAIL:
        method = _ALTERNATE_RAIL[previous.method]
        # A different rail cannot fail for an instrument reason belonging to the
        # old one.
        if reason in {ErrorReason.EXPIRED_CARD, ErrorReason.INTERNATIONAL_NOT_ALLOWED}:
            reason = ErrorReason.CARD_DECLINED
            source = ErrorSource.BANK
            step = ErrorStep.AUTHORIZATION

    if downtime_active and action is not ActionKind.RETRY_ALTERNATE_RAIL:
        reason = ErrorReason.ISSUER_DOWN
        source = ErrorSource.BANK
        step = ErrorStep.AUTHORIZATION

    if action in {ActionKind.SEND_PAYMENT_LINK, ActionKind.OFFER_DOWNSELL}:
        # A link that is not paid expires; nobody declined anything.
        reason = ErrorReason.PAYMENT_TIMEOUT
        source = ErrorSource.CUSTOMER
        step = ErrorStep.INITIATION

    return FailureEvent(
        event_id=f"{previous.entity_id}_r{previous.attempt_number + 1}",
        occurred_at=at + timedelta(seconds=30),
        surface=previous.surface,
        entity_id=previous.entity_id,
        customer_id=previous.customer_id,
        amount=previous.amount,
        method=method,
        error_source=source,
        error_step=step,
        error_reason=reason,
        attempt_number=previous.attempt_number + 1,
        issuer=previous.issuer,
        subscription_id=previous.subscription_id,
        error_description=f"retry via {action} failed",
    )
