"""The recovery case: one unit of revenue at risk, and its lifecycle.

The state machine here is not bookkeeping. It is the mechanism that prevents the
worst bug this system can have — charging a customer twice because a retry
succeeded while the scheduler was already committing the next attempt. Every
money action re-checks terminality against the case immediately before firing,
and the only way to reach a terminal state is through `transition`, which refuses
to leave one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable

from recoup.domain.events import FailureEvent, Surface
from recoup.domain.money import Money


class CaseState(str, Enum):
    OPEN = "open"
    DIAGNOSED = "diagnosed"
    # Deferred on purpose: quiet hours, an insufficient-funds cool-off, or a
    # live bank downtime window. Distinct from SCHEDULED because "we chose to
    # wait" and "we chose to act later" are different decisions in the audit log.
    AWAITING_WINDOW = "awaiting_window"
    ACTION_SCHEDULED = "action_scheduled"
    ACTION_EXECUTED = "action_executed"

    RECOVERED = "recovered"
    EXHAUSTED = "exhausted"
    UNECONOMIC = "uneconomic"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"
    HALTED = "halted"
    # Control-arm case that reached the end of the horizon without paying. Its own
    # state rather than a reuse of SUPPRESSED, because the reason nothing happened
    # matters: SUPPRESSED means the policy refused to act, HELD_OUT means the policy
    # was never allowed to. Conflating them would make the holdout look like a
    # decision the agent made.
    HELD_OUT = "held_out"

    def __str__(self) -> str:
        return self.value


TERMINAL_STATES: frozenset[CaseState] = frozenset(
    {
        CaseState.RECOVERED,
        CaseState.EXHAUSTED,
        CaseState.UNECONOMIC,
        CaseState.SUPPRESSED,
        # Terminal *for the agent*. A human may act afterwards; the agent will not.
        CaseState.ESCALATED,
        CaseState.HALTED,
        CaseState.HELD_OUT,
    }
)

_ALLOWED: dict[CaseState, frozenset[CaseState]] = {
    # RECOVERED appears in every non-terminal state's set because a customer can
    # pay unprompted at any moment — before we diagnose, while we wait, or
    # between scheduling an action and firing it. Modelling that is not a
    # courtesy to the simulation: it is the event that makes gross recovery an
    # overstatement, and a state machine that cannot represent it would force the
    # agent to take credit for it.
    CaseState.OPEN: frozenset(
        {
            CaseState.DIAGNOSED,
            CaseState.RECOVERED,
            CaseState.SUPPRESSED,
            CaseState.ESCALATED,
            CaseState.HALTED,
            CaseState.HELD_OUT,
        }
    ),
    CaseState.DIAGNOSED: frozenset(
        {
            CaseState.ACTION_SCHEDULED,
            CaseState.AWAITING_WINDOW,
            CaseState.RECOVERED,
            CaseState.UNECONOMIC,
            CaseState.EXHAUSTED,
            CaseState.SUPPRESSED,
            CaseState.ESCALATED,
            CaseState.HALTED,
            CaseState.HELD_OUT,
        }
    ),
    CaseState.AWAITING_WINDOW: frozenset(
        {
            CaseState.ACTION_SCHEDULED,
            CaseState.AWAITING_WINDOW,
            CaseState.UNECONOMIC,
            CaseState.EXHAUSTED,
            CaseState.SUPPRESSED,
            CaseState.ESCALATED,
            CaseState.HALTED,
            # The customer paid on their own while we were waiting. This is the
            # transition that makes gross recovery a lie and control arms
            # necessary.
            CaseState.RECOVERED,
        }
    ),
    CaseState.ACTION_SCHEDULED: frozenset(
        {CaseState.ACTION_EXECUTED, CaseState.RECOVERED, CaseState.HALTED, CaseState.SUPPRESSED}
    ),
    CaseState.ACTION_EXECUTED: frozenset(
        {
            CaseState.RECOVERED,
            # Failed again: back round for another diagnosis with the new event.
            CaseState.DIAGNOSED,
            CaseState.AWAITING_WINDOW,
            CaseState.EXHAUSTED,
            CaseState.UNECONOMIC,
            CaseState.ESCALATED,
            CaseState.HALTED,
        }
    ),
}


class IllegalTransition(RuntimeError):
    """Raised on an undeclared state change, including any exit from a terminal state."""


class Arm(str, Enum):
    """Experiment assignment, fixed at case creation.

    CONTROL cases run the full decision pipeline and log what they *would* have
    done, but never execute a money action or contact anyone. That is what makes
    the reported figure incremental recovery rather than gross recovery.
    """

    TREATMENT = "treatment"
    CONTROL = "control"

    def __str__(self) -> str:
        return self.value


class ActionKind(str, Enum):
    RETRY_SAME_RAIL = "retry_same_rail"
    RETRY_ALTERNATE_RAIL = "retry_alternate_rail"
    SEND_PAYMENT_LINK = "send_payment_link"
    OFFER_DOWNSELL = "offer_downsell"
    RESCHEDULE_MANDATE = "reschedule_mandate"

    def __str__(self) -> str:
        return self.value

    @property
    def contacts_customer(self) -> bool:
        """Whether this action consumes the customer's contact-frequency budget.

        A silent rail retry does not reach the customer; a payment link does.
        The distinction stops the frequency cap from throttling actions that
        cost the customer no attention.
        """
        return self in {
            ActionKind.SEND_PAYMENT_LINK,
            ActionKind.OFFER_DOWNSELL,
        }


@dataclass(frozen=True, slots=True)
class Attempt:
    """One executed money action and its outcome."""

    attempted_at: datetime
    action: ActionKind
    idempotency_key: str
    succeeded: bool | None = None
    cost: Money = field(default_factory=lambda: Money.zero())
    note: str = ""


@dataclass(slots=True)
class RecoveryCase:
    """Mutable working state for one at-risk amount."""

    case_id: str
    customer_id: str
    surface: Surface
    entity_id: str
    amount_at_risk: Money
    opened_at: datetime
    arm: Arm = Arm.TREATMENT
    state: CaseState = CaseState.OPEN
    updated_at: datetime | None = None
    closed_at: datetime | None = None
    attempts: list[Attempt] = field(default_factory=list)
    contacts: list[datetime] = field(default_factory=list)
    history: list[FailureEvent] = field(default_factory=list)
    next_action_at: datetime | None = None

    @classmethod
    def open_from(cls, event: FailureEvent, *, arm: Arm = Arm.TREATMENT) -> RecoveryCase:
        case = cls(
            case_id=_case_id(event),
            customer_id=event.customer_id,
            surface=event.surface,
            entity_id=event.entity_id,
            amount_at_risk=event.amount,
            opened_at=event.occurred_at,
            arm=arm,
        )
        case.history.append(event)
        return case

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def latest_event(self) -> FailureEvent:
        if not self.history:
            raise ValueError(f"case {self.case_id} has no failure history")
        return self.history[-1]

    def transition(self, to: CaseState, *, at: datetime, reason: str = "") -> None:
        """Move to a new state, or refuse.

        Terminality is checked first and unconditionally. A case that has reached
        RECOVERED cannot be reopened into a state from which an action could
        fire, which is the invariant that makes double-charging structurally
        impossible rather than merely unlikely.
        """
        if self.is_terminal:
            raise IllegalTransition(
                f"case {self.case_id} is terminal in {self.state}; refused move to {to}"
                + (f" ({reason})" if reason else "")
            )
        if to not in _ALLOWED.get(self.state, frozenset()):
            raise IllegalTransition(
                f"case {self.case_id}: {self.state} -> {to} is not a declared transition"
            )
        self.state = to
        self.updated_at = at
        if to in TERMINAL_STATES:
            self.closed_at = at

    def guard_actionable(self) -> None:
        """Assert an action may fire right now. Call inside the executor.

        The scheduler decided to act at some earlier moment; between that decision
        and this call a webhook may have arrived saying the customer already paid.
        Re-checking here, rather than trusting the scheduler's snapshot, is what
        closes the race.
        """
        if self.is_terminal:
            raise IllegalTransition(
                f"case {self.case_id} reached {self.state} before the action fired; "
                "aborting to avoid acting on a closed case"
            )
        if self.arm is Arm.CONTROL:
            raise IllegalTransition(
                f"case {self.case_id} is in the control arm; executing a money action "
                "here would contaminate the incrementality measurement"
            )

    def record_attempt(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        if attempt.action.contacts_customer:
            self.contacts.append(attempt.attempted_at)

    def contacts_since(self, since: datetime) -> int:
        """Count contacts in the half-open interval `(since, now]`.

        Strictly greater, not `>=`, and the boundary matters. `rule_contact_frequency`
        blocks when the cap is used up and defers to `oldest_counted_contact + 7d`,
        which is the first instant the oldest contact should have aged out. Under a
        closed interval that instant still counted it, so the rule blocked again on
        exactly the timestamp it had asked to be woken at, the orchestrator saw a
        deferral that did not advance the clock, and the case escalated to a human
        instead of retrying. A one-instant disagreement between the rule that sets
        the deadline and the rule that checks it, surfaced by a case in the demo
        walkthrough ending in an escalation whose stated reason made no sense.
        """
        return sum(1 for when in self.contacts if when > since)

    def contacts_in_last(self, window: timedelta, *, now: datetime) -> int:
        return self.contacts_since(now - window)

    def last_attempt_at(self) -> datetime | None:
        return self.attempts[-1].attempted_at if self.attempts else None

    def idempotency_key_for(self, action: ActionKind) -> str:
        """Deterministic key for the next attempt.

        Derived from case, attempt ordinal and action, so a replayed webhook that
        re-drives the same decision produces the same key and the gateway
        collapses it into one charge instead of two.
        """
        material = f"{self.case_id}:{self.attempts_used + 1}:{action}"
        return "rcp_" + hashlib.sha256(material.encode()).hexdigest()[:24]

    def total_cost(self) -> Money:
        spent = Money.zero(self.amount_at_risk.currency)
        for attempt in self.attempts:
            spent = spent + attempt.cost
        return spent

    def recovered_amount(self) -> Money:
        """Money actually recovered. Zero unless the case reached RECOVERED."""
        if self.state is not CaseState.RECOVERED:
            return Money.zero(self.amount_at_risk.currency)
        return self.amount_at_risk


def _case_id(event: FailureEvent) -> str:
    return "case_" + hashlib.sha256(
        f"{event.surface}:{event.entity_id}".encode()
    ).hexdigest()[:20]


def money_at_risk(cases: Iterable[RecoveryCase], currency: str = "INR") -> Money:
    result = Money.zero(currency)
    for case in cases:
        result = result + case.amount_at_risk
    return result
