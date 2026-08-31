"""Value objects and entities. Standard library only — see recoup/__init__.py."""

from recoup.domain.case import (
    ActionKind,
    Arm,
    Attempt,
    CaseState,
    IllegalTransition,
    RecoveryCase,
    money_at_risk,
    TERMINAL_STATES,
)
from recoup.domain.downtime import DowntimeWindow
from recoup.domain.events import FailureEvent, Surface
from recoup.domain.money import CurrencyMismatch, Money, total
from recoup.domain.taxonomy import (
    NEVER_AUTO_RETRY,
    REQUIRES_RAIL_SWITCH,
    ErrorReason,
    ErrorSource,
    ErrorStep,
    PaymentMethod,
)

__all__ = [
    "ActionKind",
    "Arm",
    "Attempt",
    "CaseState",
    "CurrencyMismatch",
    "DowntimeWindow",
    "ErrorReason",
    "ErrorSource",
    "ErrorStep",
    "FailureEvent",
    "IllegalTransition",
    "Money",
    "NEVER_AUTO_RETRY",
    "PaymentMethod",
    "REQUIRES_RAIL_SWITCH",
    "RecoveryCase",
    "Surface",
    "TERMINAL_STATES",
    "money_at_risk",
    "total",
]
