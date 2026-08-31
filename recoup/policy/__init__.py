"""Bounded decision-making: stopping rules, the expected-value gate, shadow mode.

Standard library only. `decide()` is pure — see recoup/policy/engine.py.
"""

from recoup.policy.engine import Decision, Outcome, decide
from recoup.policy.rules import (
    GLOBAL_RULES,
    PER_ACTION_RULES,
    BatchStats,
    DecisionContext,
    PolicyConfig,
    RuleOutcome,
    Severity,
    expected_value,
)
from recoup.policy.timing import IST, is_within_contact_hours, next_salary_window, to_ist

__all__ = [
    "BatchStats",
    "Decision",
    "DecisionContext",
    "GLOBAL_RULES",
    "IST",
    "Outcome",
    "PER_ACTION_RULES",
    "PolicyConfig",
    "RuleOutcome",
    "Severity",
    "decide",
    "expected_value",
    "is_within_contact_hours",
    "next_salary_window",
    "to_ist",
]
