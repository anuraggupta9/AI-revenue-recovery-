"""The decision engine.

Deliberately pure: `decide()` performs no IO, touches no clock, and mutates
nothing. It takes a context and returns a Decision carrying every rule outcome it
produced. The orchestrator is what writes to the audit log and executes actions.

That separation is what makes shadow mode almost free. A control-arm case runs
exactly this function and gets exactly this Decision — the only difference is that
the orchestrator logs it as a SHADOW_DECISION instead of acting on it. Retrofitting
that split after the executor exists is painful, which is why it is here from the
start rather than bolted on for the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum

from recoup.domain.case import ActionKind
from recoup.domain.money import Money
from recoup.policy.rules import (
    FULL_RULES,
    DecisionContext,
    RuleOutcome,
    RuleSet,
    Severity,
    combined_defer_time,
    expected_value,
)
from recoup.policy.timing import ist_stamp


class Outcome(str, Enum):
    ACT = "act"
    DEFER = "defer"
    STOP_EXHAUSTED = "stop_exhausted"
    STOP_UNECONOMIC = "stop_uneconomic"
    STOP_SUPPRESSED = "stop_suppressed"
    ESCALATE = "escalate"
    HALT = "halt"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Decision:
    outcome: Outcome
    rationale: str
    action: ActionKind | None = None
    execute_at: datetime | None = None
    expected_value: Money | None = None
    p_success: Decimal = Decimal("0")
    # Every rule that was evaluated, passes included. This is the audit trail.
    rule_outcomes: tuple[RuleOutcome, ...] = field(default_factory=tuple)
    # Actions we looked at and rejected, with the reason. Turns the trail into an
    # explanation of the choice rather than a record of the outcome.
    declined: tuple[tuple[ActionKind, str], ...] = field(default_factory=tuple)

    @property
    def will_act(self) -> bool:
        return self.outcome is Outcome.ACT


def _severity_to_outcome(severity: Severity | None, rule: str) -> Outcome:
    if severity is Severity.HALT:
        return Outcome.HALT
    if severity is Severity.ESCALATE:
        return Outcome.ESCALATE
    if rule == "attempt_cap":
        return Outcome.STOP_EXHAUSTED
    if rule == "ev_floor":
        return Outcome.STOP_UNECONOMIC
    return Outcome.STOP_SUPPRESSED


def decide(ctx: DecisionContext, rules: RuleSet = FULL_RULES) -> Decision:
    """Choose at most one bounded action, or decline with a reason.

    `rules` is a parameter rather than a module constant so the evaluation can run
    a genuinely rule-poor baseline through the identical code path. Defaults to the
    full set, so ordinary callers never think about it.
    """
    evaluated: list[RuleOutcome] = []

    # Stage one: rules that do not depend on the action. A block here ends the
    # decision regardless of what we might have proposed.
    for rule in rules.global_rules:
        outcome = rule(ctx)
        evaluated.append(outcome)
        if outcome.blocks:
            return Decision(
                outcome=_severity_to_outcome(outcome.severity, outcome.rule),
                rationale=f"{outcome.rule}: {outcome.detail}",
                p_success=ctx.p_success,
                rule_outcomes=tuple(evaluated),
            )

    if not ctx.diagnosis.candidate_actions:
        return Decision(
            outcome=Outcome.ESCALATE,
            rationale=(
                f"diagnosis {ctx.diagnosis.root_cause} proposes no automated action; "
                "handing to a human queue"
            ),
            p_success=ctx.p_success,
            rule_outcomes=tuple(evaluated),
        )

    # Stage two: walk candidate actions best-first. The diagnosis expresses
    # clinical preference; the rules keep veto power over each option.
    declined: list[tuple[ActionKind, str]] = []
    deferrals: list[RuleOutcome] = []

    for action in ctx.diagnosis.candidate_actions:
        # Stamped with the action here rather than inside each rule, so a new
        # per-action rule cannot forget to identify what it was judging.
        per_action: list[RuleOutcome] = [
            replace(rule(ctx, action), action=action) for rule in rules.per_action_rules
        ]
        evaluated.extend(per_action)
        blockers = [o for o in per_action if o.blocks]

        if not blockers:
            return Decision(
                outcome=Outcome.ACT,
                rationale=(
                    f"{action} cleared all {len(rules.per_action_rules)} action rules; "
                    f"root cause {ctx.diagnosis.root_cause} at p={ctx.p_for(action):.3f}"
                ),
                action=action,
                execute_at=ctx.now,
                expected_value=expected_value(ctx, action),
                p_success=ctx.p_for(action),
                rule_outcomes=tuple(evaluated),
                declined=tuple(declined),
            )

        summary = "; ".join(f"{o.rule}: {o.detail}" for o in blockers)
        declined.append((action, summary))
        # A deferral is a maybe, so remember it in case no action clears outright.
        deferrals.extend(o for o in blockers if o.severity is Severity.DEFER)

    # Every action was blocked. A deferral beats a stop: waiting preserves the
    # possibility of recovery, whereas stopping forecloses it.
    if deferrals:
        when = combined_defer_time(deferrals)
        return Decision(
            outcome=Outcome.DEFER,
            rationale=(
                "all candidate actions are blocked for now; earliest moment every "
                f"deferral is satisfied is {ist_stamp(when)}"
                if when
                else "all candidate actions are deferred"
            ),
            execute_at=when,
            p_success=ctx.p_success,
            rule_outcomes=tuple(evaluated),
            declined=tuple(declined),
        )

    hard = [o for o in evaluated if o.blocks and o.severity is Severity.HARD]
    governing = hard[-1] if hard else None
    return Decision(
        outcome=_severity_to_outcome(
            governing.severity if governing else None,
            governing.rule if governing else "",
        ),
        rationale=(
            f"no candidate action is permitted: {governing.rule}: {governing.detail}"
            if governing
            else "no candidate action is permitted"
        ),
        expected_value=(
            expected_value(ctx, ctx.diagnosis.candidate_actions[0])
            if ctx.diagnosis.candidate_actions
            else None
        ),
        p_success=ctx.p_success,
        rule_outcomes=tuple(evaluated),
        declined=tuple(declined),
    )
