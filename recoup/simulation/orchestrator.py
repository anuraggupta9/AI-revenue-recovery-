"""The run loop.

A virtual-clock discrete-event scheduler. Cases are woken from a global heap
ordered by due time, each wake-up runs the full diagnose → decide → execute
cycle, and the clock only ever moves forward. That structure is what lets a
45-day recovery horizon — including deferrals that wait for a salary window a
fortnight out — run in well under a second and reproduce exactly from a seed.

Two properties are load-bearing and worth stating plainly.

The loop contains no policy. It never decides when to retry, whether an action is
worth taking, or when to give up; it asks `decide()` and does as it is told. When
a retry fails, the loop re-enters the decision at the same instant rather than
picking a delay itself, and the deferral that comes back is what sets the next
wake-up. This is why swapping `PolicyConfig` is sufficient to build the baseline
arms — there is no second copy of the timing logic hiding in here.

Control-arm cases run the same pipeline and their decisions are logged as
SHADOW_DECISION, but nothing executes and no one is contacted. They are left free
to self-recover, and the difference between the arms is the only honest measure of
what the agent actually added.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable

from recoup.audit import AuditLog, EntryKind
from recoup.diagnosis import Diagnosis, diagnose
from recoup.domain.case import (
    Arm,
    Attempt,
    CaseState,
    IllegalTransition,
    RecoveryCase,
)
from recoup.domain.events import FailureEvent
from recoup.gateway.base import PaymentGateway
from recoup.model.estimator import HeuristicEstimator, ProbabilityEstimator
from recoup.policy.engine import Decision, Outcome, decide
from recoup.policy.rules import FULL_RULES, BatchStats, DecisionContext, PolicyConfig, RuleSet
from recoup.simulation.world import SimulatedWorld

# A case that keeps deferring without ever acting is a bug, not a strategy. Every
# deferral rule provably advances the clock, so hitting this cap means one of them
# stopped doing so — and the safe response is to put the money in front of a human
# rather than spin.
MAX_WAKEUPS_PER_CASE = 40

_TERMINAL_FOR_OUTCOME = {
    Outcome.STOP_EXHAUSTED: CaseState.EXHAUSTED,
    Outcome.STOP_UNECONOMIC: CaseState.UNECONOMIC,
    Outcome.STOP_SUPPRESSED: CaseState.SUPPRESSED,
    Outcome.ESCALATE: CaseState.ESCALATED,
    Outcome.HALT: CaseState.HALTED,
}


@dataclass(slots=True)
class RunResult:
    cases: list[RecoveryCase]
    log: AuditLog
    started_at: datetime
    horizon_end: datetime
    halted: bool = False
    gateway_calls: int = 0
    duplicates_suppressed: int = 0
    shadow_decisions: int = 0
    estimator: str = ""
    rules: str = "full"
    label: str = "recoup"
    config: PolicyConfig = field(default_factory=PolicyConfig)

    @property
    def by_arm(self) -> dict[Arm, list[RecoveryCase]]:
        out: dict[Arm, list[RecoveryCase]] = {Arm.TREATMENT: [], Arm.CONTROL: []}
        for case in self.cases:
            out[case.arm].append(case)
        return out


def assign_arm(case_id: str, *, holdout_share: Decimal, salt: str = "arm") -> Arm:
    """Deterministic holdout assignment.

    Hashing the case id rather than drawing from a generator means a case lands in
    the same arm regardless of how many other cases exist or what order they are
    processed in. Reassignment between runs would make every comparison
    meaningless, and a shared RNG would cause exactly that.
    """
    digest = hashlib.sha256(f"{salt}:{case_id}".encode()).digest()
    draw = Decimal(int.from_bytes(digest[:8], "big")) / Decimal(2**64)
    return Arm.CONTROL if draw < holdout_share else Arm.TREATMENT


class Orchestrator:
    """Drives cases through the pipeline against a virtual clock."""

    def __init__(
        self,
        *,
        gateway: PaymentGateway,
        world: SimulatedWorld,
        log: AuditLog,
        config: PolicyConfig | None = None,
        estimator: ProbabilityEstimator | None = None,
        rules: RuleSet = FULL_RULES,
        diagnoser: Callable[[FailureEvent], Diagnosis] = diagnose,
        holdout_share: Decimal = Decimal("0.20"),
        label: str = "recoup",
    ) -> None:
        self.gateway = gateway
        self.world = world
        self.log = log
        self.config = config or PolicyConfig()
        self.estimator = estimator or HeuristicEstimator()
        self.rules = rules
        # Injected so a cause-blind baseline can be built without a second run
        # loop. The fixed-ladder arm supplies a diagnoser that always returns
        # "retry the same rail", which is precisely the capability it is missing.
        self.diagnoser = diagnoser
        self.holdout_share = holdout_share
        self.label = label

        self._cases: dict[str, RecoveryCase] = {}
        self._heap: list[tuple[datetime, int, str]] = []
        self._seq = 0
        self._seen_dedupe_keys: set[str] = set()
        self._self_recovery: dict[str, datetime | None] = {}
        self._wakeups: dict[str, int] = {}
        self._stats = BatchStats()
        self._duplicates = 0
        self._shadows = 0
        self._halted = False

    # -- ingestion ---------------------------------------------------------

    def ingest(self, event: FailureEvent, *, horizon_days: int) -> RecoveryCase | None:
        """Open a case, or recognise the event as one we already handled.

        Deduplication is on the event's content key, not the transport id, because
        a gateway that redelivers a webhook may well assign a fresh id to the
        second copy. Dropping on transport id alone is how duplicate charges get
        made.
        """
        if event.dedupe_key in self._seen_dedupe_keys:
            self._duplicates += 1
            self.log.append(
                EntryKind.DUPLICATE_SUPPRESSED,
                case_id="",
                at=event.occurred_at,
                event_id=event.event_id,
                dedupe_key=event.dedupe_key,
                detail="content key already seen; no second case opened",
            )
            return None
        self._seen_dedupe_keys.add(event.dedupe_key)

        arm = assign_arm(
            RecoveryCase.open_from(event).case_id, holdout_share=self.holdout_share
        )
        case = RecoveryCase.open_from(event, arm=arm)
        self._cases[case.case_id] = case
        self._self_recovery[case.case_id] = self.world.self_recovery_at(
            case, horizon_days=horizon_days
        )

        self.log.append(
            EntryKind.EVENT_INGESTED,
            case_id=case.case_id,
            at=event.occurred_at,
            arm=case.arm,
            surface=event.surface,
            entity_id=event.entity_id,
            amount=event.amount,
            method=event.method,
            error_source=event.error_source,
            error_step=event.error_step,
            error_reason=event.error_reason,
            issuer=event.issuer,
        )
        self._schedule(case.case_id, event.occurred_at)
        return case

    def _schedule(self, case_id: str, when: datetime) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (when, self._seq, case_id))

    # -- main loop ---------------------------------------------------------

    def run(
        self,
        events: list[FailureEvent],
        *,
        horizon_days: int,
        start: datetime | None = None,
    ) -> RunResult:
        started = start or (events[0].occurred_at if events else datetime.now())
        horizon_end = started + timedelta(days=horizon_days)

        for event in events:
            self.ingest(event, horizon_days=horizon_days)

        while self._heap and not self._halted:
            now, _, case_id = heapq.heappop(self._heap)
            if now > horizon_end:
                # Past the measurement window. Leaving it unprocessed rather than
                # forcing it closed keeps the horizon honest: the case is simply
                # unresolved at the point we stopped looking.
                continue
            self._tick(self._cases[case_id], now=now, horizon_end=horizon_end)

        self._finalise(horizon_end)
        return RunResult(
            cases=list(self._cases.values()),
            log=self.log,
            started_at=started,
            horizon_end=horizon_end,
            halted=self._halted,
            gateway_calls=getattr(self.gateway, "calls", 0),
            duplicates_suppressed=self._duplicates,
            shadow_decisions=self._shadows,
            estimator=self.estimator.name,
            rules=self.rules.name,
            label=self.label,
            config=self.config,
        )

    def _tick(self, case: RecoveryCase, *, now: datetime, horizon_end: datetime) -> None:
        if case.is_terminal:
            return

        # The customer may have paid unprompted since we last looked. Checked
        # before anything else and with <= rather than <, so a tie is resolved
        # against the agent: we do not let it take credit for a payment that was
        # already on its way.
        recovered_at = self._self_recovery.get(case.case_id)
        if recovered_at is not None and recovered_at <= now:
            self._close(case, CaseState.RECOVERED, at=recovered_at, reason="self_recovery")
            return

        self._wakeups[case.case_id] = self._wakeups.get(case.case_id, 0) + 1
        if self._wakeups[case.case_id] > MAX_WAKEUPS_PER_CASE:
            self._close(
                case,
                CaseState.ESCALATED,
                at=now,
                reason=f"exceeded {MAX_WAKEUPS_PER_CASE} wake-ups without resolving",
            )
            return

        diagnosis = self._diagnose(case, now=now)
        decision = self._decide(case, diagnosis, now=now)

        if case.arm is Arm.CONTROL:
            self._shadow(case, decision, now=now)
            return

        self._apply(case, decision, diagnosis, now=now, horizon_end=horizon_end)

    # -- pipeline stages ---------------------------------------------------

    def _diagnose(self, case: RecoveryCase, *, now: datetime) -> Diagnosis:
        diagnosis = self.diagnoser(case.latest_event)

        # The state change is separate from the diagnosis itself, because a case
        # woken out of AWAITING_WINDOW is re-diagnosed on every pass but has no
        # declared route back to DIAGNOSED — it goes straight to scheduling.
        if case.state in {CaseState.OPEN, CaseState.ACTION_EXECUTED}:
            case.transition(CaseState.DIAGNOSED, at=now, reason=str(diagnosis.root_cause))
            self.log.append(
                EntryKind.DIAGNOSED,
                case_id=case.case_id,
                at=now,
                root_cause=diagnosis.root_cause,
                confidence=diagnosis.confidence,
                source=diagnosis.source,
                error_reason=case.latest_event.error_reason,
                candidate_actions=[str(a) for a in diagnosis.candidate_actions],
            )
        return diagnosis

    def _decide(self, case: RecoveryCase, diagnosis: Diagnosis, *, now: datetime) -> Decision:
        event = case.latest_event
        downtime_until = self.gateway.downtime_until(event.issuer, at=now)
        probabilities = self.estimator.estimate(
            case,
            diagnosis,
            at=now,
            downtime_active=downtime_until is not None,
        )
        ctx = DecisionContext(
            case=case,
            diagnosis=diagnosis,
            now=now,
            config=self.config,
            p_success=max(probabilities.values()) if probabilities else Decimal("0"),
            action_probabilities=probabilities,
            opted_out=self.world.is_opted_out(case.customer_id),
            downtime_until=downtime_until,
            batch=self._stats,
        )
        decision = decide(ctx, self.rules)

        # Every rule, passes included. The passes are the part that proves the
        # check happened rather than merely that a block occurred.
        #
        # `action` is None for the global rules and set for the per-action ones.
        # Without it a decision over three candidate actions emits twenty-one
        # indistinguishable rule entries, and the trail cannot answer "why was the
        # rail switch rejected" — which is the question it exists to answer.
        for outcome in decision.rule_outcomes:
            self.log.append(
                EntryKind.RULE_EVALUATED,
                case_id=case.case_id,
                at=now,
                rule=outcome.rule,
                action=outcome.action,
                passed=outcome.passed,
                severity=outcome.severity,
                detail=outcome.detail,
                defer_until=outcome.defer_until,
            )
        for action, why in decision.declined:
            self.log.append(
                EntryKind.ACTION_DECLINED,
                case_id=case.case_id,
                at=now,
                action=action,
                reason=why,
            )
        return decision

    def _shadow(self, case: RecoveryCase, decision: Decision, *, now: datetime) -> None:
        """Record what a control-arm case would have done, and do nothing.

        The case is deliberately left in a non-terminal state. Closing it on the
        strength of a decision that was never executed would corrupt the
        comparison, because the control arm's whole job is to show what happens to
        this money when the agent does not touch it.
        """
        self._shadows += 1
        self.log.append(
            EntryKind.SHADOW_DECISION,
            case_id=case.case_id,
            at=now,
            outcome=decision.outcome,
            action=decision.action,
            rationale=decision.rationale,
            p_success=decision.p_success,
            expected_value=decision.expected_value,
        )

    def _apply(
        self,
        case: RecoveryCase,
        decision: Decision,
        diagnosis: Diagnosis,
        *,
        now: datetime,
        horizon_end: datetime,
    ) -> None:
        if decision.outcome is Outcome.HALT:
            self.log.append(
                EntryKind.CIRCUIT_BREAKER,
                case_id=case.case_id,
                at=now,
                rationale=decision.rationale,
                actions_executed=self._stats.actions_executed,
                actions_failed=self._stats.actions_failed,
            )
            self._halted = True
            self._close(case, CaseState.HALTED, at=now, reason=decision.rationale)
            return

        if decision.outcome is Outcome.DEFER:
            when = decision.execute_at
            if when is None or when <= now:
                # Every deferral rule is written to advance the clock. If one did
                # not, escalating is the only safe move — retrying at the same
                # instant is an infinite loop.
                self._close(
                    case,
                    CaseState.ESCALATED,
                    at=now,
                    reason="deferral did not advance the clock",
                )
                return
            case.transition(CaseState.AWAITING_WINDOW, at=now, reason=decision.rationale)
            case.next_action_at = when
            self.log.append(
                EntryKind.STATE_TRANSITION,
                case_id=case.case_id,
                at=now,
                to=CaseState.AWAITING_WINDOW,
                defer_until=when,
                rationale=decision.rationale,
            )
            if when <= horizon_end:
                self._schedule(case.case_id, when)
            return

        if decision.outcome in _TERMINAL_FOR_OUTCOME:
            self._close(
                case,
                _TERMINAL_FOR_OUTCOME[decision.outcome],
                at=now,
                reason=decision.rationale,
            )
            if decision.outcome is Outcome.ESCALATE:
                self.log.append(
                    EntryKind.ESCALATED,
                    case_id=case.case_id,
                    at=now,
                    root_cause=diagnosis.root_cause,
                    confidence=diagnosis.confidence,
                    amount=case.amount_at_risk,
                    error_reason=case.latest_event.error_reason,
                    rationale=decision.rationale,
                )
            return

        self._execute(case, decision, now=now, horizon_end=horizon_end)

    def _execute(
        self,
        case: RecoveryCase,
        decision: Decision,
        *,
        now: datetime,
        horizon_end: datetime,
    ) -> None:
        action = decision.action
        assert action is not None, "an ACT decision must name an action"

        case.transition(CaseState.ACTION_SCHEDULED, at=now, reason=str(action))
        self.log.append(
            EntryKind.ACTION_CHOSEN,
            case_id=case.case_id,
            at=now,
            action=action,
            rationale=decision.rationale,
            p_success=decision.p_success,
        )
        self.log.append(
            EntryKind.EV_COMPUTED,
            case_id=case.case_id,
            at=now,
            action=action,
            p_success=decision.p_success,
            expected_value=decision.expected_value,
            amount_at_risk=case.amount_at_risk,
        )

        # Re-check immediately before the money moves. The decision above was made
        # against a snapshot; this is the call that refuses to act on a case which
        # closed in between.
        try:
            case.guard_actionable()
        except IllegalTransition as exc:
            self.log.append(
                EntryKind.ACTION_DECLINED,
                case_id=case.case_id,
                at=now,
                action=action,
                reason=f"guard refused execution: {exc}",
            )
            return

        key = case.idempotency_key_for(action)
        result = self.gateway.execute(case, action, at=now, idempotency_key=key)

        case.record_attempt(
            Attempt(
                attempted_at=now,
                action=action,
                idempotency_key=key,
                succeeded=result.succeeded,
                cost=result.cost,
                note=result.detail,
            )
        )
        self._stats = BatchStats(
            actions_executed=self._stats.actions_executed + 1,
            actions_failed=self._stats.actions_failed + (0 if result.succeeded else 1),
        )
        self.log.append(
            EntryKind.ACTION_EXECUTED,
            case_id=case.case_id,
            at=now,
            action=action,
            idempotency_key=key,
            replayed=result.replayed,
        )
        self.log.append(
            EntryKind.ACTION_RESULT,
            case_id=case.case_id,
            at=now,
            action=action,
            succeeded=result.succeeded,
            cost=result.cost,
            gateway_ref=result.gateway_ref,
            detail=result.detail,
        )

        case.transition(CaseState.ACTION_EXECUTED, at=now, reason=result.detail)

        if result.succeeded:
            self._close(case, CaseState.RECOVERED, at=now, reason=f"{action} succeeded")
            return

        if result.new_failure is not None:
            case.history.append(result.new_failure)

        # Re-enter the decision at the same instant. The retry-spacing rule owns
        # the delay, not this loop.
        self._schedule(case.case_id, now)

    # -- closing -----------------------------------------------------------

    def _close(
        self, case: RecoveryCase, state: CaseState, *, at: datetime, reason: str
    ) -> None:
        case.transition(state, at=at, reason=reason)
        self.log.append(
            EntryKind.STATE_TRANSITION,
            case_id=case.case_id,
            at=at,
            to=state,
            reason=reason,
            attempts_used=case.attempts_used,
            amount=case.amount_at_risk,
        )

    def _finalise(self, horizon_end: datetime) -> None:
        """Resolve unprompted recoveries, then close out the holdout.

        Control-arm cases never leave the loop with a terminal state, and treatment
        cases that stopped early are no longer being woken. Both can still be paid
        by the customer before the horizon closes, and the control arm is worthless
        unless that is counted.

        Order matters: self-recovery is applied first, and only cases still open
        afterwards are marked HELD_OUT. Closing the holdout before checking would
        discard exactly the outcomes it exists to measure.
        """
        for case in self._cases.values():
            if case.is_terminal:
                continue
            recovered_at = self._self_recovery.get(case.case_id)
            if recovered_at is not None and recovered_at <= horizon_end:
                self._close(
                    case, CaseState.RECOVERED, at=recovered_at, reason="self_recovery"
                )
                continue
            if case.arm is Arm.CONTROL:
                self._close(
                    case,
                    CaseState.HELD_OUT,
                    at=horizon_end,
                    reason="control arm; never actioned by design",
                )
        # Treatment cases still open here are genuinely unresolved at the horizon —
        # mostly deferrals whose window falls outside it. They are left in their
        # working state rather than forced closed, because inventing a terminal
        # state for them would report a conclusion the run never reached.
