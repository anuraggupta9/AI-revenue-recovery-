"""Diagnosis and policy tests.

The precedence tests matter most. Individual rules are easy; what is easy to get
wrong is what happens when two of them disagree, and those interactions are where
a compliance claim quietly becomes false.
"""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from recoup.diagnosis import RootCause, diagnose
from recoup.domain.case import ActionKind, Arm, Attempt, RecoveryCase
from recoup.domain.events import FailureEvent, Surface
from recoup.domain.money import Money
from recoup.domain.taxonomy import ErrorReason, ErrorSource, ErrorStep, PaymentMethod
from recoup.policy import (
    BatchStats,
    DecisionContext,
    Outcome,
    PolicyConfig,
    decide,
    expected_value,
)
from recoup.policy.timing import IST, is_within_contact_hours, next_salary_window, to_ist

# 2026-08-21 09:30 UTC is 15:00 IST — inside contact hours, mid-month.
NOON_IST = datetime(2026, 8, 21, 9, 30, tzinfo=timezone.utc)


def make_event(**overrides) -> FailureEvent:
    defaults = dict(
        event_id="pay_1",
        occurred_at=NOON_IST,
        surface=Surface.PAYMENT,
        entity_id="pay_1",
        customer_id="cust_1",
        amount=Money.from_rupees("2000"),
        method=PaymentMethod.CARD,
        error_source=ErrorSource.BANK,
        error_step=ErrorStep.AUTHORIZATION,
        error_reason=ErrorReason.INSUFFICIENT_FUNDS,
    )
    defaults.update(overrides)
    return FailureEvent(**defaults)


def make_ctx(event: FailureEvent | None = None, **overrides) -> DecisionContext:
    event = event or make_event()
    case = overrides.pop("case", None) or RecoveryCase.open_from(event)
    defaults = dict(
        case=case,
        diagnosis=diagnose(event),
        now=NOON_IST,
        config=PolicyConfig(),
        p_success=Decimal("0.40"),
    )
    defaults.update(overrides)
    return DecisionContext(**defaults)


class TestDiagnosis(unittest.TestCase):
    def test_known_reason_maps_confidently(self):
        d = diagnose(make_event(error_reason=ErrorReason.INCORRECT_OTP))
        self.assertEqual(d.root_cause, RootCause.AUTH_FRICTION)
        self.assertTrue(d.is_confident)
        self.assertEqual(d.source, "taxonomy")

    def test_expired_card_prefers_a_rail_switch(self):
        d = diagnose(make_event(error_reason=ErrorReason.EXPIRED_CARD))
        self.assertEqual(d.candidate_actions[0], ActionKind.RETRY_ALTERNATE_RAIL)

    def test_risk_decline_offers_no_action(self):
        d = diagnose(make_event(error_reason=ErrorReason.RISK_DECLINED))
        self.assertEqual(d.root_cause, RootCause.RISK_BLOCKED)
        self.assertEqual(d.candidate_actions, ())
        self.assertFalse(d.is_actionable)

    def test_unknown_reason_falls_back_to_source_and_step(self):
        d = diagnose(
            make_event(
                error_reason=ErrorReason.UNKNOWN,
                error_source=ErrorSource.GATEWAY,
                error_step=ErrorStep.AUTHORIZATION,
            )
        )
        self.assertEqual(d.root_cause, RootCause.GATEWAY_ROUTING)
        self.assertEqual(d.source, "taxonomy_fallback")

    def test_wholly_unmapped_failure_abstains(self):
        d = diagnose(
            make_event(
                error_reason=ErrorReason.UNKNOWN,
                error_source=ErrorSource.UNKNOWN,
                error_step=ErrorStep.UNKNOWN,
            )
        )
        self.assertEqual(d.root_cause, RootCause.UNKNOWN)
        self.assertFalse(d.is_confident)
        self.assertEqual(d.source, "abstain")

    def test_diagnosis_is_deterministic(self):
        event = make_event()
        self.assertEqual(diagnose(event), diagnose(event))


class TestTiming(unittest.TestCase):
    def test_ist_offset(self):
        self.assertEqual(to_ist(NOON_IST).hour, 15)

    def test_contact_hours_boundaries(self):
        # 03:00 UTC is 08:30 IST — before the window opens.
        self.assertFalse(is_within_contact_hours(datetime(2026, 8, 21, 3, 0, tzinfo=timezone.utc)))
        # 04:00 UTC is 09:30 IST — inside.
        self.assertTrue(is_within_contact_hours(datetime(2026, 8, 21, 4, 0, tzinfo=timezone.utc)))
        # 14:00 UTC is 19:30 IST — after it closes.
        self.assertFalse(is_within_contact_hours(datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc)))

    def test_naive_input_is_refused(self):
        with self.assertRaises(ValueError):
            to_ist(datetime(2026, 8, 21, 9, 30))

    def test_salary_window_lands_on_the_first(self):
        window = next_salary_window(NOON_IST)
        self.assertIn(window.day, (1, 2))
        self.assertEqual(window.month, 9)
        self.assertEqual(window.hour, 10)

    def test_salary_window_respects_a_later_cooloff(self):
        # A cool-off pushing past the 1st must land on the following month.
        ready = datetime(2026, 9, 3, 0, 0, tzinfo=IST)
        window = next_salary_window(NOON_IST, not_before=ready)
        self.assertEqual((window.month, window.day), (10, 1))

    def test_month_boundary_is_handled(self):
        # 31 January: the next window is 1 February, not 32 January.
        window = next_salary_window(datetime(2026, 1, 31, 12, 0, tzinfo=IST))
        self.assertEqual((window.month, window.day), (2, 1))


class TestExpectedValue(unittest.TestCase):
    def test_silent_retry_beats_a_message_at_equal_odds(self):
        ctx = make_ctx()
        silent = expected_value(ctx, ActionKind.RETRY_SAME_RAIL)
        contact = expected_value(ctx, ActionKind.SEND_PAYMENT_LINK)
        self.assertGreater(silent, contact)

    def test_margin_is_applied(self):
        ctx = make_ctx(p_success=Decimal("1.0"))
        # ₹2000 x 1.0 x 0.85 margin, less ₹2 retry cost.
        self.assertEqual(
            expected_value(ctx, ActionKind.RETRY_SAME_RAIL), Money.from_rupees("1698")
        )

    def test_low_probability_yields_negative_value(self):
        ctx = make_ctx(p_success=Decimal("0.001"))
        self.assertLess(expected_value(ctx, ActionKind.RETRY_SAME_RAIL), Money.zero())


class TestGlobalRulePrecedence(unittest.TestCase):
    def test_circuit_breaker_outranks_everything(self):
        ctx = make_ctx(
            batch=BatchStats(actions_executed=50, actions_failed=49),
            p_success=Decimal("0.99"),
        )
        decision = decide(ctx)
        self.assertEqual(decision.outcome, Outcome.HALT)
        self.assertIn("circuit_breaker", decision.rationale)

    def test_breaker_tolerates_an_ordinary_bad_batch(self):
        """Most recovery attempts fail. That is not an emergency.

        Pinned deliberately: the breaker was originally set at a 60% failure rate,
        which fires on a perfectly healthy batch and would have halted the demo
        run. A breaker that cries wolf during normal operation is worse than none,
        so this asserts the threshold still distinguishes "poor odds" from "broken".
        """
        ctx = make_ctx(batch=BatchStats(actions_executed=100, actions_failed=75))
        self.assertNotEqual(decide(ctx).outcome, Outcome.HALT)

    def test_breaker_stays_quiet_on_a_small_sample(self):
        # 100% failure but only 5 actions — not enough to conclude anything.
        ctx = make_ctx(batch=BatchStats(actions_executed=5, actions_failed=5))
        self.assertNotEqual(decide(ctx).outcome, Outcome.HALT)

    def test_opt_out_blocks_even_a_profitable_action(self):
        ctx = make_ctx(opted_out=True, p_success=Decimal("0.99"))
        decision = decide(ctx)
        self.assertEqual(decision.outcome, Outcome.STOP_SUPPRESSED)
        self.assertIn("opt", decision.rationale)

    def test_risk_decline_is_never_retried_however_valuable(self):
        """Escalated, not suppressed, and on half a million rupees the difference bites.

        Both outcomes stop the agent, so an earlier version of this test accepted
        STOP_SUPPRESSED and was satisfied. But suppression is the terminal state
        opt-outs land in, and nobody reviews that queue — a suspected-fraud decline
        filed there is a fraud signal the merchant never sees. The assertion is on
        the stronger property: the agent does not act, *and* a human is told.
        """
        event = make_event(
            error_reason=ErrorReason.RISK_DECLINED, amount=Money.from_rupees("500000")
        )
        decision = decide(make_ctx(event, p_success=Decimal("0.99")))
        self.assertEqual(decision.outcome, Outcome.ESCALATE)
        self.assertIsNone(decision.action)
        self.assertIn("categorical no-retry list", decision.rationale)

    def test_low_confidence_escalates_rather_than_guessing(self):
        event = make_event(
            error_reason=ErrorReason.UNKNOWN,
            error_source=ErrorSource.UNKNOWN,
            error_step=ErrorStep.UNKNOWN,
        )
        self.assertEqual(decide(make_ctx(event)).outcome, Outcome.ESCALATE)

    def test_attempt_cap_exhausts_the_case(self):
        case = RecoveryCase.open_from(make_event())
        for _ in range(3):
            case.record_attempt(
                Attempt(
                    attempted_at=NOON_IST,
                    action=ActionKind.RETRY_SAME_RAIL,
                    idempotency_key="k",
                )
            )
        decision = decide(make_ctx(case=case))
        self.assertEqual(decision.outcome, Outcome.STOP_EXHAUSTED)

    def test_every_global_rule_is_logged_even_when_passing(self):
        # An OTP failure clears all five global rules and acts.
        decision = decide(make_ctx(make_event(error_reason=ErrorReason.INCORRECT_OTP)))
        names = [o.rule for o in decision.rule_outcomes]
        for rule in (
            "circuit_breaker",
            "customer_opt_out",
            "never_auto_retry",
            "confidence_floor",
            "attempt_cap",
        ):
            self.assertIn(rule, names)
        self.assertTrue(all(o.passed for o in decision.rule_outcomes if o.rule == "attempt_cap"))


class TestActionSelection(unittest.TestCase):
    def test_otp_failure_retries_the_same_rail_immediately(self):
        decision = decide(make_ctx(make_event(error_reason=ErrorReason.INCORRECT_OTP)))
        self.assertEqual(decision.outcome, Outcome.ACT)
        self.assertEqual(decision.action, ActionKind.RETRY_SAME_RAIL)

    def test_expired_card_switches_rail_immediately(self):
        decision = decide(make_ctx(make_event(error_reason=ErrorReason.EXPIRED_CARD)))
        self.assertEqual(decision.outcome, Outcome.ACT)
        self.assertEqual(decision.action, ActionKind.RETRY_ALTERNATE_RAIL)

    def test_balance_failure_defers_to_the_salary_window(self):
        # The headline behavioural claim: no fixed 24h ladder.
        decision = decide(make_ctx(make_event(error_reason=ErrorReason.INSUFFICIENT_FUNDS)))
        self.assertEqual(decision.outcome, Outcome.DEFER)
        self.assertIsNotNone(decision.execute_at)
        self.assertIn(to_ist(decision.execute_at).day, (1, 2))

    def test_downtime_defers_a_same_rail_retry(self):
        until = NOON_IST + timedelta(hours=4)
        decision = decide(
            make_ctx(make_event(error_reason=ErrorReason.ISSUER_DOWN), downtime_until=until)
        )
        # ISSUER_DOWN offers same-rail first, then alternate. Same-rail is blocked
        # by downtime, so the alternate rail should be chosen instead of waiting.
        self.assertEqual(decision.outcome, Outcome.ACT)
        self.assertEqual(decision.action, ActionKind.RETRY_ALTERNATE_RAIL)
        self.assertTrue(decision.declined)

    def test_declined_actions_are_recorded_with_reasons(self):
        until = NOON_IST + timedelta(hours=4)
        decision = decide(
            make_ctx(make_event(error_reason=ErrorReason.ISSUER_DOWN), downtime_until=until)
        )
        declined_actions = [a for a, _ in decision.declined]
        self.assertIn(ActionKind.RETRY_SAME_RAIL, declined_actions)
        self.assertIn("issuer_downtime", decision.declined[0][1])

    def test_uneconomic_case_stops(self):
        tiny = make_event(error_reason=ErrorReason.INCORRECT_OTP, amount=Money.from_rupees("5"))
        decision = decide(make_ctx(tiny, p_success=Decimal("0.05")))
        self.assertEqual(decision.outcome, Outcome.STOP_UNECONOMIC)

    def test_a_categorical_prohibition_escalates_before_action_selection(self):
        """The name of this test used to promise escalation and it asserted suppression.

        MANDATE_REVOKED is on the never-auto-retry list, so `rule_never_auto_retry`
        fires in the global stage and action selection is never reached — meaning
        this test never touched the branch it appeared to be about. It is kept for
        the precedence claim it does make, and the empty-candidate branch is covered
        separately below.
        """
        decision = decide(make_ctx(make_event(error_reason=ErrorReason.MANDATE_REVOKED)))
        self.assertEqual(decision.outcome, Outcome.ESCALATE)
        self.assertIn("never_auto_retry", decision.rationale)
        self.assertEqual(len(decision.declined), 0)

    def test_a_diagnosis_with_no_automated_action_escalates(self):
        """The branch the test above only looked like it was covering.

        Reached by handing `decide()` a diagnosis with an empty candidate set
        directly, because every error reason that currently produces one is also on
        a categorical list and gets stopped a stage earlier.
        """
        confident_but_actionless = replace(
            diagnose(make_event()),
            root_cause=RootCause.AUTH_FRICTION,
            candidate_actions=(),
        )
        decision = decide(make_ctx(diagnosis=confident_but_actionless))
        self.assertEqual(decision.outcome, Outcome.ESCALATE)
        self.assertIn("human queue", decision.rationale)
        self.assertIsNone(decision.action)


class TestQuietHoursAndContactCap(unittest.TestCase):
    def test_quiet_hours_defers_a_customer_message(self):
        # 23:00 IST.
        late = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
        event = make_event(error_reason=ErrorReason.INVALID_CVV, occurred_at=late)
        decision = decide(make_ctx(event, now=late))
        self.assertEqual(decision.outcome, Outcome.DEFER)
        self.assertEqual(to_ist(decision.execute_at).hour, 9)

    def test_quiet_hours_does_not_block_a_silent_retry(self):
        late = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
        event = make_event(error_reason=ErrorReason.INCORRECT_OTP, occurred_at=late)
        decision = decide(make_ctx(event, now=late))
        self.assertEqual(decision.outcome, Outcome.ACT)
        self.assertEqual(decision.action, ActionKind.RETRY_SAME_RAIL)

    def test_contact_cap_defers_further_messages(self):
        event = make_event(error_reason=ErrorReason.INVALID_CVV)
        case = RecoveryCase.open_from(event)
        for offset in (1, 2):
            case.record_attempt(
                Attempt(
                    attempted_at=NOON_IST - timedelta(days=offset),
                    action=ActionKind.SEND_PAYMENT_LINK,
                    idempotency_key=f"k{offset}",
                )
            )
        decision = decide(make_ctx(event, case=case))
        self.assertEqual(decision.outcome, Outcome.DEFER)
        self.assertIsNotNone(decision.execute_at)

    def test_deferrals_compose_by_taking_the_latest(self):
        # Quiet hours (until 09:00 IST tomorrow) plus a contact cap that clears
        # in six days. The later constraint must govern.
        late = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
        event = make_event(error_reason=ErrorReason.INVALID_CVV, occurred_at=late)
        case = RecoveryCase.open_from(event)
        for offset in (0, 1):
            case.record_attempt(
                Attempt(
                    attempted_at=late - timedelta(days=offset),
                    action=ActionKind.SEND_PAYMENT_LINK,
                    idempotency_key=f"k{offset}",
                )
            )
        decision = decide(make_ctx(event, case=case, now=late))
        self.assertEqual(decision.outcome, Outcome.DEFER)
        # Six days out, not tomorrow morning.
        self.assertGreater(decision.execute_at, late + timedelta(days=5))


class TestPurity(unittest.TestCase):
    def test_decide_does_not_mutate_the_case(self):
        ctx = make_ctx()
        before = (ctx.case.state, ctx.case.attempts_used, ctx.case.updated_at)
        decide(ctx)
        self.assertEqual(before, (ctx.case.state, ctx.case.attempts_used, ctx.case.updated_at))

    def test_control_arm_reaches_the_same_decision(self):
        # Shadow mode depends on this: the control arm must run the identical
        # decision path so the comparison is apples to apples.
        event = make_event(error_reason=ErrorReason.INCORRECT_OTP)
        treatment = decide(make_ctx(event, case=RecoveryCase.open_from(event, arm=Arm.TREATMENT)))
        control = decide(make_ctx(event, case=RecoveryCase.open_from(event, arm=Arm.CONTROL)))
        self.assertEqual(treatment.outcome, control.outcome)
        self.assertEqual(treatment.action, control.action)


if __name__ == "__main__":
    unittest.main()
