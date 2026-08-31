"""Domain tests. The transition and guard tests are the load-bearing ones: they
encode the claim that this system cannot charge a customer on a case that has
already closed.
"""

import unittest
from datetime import datetime, timedelta, timezone

from recoup.domain.case import (
    ActionKind,
    Arm,
    Attempt,
    CaseState,
    IllegalTransition,
    RecoveryCase,
)
from recoup.domain.events import FailureEvent, Surface
from recoup.domain.money import Money
from recoup.domain.taxonomy import ErrorReason, ErrorSource, ErrorStep, PaymentMethod

NOW = datetime(2026, 8, 21, 9, 30, tzinfo=timezone.utc)


def make_event(**overrides) -> FailureEvent:
    defaults = dict(
        event_id="pay_TEST0001",
        occurred_at=NOW,
        surface=Surface.PAYMENT,
        entity_id="pay_TEST0001",
        customer_id="cust_001",
        amount=Money.from_rupees("499"),
        method=PaymentMethod.CARD,
        error_source=ErrorSource.BANK,
        error_step=ErrorStep.AUTHORIZATION,
        error_reason=ErrorReason.INSUFFICIENT_FUNDS,
    )
    defaults.update(overrides)
    return FailureEvent(**defaults)


class TestFailureEvent(unittest.TestCase):
    def test_naive_datetime_is_refused(self):
        with self.assertRaises(ValueError):
            make_event(occurred_at=datetime(2026, 8, 21, 9, 30))

    def test_attempt_number_is_one_based(self):
        with self.assertRaises(ValueError):
            make_event(attempt_number=0)

    def test_dedupe_key_ignores_transport_identity(self):
        # Same failure delivered twice with different envelope ids.
        a = make_event(event_id="evt_aaa")
        b = make_event(event_id="evt_bbb")
        self.assertEqual(a.dedupe_key, b.dedupe_key)

    def test_dedupe_key_separates_distinct_attempts(self):
        a = make_event(attempt_number=1)
        b = make_event(attempt_number=2)
        self.assertNotEqual(a.dedupe_key, b.dedupe_key)

    def test_rail_switch_reasons_are_flagged(self):
        self.assertTrue(make_event(error_reason=ErrorReason.EXPIRED_CARD).is_terminal_for_instrument)
        self.assertFalse(make_event(error_reason=ErrorReason.INCORRECT_OTP).is_terminal_for_instrument)


class TestRazorpayNormalisation(unittest.TestCase):
    def test_amount_is_read_as_paise(self):
        event = FailureEvent.from_razorpay(
            {"id": "pay_X", "amount": 49900, "currency": "INR", "method": "card"},
            surface=Surface.PAYMENT,
        )
        # 49900 paise is ₹499, not ₹49,900.
        self.assertEqual(event.amount, Money.from_rupees("499"))

    def test_unknown_reason_degrades_rather_than_raising(self):
        event = FailureEvent.from_razorpay(
            {
                "id": "pay_X",
                "amount": 100,
                "error": {"reason": "some_code_invented_next_quarter", "source": "bank"},
            },
            surface=Surface.PAYMENT,
        )
        self.assertEqual(event.error_reason, ErrorReason.UNKNOWN)
        self.assertEqual(event.error_source, ErrorSource.BANK)

    def test_missing_error_block_is_tolerated(self):
        event = FailureEvent.from_razorpay({"id": "pay_X", "amount": 0}, surface=Surface.PAYMENT)
        self.assertEqual(event.error_reason, ErrorReason.UNKNOWN)
        self.assertEqual(event.method, PaymentMethod.UNKNOWN)


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.case = RecoveryCase.open_from(make_event())

    def test_happy_path(self):
        for state in (
            CaseState.DIAGNOSED,
            CaseState.ACTION_SCHEDULED,
            CaseState.ACTION_EXECUTED,
            CaseState.RECOVERED,
        ):
            self.case.transition(state, at=NOW)
        self.assertTrue(self.case.is_terminal)
        self.assertEqual(self.case.closed_at, NOW)

    def test_undeclared_transition_is_refused(self):
        with self.assertRaises(IllegalTransition):
            self.case.transition(CaseState.ACTION_EXECUTED, at=NOW)

    def test_terminal_state_cannot_be_left(self):
        self.case.transition(CaseState.DIAGNOSED, at=NOW)
        self.case.transition(CaseState.UNECONOMIC, at=NOW)
        with self.assertRaises(IllegalTransition):
            self.case.transition(CaseState.ACTION_SCHEDULED, at=NOW)

    def test_customer_can_self_recover_while_we_wait(self):
        self.case.transition(CaseState.DIAGNOSED, at=NOW)
        self.case.transition(CaseState.AWAITING_WINDOW, at=NOW)
        self.case.transition(CaseState.RECOVERED, at=NOW)
        self.assertEqual(self.case.state, CaseState.RECOVERED)


class TestActionGuards(unittest.TestCase):
    def test_guard_blocks_a_closed_case(self):
        case = RecoveryCase.open_from(make_event())
        case.transition(CaseState.DIAGNOSED, at=NOW)
        case.transition(CaseState.ACTION_SCHEDULED, at=NOW)
        # The webhook lands between scheduling and execution.
        case.transition(CaseState.RECOVERED, at=NOW)
        with self.assertRaises(IllegalTransition):
            case.guard_actionable()

    def test_guard_blocks_control_arm(self):
        case = RecoveryCase.open_from(make_event(), arm=Arm.CONTROL)
        case.transition(CaseState.DIAGNOSED, at=NOW)
        with self.assertRaises(IllegalTransition):
            case.guard_actionable()

    def test_guard_allows_a_live_treatment_case(self):
        case = RecoveryCase.open_from(make_event())
        case.transition(CaseState.DIAGNOSED, at=NOW)
        case.guard_actionable()  # must not raise


class TestIdempotency(unittest.TestCase):
    def test_key_is_stable_for_the_same_decision(self):
        case = RecoveryCase.open_from(make_event())
        first = case.idempotency_key_for(ActionKind.RETRY_SAME_RAIL)
        second = case.idempotency_key_for(ActionKind.RETRY_SAME_RAIL)
        self.assertEqual(first, second)

    def test_key_changes_once_an_attempt_is_recorded(self):
        case = RecoveryCase.open_from(make_event())
        before = case.idempotency_key_for(ActionKind.RETRY_SAME_RAIL)
        case.record_attempt(
            Attempt(attempted_at=NOW, action=ActionKind.RETRY_SAME_RAIL, idempotency_key=before)
        )
        self.assertNotEqual(before, case.idempotency_key_for(ActionKind.RETRY_SAME_RAIL))

    def test_key_differs_by_action(self):
        case = RecoveryCase.open_from(make_event())
        self.assertNotEqual(
            case.idempotency_key_for(ActionKind.RETRY_SAME_RAIL),
            case.idempotency_key_for(ActionKind.SEND_PAYMENT_LINK),
        )


class TestContactBudget(unittest.TestCase):
    def test_silent_retry_does_not_consume_contact_budget(self):
        case = RecoveryCase.open_from(make_event())
        case.record_attempt(
            Attempt(attempted_at=NOW, action=ActionKind.RETRY_SAME_RAIL, idempotency_key="k")
        )
        self.assertEqual(case.attempts_used, 1)
        self.assertEqual(case.contacts_in_last(timedelta(days=7), now=NOW), 0)

    def test_payment_link_does_consume_it(self):
        case = RecoveryCase.open_from(make_event())
        case.record_attempt(
            Attempt(attempted_at=NOW, action=ActionKind.SEND_PAYMENT_LINK, idempotency_key="k")
        )
        self.assertEqual(case.contacts_in_last(timedelta(days=7), now=NOW), 1)

    def test_contacts_outside_the_window_are_excluded(self):
        case = RecoveryCase.open_from(make_event())
        case.record_attempt(
            Attempt(
                attempted_at=NOW - timedelta(days=10),
                action=ActionKind.SEND_PAYMENT_LINK,
                idempotency_key="k",
            )
        )
        self.assertEqual(case.contacts_in_last(timedelta(days=7), now=NOW), 0)

    def test_a_contact_exactly_one_window_old_has_aged_out(self):
        """The boundary that broke the contact cap.

        `rule_contact_frequency` defers to `oldest_contact + 7d`. If a contact
        exactly seven days old still counted, the rule would block again at the
        precise instant it had asked to be re-checked at, and the orchestrator would
        escalate the case for a deferral that never advanced. The interval has to be
        half-open in the same direction the rule's arithmetic assumes.
        """
        case = RecoveryCase.open_from(make_event())
        case.record_attempt(
            Attempt(
                attempted_at=NOW - timedelta(days=7),
                action=ActionKind.SEND_PAYMENT_LINK,
                idempotency_key="k",
            )
        )
        self.assertEqual(case.contacts_in_last(timedelta(days=7), now=NOW), 0)
        self.assertEqual(
            case.contacts_in_last(timedelta(days=7), now=NOW - timedelta(seconds=1)),
            1,
        )

    def test_a_contact_made_right_now_still_counts(self):
        case = RecoveryCase.open_from(make_event())
        case.record_attempt(
            Attempt(
                attempted_at=NOW,
                action=ActionKind.SEND_PAYMENT_LINK,
                idempotency_key="k",
            )
        )
        self.assertEqual(case.contacts_in_last(timedelta(days=7), now=NOW), 1)


class TestAccounting(unittest.TestCase):
    def test_nothing_is_recovered_until_the_case_says_so(self):
        case = RecoveryCase.open_from(make_event())
        self.assertEqual(case.recovered_amount(), Money.zero())
        case.transition(CaseState.DIAGNOSED, at=NOW)
        case.transition(CaseState.ACTION_SCHEDULED, at=NOW)
        case.transition(CaseState.RECOVERED, at=NOW)
        self.assertEqual(case.recovered_amount(), Money.from_rupees("499"))

    def test_costs_accumulate_across_attempts(self):
        case = RecoveryCase.open_from(make_event())
        for _ in range(3):
            case.record_attempt(
                Attempt(
                    attempted_at=NOW,
                    action=ActionKind.RETRY_SAME_RAIL,
                    idempotency_key="k",
                    cost=Money.from_rupees("2.50"),
                )
            )
        self.assertEqual(case.total_cost(), Money.from_rupees("7.50"))


if __name__ == "__main__":
    unittest.main()
