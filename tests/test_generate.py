"""Tests for the synthetic batch.

The generator is the weakest link in the whole submission: it is the world, and I
wrote it, so every result downstream inherits its mistakes. These tests are about
the one class of mistake that is objectively checkable — a failure that could not
physically have happened on the rail it is attached to.

The plausibility table below is written out independently of the generator's own
weight tables rather than imported from them. Deriving the expectation from the code
under test would make this a tautology: it would pass no matter which reasons the
generator emitted, which is exactly the failure it exists to catch.
"""

from __future__ import annotations

import dataclasses
import unittest
from collections import Counter

from recoup.domain.events import Surface
from recoup.domain.money import Money
from recoup.domain.taxonomy import ErrorReason as Reason
from recoup.domain.taxonomy import PaymentMethod as Method
from recoup.simulation.generate import DEFAULT_START, SMALL_TICKET_BUCKETS, generate

# Rail-independent: anything between the merchant and the bank can break on any
# rail, and any code can fail to parse.
_INFRASTRUCTURE = {
    Reason.GATEWAY_TECHNICAL_ERROR,
    Reason.ISSUER_DOWN,
    Reason.NETWORK_ERROR,
    Reason.RISK_DECLINED,
    Reason.SUSPECTED_FRAUD,
    Reason.UNKNOWN,
}

# What a customer sitting at a checkout can do wrong, per rail. UPI is the
# interesting one: it authenticates with a PIN in the payer's own app, so no OTP
# reason belongs to it, and a VPA has no CVV and no expiry date.
_CHECKOUT = {
    Method.CARD: {
        Reason.INSUFFICIENT_FUNDS,
        Reason.INCORRECT_OTP,
        Reason.OTP_NOT_ENTERED,
        Reason.INVALID_CVV,
        Reason.EXPIRED_CARD,
        Reason.CARD_DECLINED,
        Reason.INTERNATIONAL_NOT_ALLOWED,
        Reason.PAYMENT_TIMEOUT,
    },
    Method.UPI: {
        Reason.INSUFFICIENT_FUNDS,
        Reason.INVALID_VPA,
        Reason.UPI_COLLECT_EXPIRED,
        Reason.PAYMENT_TIMEOUT,
    },
    Method.NETBANKING: {
        Reason.INSUFFICIENT_FUNDS,
        Reason.INCORRECT_OTP,
        Reason.OTP_NOT_ENTERED,
        Reason.PAYMENT_TIMEOUT,
    },
    Method.WALLET: {
        Reason.INSUFFICIENT_FUNDS,
        Reason.INCORRECT_OTP,
        Reason.OTP_NOT_ENTERED,
        Reason.PAYMENT_TIMEOUT,
    },
    Method.EMI: {
        Reason.INSUFFICIENT_FUNDS,
        Reason.INCORRECT_OTP,
        Reason.OTP_NOT_ENTERED,
        Reason.INVALID_CVV,
        Reason.EXPIRED_CARD,
        Reason.CARD_DECLINED,
        Reason.PAYMENT_TIMEOUT,
    },
}

# A recurring debit has nobody at the keyboard, so the customer-input family is
# unavailable and the mandate's own state becomes a failure mode.
_MANDATE = {Reason.MANDATE_INSUFFICIENT_BALANCE, Reason.MANDATE_REVOKED}

# ...with one exception: RBI requires an additional authentication factor on
# recurring card debits above this amount, which does put an OTP in front of
# someone. Below it, an OTP reason on a mandate charge is a bug.
_AFA_THRESHOLD = Money.from_rupees("15000")
_AFA_ONLY = {Reason.INCORRECT_OTP, Reason.OTP_NOT_ENTERED}
_AFA_METHODS = {Method.CARD, Method.EMI}

_MANDATE_ONLY = _MANDATE
_CHECKOUT_ONLY = {
    Reason.INSUFFICIENT_FUNDS,
    Reason.INVALID_CVV,
    Reason.INVALID_VPA,
    Reason.EXPIRED_CARD,
    Reason.CARD_DECLINED,
    Reason.INTERNATIONAL_NOT_ALLOWED,
    Reason.UPI_COLLECT_EXPIRED,
    Reason.PAYMENT_TIMEOUT,
}


def _permitted(method: Method, surface: Surface, amount: Money) -> set[Reason]:
    if surface is not Surface.SUBSCRIPTION_CHARGE:
        return _CHECKOUT[method] | _INFRASTRUCTURE
    allowed = _MANDATE | _INFRASTRUCTURE
    if method in _AFA_METHODS and amount > _AFA_THRESHOLD:
        allowed |= _AFA_ONLY
    return allowed


class GeneratedFailuresArePhysicallyPossible(unittest.TestCase):
    """The batch used for every published number, checked event by event.

    An earlier version drew reason and method independently, so a quarter of the
    batch was impossible: `invalid_vpa` on cards, `expired_card` on UPI,
    `incorrect_otp` on UPI. Nothing failed, and no metric could see it — an
    impossible combination is a free feature for the estimator to fit, because no
    real distribution generates it and so nothing contradicts whatever it learns.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.batch = generate(count=2000, seed=20260821, horizon_days=45)

    def test_no_reason_is_impossible_on_its_rail(self) -> None:
        offenders = Counter(
            (event.method.value, event.surface.value, event.error_reason.value)
            for event in self.batch.events
            if event.error_reason
            not in _permitted(event.method, event.surface, event.amount)
        )
        self.assertEqual(
            offenders,
            Counter(),
            "generated failures that could not have happened on that rail: "
            f"{offenders.most_common()}",
        )

    def test_mandate_reasons_appear_only_on_subscription_charges(self) -> None:
        for event in self.batch.events:
            if event.error_reason in _MANDATE_ONLY:
                self.assertIs(
                    event.surface,
                    Surface.SUBSCRIPTION_CHARGE,
                    f"{event.error_reason} on a one-time payment",
                )

    def test_checkout_reasons_never_appear_on_a_recurring_debit(self) -> None:
        """The converse, and the one that catches a lazily reused table.

        `insufficient_funds` is in here deliberately: a mandate charge that fails
        for balance reports the mandate-specific code, and quietly reporting the
        one-off code instead would make the two surfaces indistinguishable to the
        diagnosis layer.
        """
        for event in self.batch.events:
            if event.surface is Surface.SUBSCRIPTION_CHARGE:
                self.assertNotIn(event.error_reason, _CHECKOUT_ONLY)

    def test_additional_factor_reasons_stay_above_the_threshold(self) -> None:
        for event in self.batch.events:
            if (
                event.surface is Surface.SUBSCRIPTION_CHARGE
                and event.error_reason in _AFA_ONLY
            ):
                self.assertIn(event.method, _AFA_METHODS)
                self.assertGreater(event.amount, _AFA_THRESHOLD)

    def test_every_prohibited_reason_is_actually_exercised(self) -> None:
        """The three categorical prohibitions must appear in the batch.

        `mandate_revoked` used to be absent: it was on the never-auto-retry list
        and mapped in the taxonomy, but no weight table emitted it, so the
        prohibition was only ever tested by unit tests that constructed the event
        by hand. A safety rule that the reference run never triggers is a safety
        rule nobody has seen work.
        """
        present = {event.error_reason for event in self.batch.events}
        for reason in (Reason.RISK_DECLINED, Reason.SUSPECTED_FRAUD, Reason.MANDATE_REVOKED):
            self.assertIn(reason, present)

    def test_unknown_reasons_reach_the_escalation_path(self) -> None:
        unknown = sum(
            1 for event in self.batch.events if event.error_reason is Reason.UNKNOWN
        )
        self.assertGreater(unknown, 0, "nothing in the batch exercises the abstain path")


class BatchIsReproducible(unittest.TestCase):
    def test_same_seed_gives_an_identical_batch(self) -> None:
        left = generate(count=120, seed=7, horizon_days=30)
        right = generate(count=120, seed=7, horizon_days=30)
        self.assertEqual(left.events, right.events)
        self.assertEqual(left.downtime, right.downtime)

    def test_a_different_seed_gives_a_different_batch(self) -> None:
        left = generate(count=120, seed=7, horizon_days=30)
        right = generate(count=120, seed=8, horizon_days=30)
        self.assertNotEqual(left.events, right.events)

    def test_extending_a_batch_leaves_the_earlier_records_alone(self) -> None:
        """The property that makes the seed stream per-index rather than global.

        Without it, raising `count` to get a tighter confidence interval would
        silently change every case in the batch, and two runs at different sizes
        would not be comparable at all.

        `customer_id` is the one documented exception: the customer pool scales with
        the batch on purpose, so that a larger merchant has proportionally more
        customers instead of the same few failing more often. It is excluded here
        and pinned by the next two tests.
        """
        short = generate(count=100, seed=7, horizon_days=30)
        long = generate(count=200, seed=7, horizon_days=30)
        by_id = {event.event_id: event for event in long.events}
        for event in short.events:
            grown = by_id[event.event_id]
            self.assertEqual(
                dataclasses.replace(event, customer_id=""),
                dataclasses.replace(grown, customer_id=""),
            )

    def test_customer_density_does_not_drift_with_batch_size(self) -> None:
        """Why the exception above is worth having.

        The contact-frequency cap is per customer per seven days. If failures per
        customer grew with `count`, the cap would bind more often in a larger batch
        and the policy would behave differently at a sample size chosen purely to
        narrow a confidence interval. Holding density fixed is what makes `count` a
        precision knob rather than a policy knob.
        """
        densities = []
        for count in (200, 800, 2000):
            batch = generate(count=count, seed=7, horizon_days=30)
            distinct = len({event.customer_id for event in batch.events})
            densities.append(len(batch.events) / distinct)
        self.assertLess(max(densities) - min(densities), 0.15, f"drifted: {densities}")

    def test_the_customer_draw_cannot_perturb_any_other_field(self) -> None:
        """A regression test for the collateral damage, not the exception itself.

        The customer used to be drawn from the shared per-event stream over a
        count-dependent range, which consumed a count-dependent amount of randomness
        and shifted every draw after it — moving roughly one issuer in a hundred
        between batch sizes for no reason anybody intended. It now has its own
        stream, so the blast radius is exactly the one field.
        """
        short = generate(count=100, seed=7, horizon_days=30)
        long = generate(count=200, seed=7, horizon_days=30)
        by_id = {event.event_id: event for event in long.events}
        self.assertEqual(
            [e.issuer for e in short.events],
            [by_id[e.event_id].issuer for e in short.events],
        )


class BatchShape(unittest.TestCase):
    def test_events_are_ordered_by_occurrence(self) -> None:
        batch = generate(count=300, seed=11, horizon_days=45)
        times = [event.occurred_at for event in batch.events]
        self.assertEqual(times, sorted(times))

    def test_failures_land_early_enough_to_be_recoverable(self) -> None:
        """Events are clustered in the first third of the horizon on purpose.

        A balance retry can wait days for a salary window. If failures were spread
        across the whole horizon, the horizon would truncate the agent's own
        strategy and the comparison would understate it for a reason that has
        nothing to do with the policy.
        """
        batch = generate(count=300, seed=11, horizon_days=45)
        cutoff = DEFAULT_START.timestamp() + 15 * 86400
        self.assertTrue(all(e.occurred_at.timestamp() < cutoff for e in batch.events))

    def test_subscription_share_is_respected(self) -> None:
        batch = generate(count=2000, seed=11, horizon_days=45, subscription_share=0.35)
        share = sum(
            1 for e in batch.events if e.surface is Surface.SUBSCRIPTION_CHARGE
        ) / len(batch.events)
        self.assertAlmostEqual(share, 0.35, delta=0.03)

    def test_no_subscriptions_when_the_share_is_zero(self) -> None:
        batch = generate(count=200, seed=11, subscription_share=0.0)
        self.assertTrue(
            all(e.surface is Surface.PAYMENT for e in batch.events)
        )
        self.assertTrue(all(e.subscription_id is None for e in batch.events))

    def test_subscription_charges_carry_a_subscription_id(self) -> None:
        batch = generate(count=300, seed=11)
        for event in batch.events:
            if event.surface is Surface.SUBSCRIPTION_CHARGE:
                self.assertIsNotNone(event.subscription_id)
            else:
                self.assertIsNone(event.subscription_id)

    def test_small_ticket_portfolio_shifts_the_amounts_down(self) -> None:
        default = generate(count=400, seed=11)
        small = generate(count=400, seed=11, amount_buckets=SMALL_TICKET_BUCKETS)
        self.assertLess(
            max(e.amount.paise for e in small.events),
            max(e.amount.paise for e in default.events),
        )

    def test_downtime_windows_are_ordered_and_nonempty(self) -> None:
        batch = generate(count=50, seed=20260821, horizon_days=45)
        self.assertTrue(batch.downtime)
        for window in batch.downtime:
            self.assertLess(window.starts_at, window.ends_at)

    def test_every_event_is_a_first_attempt(self) -> None:
        """The batch is the agent's input, not a record of prior recovery work.

        If a generated event arrived already carrying attempts, the attempt cap
        would bind for reasons that predate the agent and the recovered figures
        would be measuring the generator.
        """
        batch = generate(count=300, seed=11)
        self.assertTrue(all(e.attempt_number == 1 for e in batch.events))


if __name__ == "__main__":
    unittest.main()
