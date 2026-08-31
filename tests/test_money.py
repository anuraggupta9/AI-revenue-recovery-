"""Money is the foundation every reported number rests on, so it gets the
most paranoid tests in the repo. The float rejections in particular are the
point: they are what make the paise/rupee confusion a crash instead of a
quietly wrong headline figure.
"""

import unittest
from decimal import Decimal

from recoup.domain.money import CurrencyMismatch, Money, total


class TestConstruction(unittest.TestCase):
    def test_from_paise_is_exact(self):
        self.assertEqual(Money.from_paise(49900).paise, 49900)

    def test_from_rupees_accepts_str_int_and_decimal(self):
        self.assertEqual(Money.from_rupees("499").paise, 49900)
        self.assertEqual(Money.from_rupees(499).paise, 49900)
        self.assertEqual(Money.from_rupees(Decimal("499.50")).paise, 49950)

    def test_from_rupees_rejects_float(self):
        # The whole reason this class exists.
        with self.assertRaises(TypeError):
            Money.from_rupees(499.50)

    def test_from_rupees_rejects_sub_paise_precision(self):
        with self.assertRaises(ValueError):
            Money.from_rupees("499.005")

    def test_paise_must_be_int(self):
        with self.assertRaises(TypeError):
            Money(currency="INR", paise=499.0)

    def test_bool_is_not_an_int_here(self):
        # bool subclasses int, which would otherwise sneak through as 0 or 1.
        with self.assertRaises(TypeError):
            Money(currency="INR", paise=True)

    def test_currency_is_normalised_and_validated(self):
        self.assertEqual(Money.from_paise(1, "inr").currency, "INR")
        with self.assertRaises(ValueError):
            Money.from_paise(1, "RUPEE")
        with self.assertRaises(ValueError):
            Money.from_paise(1, "1NR")

    def test_is_immutable(self):
        amount = Money.from_paise(100)
        with self.assertRaises(Exception):
            amount.paise = 200  # type: ignore[misc]

    def test_is_hashable(self):
        self.assertEqual(len({Money.from_paise(100), Money.from_paise(100)}), 1)


class TestArithmetic(unittest.TestCase):
    def test_add_and_subtract(self):
        self.assertEqual(Money.from_paise(100) + Money.from_paise(50), Money.from_paise(150))
        self.assertEqual(Money.from_paise(100) - Money.from_paise(150), Money.from_paise(-50))

    def test_currency_mismatch_is_refused(self):
        with self.assertRaises(CurrencyMismatch):
            Money.from_paise(100, "INR") + Money.from_paise(100, "USD")

    def test_cannot_add_a_bare_number(self):
        with self.assertRaises(TypeError):
            Money.from_paise(100) + 100  # type: ignore[operator]

    def test_scale_rejects_float(self):
        with self.assertRaises(TypeError):
            Money.from_paise(10_000).scale(0.35)

    def test_scale_by_probability(self):
        # An expected-value calculation: 35% chance of recovering ₹100.
        self.assertEqual(Money.from_rupees("100").scale("0.35"), Money.from_rupees("35"))

    def test_scale_uses_bankers_rounding(self):
        # 5 paise scaled by 0.5 is 2.5 paise. Half-even goes to the even
        # neighbour (2), not up — so a batch of these does not drift upward.
        self.assertEqual(Money.from_paise(5).scale("0.5").paise, 2)
        self.assertEqual(Money.from_paise(7).scale("0.5").paise, 4)

    def test_negate_and_abs(self):
        self.assertEqual(-Money.from_paise(100), Money.from_paise(-100))
        self.assertEqual(abs(Money.from_paise(-100)), Money.from_paise(100))

    def test_truthiness_tracks_nonzero(self):
        self.assertFalse(Money.zero())
        self.assertTrue(Money.from_paise(1))
        self.assertTrue(Money.from_paise(-1))

    def test_is_positive_excludes_zero(self):
        self.assertFalse(Money.zero().is_positive)
        self.assertTrue(Money.from_paise(1).is_positive)


class TestOrderingAndTotals(unittest.TestCase):
    def test_comparison(self):
        self.assertLess(Money.from_paise(100), Money.from_paise(200))
        self.assertEqual(
            max(Money.from_paise(1), Money.from_paise(300)), Money.from_paise(300)
        )

    def test_total_of_empty_is_zero(self):
        self.assertEqual(total([]), Money.zero())

    def test_total_sums(self):
        amounts = [Money.from_rupees("10"), Money.from_rupees("20.50")]
        self.assertEqual(total(amounts), Money.from_rupees("30.50"))


class TestDisplay(unittest.TestCase):
    def test_rupees_property_is_exact(self):
        self.assertEqual(Money.from_paise(49950).rupees, Decimal("499.5"))

    def test_str_groups_thousands_and_pads_paise(self):
        self.assertEqual(str(Money.from_paise(123456789)), "₹1,234,567.89")
        self.assertEqual(str(Money.from_paise(5)), "₹0.05")
        self.assertEqual(str(Money.from_paise(-49900)), "-₹499.00")

    def test_repr_round_trips(self):
        amount = Money.from_paise(49900)
        self.assertEqual(eval(repr(amount)), amount)  # noqa: S307


if __name__ == "__main__":
    unittest.main()
