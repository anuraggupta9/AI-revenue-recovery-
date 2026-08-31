"""Money, denominated in the smallest currency unit.

Razorpay reports every amount in paise: an order for 499 rupees arrives as
`"amount": 49900`. Treating that as rupees, or dividing by 100 into a float
somewhere in the middle of a pipeline, is the single most expensive class of bug
available in this codebase — it does not raise, it silently misstates the
headline number the whole submission rests on.

So there is no float anywhere in this module. Amounts are integer paise;
anything that needs a fraction of a rupee goes through Decimal with an explicit
rounding mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Union

PAISE_PER_RUPEE = 100

Numeric = Union[int, str, Decimal]


class CurrencyMismatch(ValueError):
    """Raised when two amounts in different currencies are combined."""


@dataclass(frozen=True, slots=True, order=True)
class Money:
    """An exact amount of money. Immutable, hashable, comparable.

    Construct with the named helpers rather than the initialiser; they make the
    unit explicit at the call site, which is the whole point:

        Money.from_paise(49900)     # what the gateway gave us
        Money.from_rupees("499")    # what a human typed
    """

    # Field order matters: `order=True` compares currency first, which keeps
    # sorting deterministic across mixed-currency collections even though
    # arithmetic on them is rejected.
    currency: str
    paise: int

    def __post_init__(self) -> None:
        if not isinstance(self.paise, int) or isinstance(self.paise, bool):
            raise TypeError(
                f"paise must be int, got {type(self.paise).__name__}. "
                "Use Money.from_rupees() if you have a rupee value."
            )
        if not self.currency.isalpha() or len(self.currency) != 3:
            raise ValueError(f"currency must be a 3-letter code, got {self.currency!r}")
        object.__setattr__(self, "currency", self.currency.upper())

    @classmethod
    def from_paise(cls, paise: int, currency: str = "INR") -> Money:
        return cls(currency=currency, paise=paise)

    @classmethod
    def from_rupees(cls, rupees: Numeric, currency: str = "INR") -> Money:
        """Convert a rupee amount to exact paise.

        Floats are rejected on purpose: Decimal(0.1) is 0.1000000000000000055,
        and a value that arrives here as a float has already lost precision
        somewhere upstream where it is much harder to find. Pass a str.
        """
        if isinstance(rupees, float):
            raise TypeError(
                "refusing float rupees — precision is already lost. "
                'Pass a string instead, e.g. Money.from_rupees("499.50")'
            )
        amount = Decimal(rupees) * PAISE_PER_RUPEE
        if amount != amount.to_integral_value():
            raise ValueError(f"{rupees} rupees is not a whole number of paise")
        return cls(currency=currency, paise=int(amount))

    @classmethod
    def zero(cls, currency: str = "INR") -> Money:
        return cls(currency=currency, paise=0)

    @property
    def rupees(self) -> Decimal:
        """Exact rupee value. For display and reporting, never for arithmetic."""
        return Decimal(self.paise) / PAISE_PER_RUPEE

    def scale(self, factor: Numeric, rounding: str = ROUND_HALF_EVEN) -> Money:
        """Multiply by a fraction — a recovery probability, a margin, a fee rate.

        Rounding is banker's by default so that repeatedly scaling many small
        amounts across a batch does not accumulate a systematic upward bias in
        the reported expected value.
        """
        if isinstance(factor, float):
            raise TypeError(
                "refusing float factor — pass a str or Decimal so the rounding "
                "of money is explicit and reproducible"
            )
        with localcontext() as ctx:
            ctx.rounding = rounding
            scaled = (Decimal(self.paise) * Decimal(factor)).to_integral_value()
        return Money(currency=self.currency, paise=int(scaled))

    def _check(self, other: Money) -> None:
        if not isinstance(other, Money):
            raise TypeError(f"cannot combine Money with {type(other).__name__}")
        if self.currency != other.currency:
            raise CurrencyMismatch(f"{self.currency} vs {other.currency}")

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(currency=self.currency, paise=self.paise + other.paise)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(currency=self.currency, paise=self.paise - other.paise)

    def __neg__(self) -> Money:
        return Money(currency=self.currency, paise=-self.paise)

    def __abs__(self) -> Money:
        return Money(currency=self.currency, paise=abs(self.paise))

    def __bool__(self) -> bool:
        return self.paise != 0

    @property
    def is_positive(self) -> bool:
        return self.paise > 0

    def __str__(self) -> str:
        sign = "-" if self.paise < 0 else ""
        whole, frac = divmod(abs(self.paise), PAISE_PER_RUPEE)
        symbol = "₹" if self.currency == "INR" else f"{self.currency} "
        return f"{sign}{symbol}{whole:,}.{frac:02d}"

    def __repr__(self) -> str:
        return f"Money.from_paise({self.paise}, {self.currency!r})"


def total(amounts: object, currency: str = "INR") -> Money:
    """Sum an iterable of Money, returning zero for an empty one.

    Exists because sum() needs a typed start value and `sum(xs, Money.zero())`
    reads badly at every call site.
    """
    result = Money.zero(currency)
    for amount in amounts:  # type: ignore[union-attr]
        result = result + amount
    return result
