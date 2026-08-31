"""Synthetic batch generator.

Produces failure events whose mix is meant to be plausible for an Indian
mid-market merchant: mostly cards and UPI, a long tail of amounts, and a reason
distribution weighted toward soft declines rather than the dramatic ones. The
exact weights are estimates, stated here rather than hidden, and the sensitivity
analysis in EVALUATION.md re-runs everything with them perturbed.

The reason is drawn conditional on the rail and on whether the attempt was a
checkout or a recurring debit, because most reason/rail pairs are impossible and
generating them makes the rest of the system look better than it is.

One known gap in the other direction: UPI has no incorrect-PIN reason here. It is
a large real bucket, but the taxonomy this repo maps against has no code for it,
and inventing an enum value to make the histogram look right would be inventing
data. UPI failures are correspondingly weighted toward balance and collect expiry.

Determinism is a hard requirement, not a convenience. Every draw comes from a
stream seeded by the batch seed and the record index, so adding a record does not
shift the ones before it and any reported figure can be reproduced exactly.

One documented exception: the customer a failure belongs to depends on `count`,
because the customer pool scales with the batch. Two runs at different sizes
therefore agree on every field of a given event except which customer it hit. The
reason is given at the draw itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TypeVar

from recoup.domain.events import FailureEvent, Surface
from recoup.domain.money import Money
from recoup.domain.taxonomy import ErrorReason, ErrorSource, ErrorStep, PaymentMethod
from recoup.domain.downtime import DowntimeWindow

_T = TypeVar("_T")

# Fixed default so a run with no arguments is reproducible down to the timestamp.
DEFAULT_START = datetime(2026, 7, 20, 6, 0, tzinfo=timezone.utc)

# Reason mix, conditional on the rail. Soft declines dominate because they do in
# reality — which is also why a recovery agent is worth building at all.
#
# The conditioning is the point. An earlier version drew reason and method
# independently from one table, which put `invalid_vpa` on cards and `expired_card`
# on UPI in a quarter of the batch; see docs/BUILD_LOG.md. Beyond being wrong on its
# face, an impossible combination is a free feature for the estimator to fit and a
# free win for the diagnosis layer, because nothing in the real world generates it.
#
# Each table ends with UNKNOWN so the escalation path is exercised by the batch
# rather than only by unit tests.
_CARD_REASONS: list[tuple[ErrorReason, int]] = [
    (ErrorReason.INSUFFICIENT_FUNDS, 22),
    # The largest single bucket on Indian cards: 3-D Secure sends an OTP and the
    # customer mistypes it, or never comes back to the page at all.
    (ErrorReason.INCORRECT_OTP, 18),
    (ErrorReason.CARD_DECLINED, 14),
    (ErrorReason.OTP_NOT_ENTERED, 8),
    (ErrorReason.GATEWAY_TECHNICAL_ERROR, 8),
    (ErrorReason.EXPIRED_CARD, 7),
    (ErrorReason.INVALID_CVV, 6),
    (ErrorReason.ISSUER_DOWN, 5),
    (ErrorReason.PAYMENT_TIMEOUT, 4),
    (ErrorReason.NETWORK_ERROR, 3),
    (ErrorReason.RISK_DECLINED, 2),
    (ErrorReason.INTERNATIONAL_NOT_ALLOWED, 2),
    (ErrorReason.SUSPECTED_FRAUD, 1),
    (ErrorReason.UNKNOWN, 2),
]

# No OTP and no CVV: UPI authenticates with a PIN entered in the payer's own app,
# and the card lifecycle reasons have no meaning on a VPA. Collect expiry — the
# customer simply never approving the request — takes the place card OTP failure
# holds above.
_UPI_REASONS: list[tuple[ErrorReason, int]] = [
    (ErrorReason.INSUFFICIENT_FUNDS, 30),
    (ErrorReason.UPI_COLLECT_EXPIRED, 18),
    (ErrorReason.PAYMENT_TIMEOUT, 14),
    (ErrorReason.ISSUER_DOWN, 9),
    (ErrorReason.GATEWAY_TECHNICAL_ERROR, 9),
    (ErrorReason.INVALID_VPA, 7),
    (ErrorReason.NETWORK_ERROR, 4),
    (ErrorReason.RISK_DECLINED, 2),
    (ErrorReason.SUSPECTED_FRAUD, 1),
    (ErrorReason.UNKNOWN, 2),
]

# Netbanking does use an OTP, on the bank's own page, so authentication failure
# belongs here. Nothing card-shaped does.
_NETBANKING_REASONS: list[tuple[ErrorReason, int]] = [
    (ErrorReason.INSUFFICIENT_FUNDS, 26),
    (ErrorReason.INCORRECT_OTP, 18),
    (ErrorReason.PAYMENT_TIMEOUT, 16),
    (ErrorReason.OTP_NOT_ENTERED, 12),
    (ErrorReason.ISSUER_DOWN, 10),
    (ErrorReason.GATEWAY_TECHNICAL_ERROR, 9),
    (ErrorReason.NETWORK_ERROR, 4),
    (ErrorReason.RISK_DECLINED, 2),
    (ErrorReason.SUSPECTED_FRAUD, 1),
    (ErrorReason.UNKNOWN, 3),
]

# Wallets fail on balance far more than anything else, because a wallet holds a
# float the customer topped up rather than drawing on an account.
_WALLET_REASONS: list[tuple[ErrorReason, int]] = [
    (ErrorReason.INSUFFICIENT_FUNDS, 34),
    (ErrorReason.PAYMENT_TIMEOUT, 18),
    (ErrorReason.INCORRECT_OTP, 12),
    (ErrorReason.OTP_NOT_ENTERED, 10),
    (ErrorReason.GATEWAY_TECHNICAL_ERROR, 9),
    (ErrorReason.ISSUER_DOWN, 6),
    (ErrorReason.NETWORK_ERROR, 5),
    (ErrorReason.RISK_DECLINED, 2),
    (ErrorReason.SUSPECTED_FRAUD, 1),
    (ErrorReason.UNKNOWN, 3),
]

# EMI is a card underneath, so it inherits the card reasons; "insufficient funds"
# means the credit limit rather than a balance, and a domestic EMI plan cannot fail
# for being an international transaction.
_EMI_REASONS: list[tuple[ErrorReason, int]] = [
    (reason, weight)
    for reason, weight in _CARD_REASONS
    if reason is not ErrorReason.INTERNATIONAL_NOT_ALLOWED
]

_REASONS_BY_METHOD: dict[PaymentMethod, list[tuple[ErrorReason, int]]] = {
    PaymentMethod.CARD: _CARD_REASONS,
    PaymentMethod.UPI: _UPI_REASONS,
    PaymentMethod.NETBANKING: _NETBANKING_REASONS,
    PaymentMethod.WALLET: _WALLET_REASONS,
    PaymentMethod.EMI: _EMI_REASONS,
}

# A recurring debit is not a checkout. Nobody is at the keyboard, so the whole
# customer-input family — mistyped OTP, abandoned collect request, wrong VPA — is
# unavailable, and the failure surface narrows to the bank, the rails, and the
# state of the mandate itself. This is why a failed subscription charge is a
# different recovery problem rather than the same one with a different label:
# there is no session to re-engage, only a debit to re-present.
_MANDATE_REASONS: list[tuple[ErrorReason, int]] = [
    (ErrorReason.MANDATE_INSUFFICIENT_BALANCE, 56),
    (ErrorReason.GATEWAY_TECHNICAL_ERROR, 12),
    (ErrorReason.ISSUER_DOWN, 10),
    (ErrorReason.MANDATE_REVOKED, 9),
    (ErrorReason.NETWORK_ERROR, 5),
    # Lighter than the checkout tables on purpose. A recurring debit runs against
    # a payer the customer already authorised, so the risk engine has far less
    # reason to stop it than it does an unrecognised card at a checkout.
    (ErrorReason.RISK_DECLINED, 3),
    (ErrorReason.SUSPECTED_FRAUD, 1),
    (ErrorReason.UNKNOWN, 4),
]

# The one exception to "nobody is at the keyboard". RBI requires an additional
# factor of authentication for recurring card debits above ₹15,000, so those do
# put an OTP in front of the customer and can fail the way a checkout fails.
# Below the threshold the debit is silent and cannot.
_AFA_THRESHOLD = Money.from_rupees("15000")
_AFA_METHODS = frozenset({PaymentMethod.CARD, PaymentMethod.EMI})
_AFA_REASONS: list[tuple[ErrorReason, int]] = [
    (ErrorReason.INCORRECT_OTP, 14),
    (ErrorReason.OTP_NOT_ENTERED, 9),
]

# (source, step) consistent with each reason. Getting this wrong would let the
# fallback mapping look better than it is.
_CONTEXT: dict[ErrorReason, tuple[ErrorSource, ErrorStep]] = {
    ErrorReason.INSUFFICIENT_FUNDS: (ErrorSource.BANK, ErrorStep.AUTHORIZATION),
    ErrorReason.MANDATE_INSUFFICIENT_BALANCE: (ErrorSource.BANK, ErrorStep.AUTHORIZATION),
    ErrorReason.INCORRECT_OTP: (ErrorSource.CUSTOMER, ErrorStep.AUTHENTICATION),
    ErrorReason.OTP_NOT_ENTERED: (ErrorSource.CUSTOMER, ErrorStep.AUTHENTICATION),
    ErrorReason.INVALID_CVV: (ErrorSource.CUSTOMER, ErrorStep.AUTHENTICATION),
    ErrorReason.INVALID_VPA: (ErrorSource.CUSTOMER, ErrorStep.INITIATION),
    ErrorReason.EXPIRED_CARD: (ErrorSource.BANK, ErrorStep.AUTHORIZATION),
    ErrorReason.CARD_DECLINED: (ErrorSource.BANK, ErrorStep.AUTHORIZATION),
    ErrorReason.INTERNATIONAL_NOT_ALLOWED: (ErrorSource.BANK, ErrorStep.AUTHORIZATION),
    ErrorReason.ISSUER_DOWN: (ErrorSource.BANK, ErrorStep.AUTHORIZATION),
    ErrorReason.GATEWAY_TECHNICAL_ERROR: (ErrorSource.GATEWAY, ErrorStep.AUTHORIZATION),
    ErrorReason.NETWORK_ERROR: (ErrorSource.GATEWAY, ErrorStep.RESPONSE),
    ErrorReason.PAYMENT_TIMEOUT: (ErrorSource.CUSTOMER, ErrorStep.INITIATION),
    ErrorReason.UPI_COLLECT_EXPIRED: (ErrorSource.CUSTOMER, ErrorStep.INITIATION),
    ErrorReason.RISK_DECLINED: (ErrorSource.BUSINESS, ErrorStep.AUTHORIZATION),
    ErrorReason.SUSPECTED_FRAUD: (ErrorSource.BUSINESS, ErrorStep.AUTHORIZATION),
    ErrorReason.MANDATE_REVOKED: (ErrorSource.CUSTOMER, ErrorStep.AUTHORIZATION),
    # Unmapped on purpose: source and step are unknown too, so the case escalates
    # rather than being rescued by the fallback table.
    ErrorReason.UNKNOWN: (ErrorSource.UNKNOWN, ErrorStep.UNKNOWN),
}

_METHOD_WEIGHTS: list[tuple[PaymentMethod, int]] = [
    (PaymentMethod.UPI, 46),
    (PaymentMethod.CARD, 38),
    (PaymentMethod.NETBANKING, 9),
    (PaymentMethod.WALLET, 5),
    (PaymentMethod.EMI, 2),
]

_ISSUERS = ("HDFC", "ICICI", "SBIN", "AXIS", "KOTAK", "PAYTM", "YESB")

# Amount buckets in rupees, with weights. Long-tailed: many small tickets, a few
# large ones. The tail matters because expected value scales with amount, so the
# policy's willingness to act should differ visibly across it.
_AMOUNT_BUCKETS: list[tuple[int, int, int]] = [
    (99, 499, 30),
    (500, 1_499, 28),
    (1_500, 4_999, 22),
    (5_000, 14_999, 13),
    (15_000, 49_999, 6),
    (50_000, 250_000, 1),
]

# A deliberately small-ticket portfolio, for the experiment in EVALUATION.md that
# asks whether the expected-value floor ever actually binds. On the default
# distribution it mostly does not — a ₹5,000 case clears a ₹1 floor at almost any
# probability — so the question of whether the estimator is well calibrated cannot
# change a decision. Shifting the whole distribution down an order of magnitude is
# how to find out whether that slackness is the explanation or an excuse.
SMALL_TICKET_BUCKETS: list[tuple[int, int, int]] = [
    (10, 49, 30),
    (50, 149, 28),
    (150, 499, 22),
    (500, 1_499, 13),
    (1_500, 4_999, 6),
    (5_000, 25_000, 1),
]


@dataclass(frozen=True, slots=True)
class GeneratedBatch:
    events: list[FailureEvent]
    downtime: list[DowntimeWindow]
    seed: int
    horizon_days: int


def _pick(rng: random.Random, weighted: list[tuple[_T, int]]) -> _T:
    total = sum(weight for _, weight in weighted)
    roll = rng.randrange(total)
    for value, weight in weighted:
        roll -= weight
        if roll < 0:
            return value
    return weighted[-1][0]


def _amount(rng: random.Random, buckets: list[tuple[int, int, int]]) -> Money:
    low, high = _pick(rng, [((low, high), weight) for low, high, weight in buckets])
    return Money.from_rupees(str(rng.randrange(low, high + 1)))


def _reason_table(
    method: PaymentMethod, surface: Surface, amount: Money
) -> list[tuple[ErrorReason, int]]:
    """Which failures this particular attempt could plausibly have suffered.

    Exposed for the test that walks every combination and asserts the generator
    cannot emit a reason outside the table for its rail.
    """
    if surface is not Surface.SUBSCRIPTION_CHARGE:
        return _REASONS_BY_METHOD[method]
    if method in _AFA_METHODS and amount > _AFA_THRESHOLD:
        return _MANDATE_REASONS + _AFA_REASONS
    return _MANDATE_REASONS


def generate(
    *,
    count: int = 400,
    seed: int = 20260821,
    start: datetime | None = None,
    horizon_days: int = 45,
    subscription_share: float = 0.35,
    amount_buckets: list[tuple[int, int, int]] | None = None,
) -> GeneratedBatch:
    """Build a batch of failures spread across the first third of the horizon.

    Events are clustered early so that deferred actions — particularly balance
    retries waiting for a salary window — have room to execute before the horizon
    closes. A horizon that truncates the agent's own strategy would understate it.
    """
    start = start or DEFAULT_START
    buckets = amount_buckets or _AMOUNT_BUCKETS
    ingest_window = max(1, horizon_days // 3)
    events: list[FailureEvent] = []

    for index in range(count):
        rng = random.Random(f"{seed}:event:{index}")

        # Rail and amount are drawn first because the reason depends on both: a VPA
        # cannot expire on a card, and the authentication requirement on a mandate
        # turns on the ticket size.
        method = _pick(rng, _METHOD_WEIGHTS)
        amount = _amount(rng, buckets)
        surface = (
            Surface.SUBSCRIPTION_CHARGE
            if rng.random() < subscription_share
            else Surface.PAYMENT
        )
        reason = _pick(rng, _reason_table(method, surface, amount))
        source, step = _CONTEXT[reason]

        occurred = start + timedelta(
            days=rng.randrange(ingest_window),
            hours=rng.randrange(24),
            minutes=rng.randrange(60),
        )
        entity = f"pay_{seed % 10000:04d}{index:05d}"
        issuer = rng.choice(_ISSUERS)

        # The customer pool scales with the batch, so a merchant with twice the
        # failures has twice the customers rather than the same few failing twice as
        # often. That keeps the per-customer contact cap binding at a rate
        # independent of batch size, which matters because `count` is the knob the
        # confidence intervals are tightened with — if density rose with count, a
        # larger sample would change the policy's behaviour and not just its
        # precision.
        #
        # The cost is that the customer attached to a given index is a function of
        # count, so it is drawn from its own stream: a count-dependent range would
        # otherwise consume a count-dependent amount of randomness and shift every
        # draw after it. That is not hypothetical — it was silently moving one
        # issuer in a hundred before this was split out.
        pool = max(2, count // 2)
        customer = random.Random(f"{seed}:customer:{index}:{pool}").randrange(1, pool)

        events.append(
            FailureEvent(
                event_id=entity,
                occurred_at=occurred,
                surface=surface,
                entity_id=entity,
                customer_id=f"cust_{customer:05d}",
                amount=amount,
                method=method,
                error_source=source,
                error_step=step,
                error_reason=reason,
                attempt_number=1,
                issuer=issuer,
                subscription_id=(
                    f"sub_{index:05d}" if surface is Surface.SUBSCRIPTION_CHARGE else None
                ),
                error_description=f"synthetic {reason}",
            )
        )

    events.sort(key=lambda event: event.occurred_at)
    return GeneratedBatch(
        events=events,
        downtime=_downtime_windows(seed=seed, start=start, horizon_days=horizon_days),
        seed=seed,
        horizon_days=horizon_days,
    )


def _downtime_windows(
    *, seed: int, start: datetime, horizon_days: int
) -> list[DowntimeWindow]:
    """A handful of issuer outages, so the downtime rule has something to catch."""
    rng = random.Random(f"{seed}:downtime")
    windows: list[DowntimeWindow] = []
    for issuer in _ISSUERS:
        for _ in range(rng.randrange(0, 3)):
            begins = start + timedelta(
                days=rng.randrange(horizon_days), hours=rng.randrange(24)
            )
            windows.append(
                DowntimeWindow(
                    issuer=issuer,
                    starts_at=begins,
                    ends_at=begins + timedelta(hours=rng.randrange(2, 9)),
                )
            )
    return windows
