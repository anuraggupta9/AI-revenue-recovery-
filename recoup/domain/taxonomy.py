"""Razorpay's failure vocabulary, as enums.

A failed payment comes back carrying four fields that together describe what went
wrong: `error_source`, `error_step`, `error_reason` and a free-text
`error_description`. The first three are effectively a ready-made root-cause
schema, and they are the reason this project does not need to guess at causes
from prose — the interesting work is mapping the triple onto an intervention,
which happens one layer up in recoup.diagnosis.

Every enum here parses permissively. Razorpay adds reason codes over time, and a
new code appearing in production must degrade to UNKNOWN and route to human
review, not raise inside the ingestion path and drop the event.
"""

from __future__ import annotations

from enum import Enum


class _Permissive(str, Enum):
    """Base for enums that must tolerate values the gateway invents later."""

    @classmethod
    def parse(cls, value: str | None) -> "_Permissive":
        if value is None:
            return cls.UNKNOWN  # type: ignore[attr-defined]
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.UNKNOWN  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return self.value


class ErrorSource(_Permissive):
    """Who or what originated the failure.

    The most decision-relevant field of the three. A GATEWAY failure is a routing
    problem and wants an alternate acquirer immediately; a CUSTOMER failure wants
    a different conversation and often a different day.
    """

    CUSTOMER = "customer"
    BUSINESS = "business"
    BANK = "bank"
    GATEWAY = "gateway"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class ErrorStep(_Permissive):
    """How far the payment got before it died.

    Matters for whether a same-session retry is even possible: a failure at
    AUTHENTICATION can often be retried immediately with the customer still
    present, whereas one at AUTHORIZATION means the issuer has already said no.
    """

    INITIATION = "payment_initiation"
    AUTHENTICATION = "payment_authentication"
    AUTHORIZATION = "payment_authorization"
    RESPONSE = "payment_response"
    UNKNOWN = "unknown"


class ErrorReason(_Permissive):
    """The specific failure code.

    Only the reasons this system handles distinctly are enumerated. Anything else
    parses to UNKNOWN and is escalated rather than guessed at — an unrecognised
    reason is precisely the case where an automated money action is least safe.
    """

    # Soft declines — the money exists or will exist, timing is the lever.
    INSUFFICIENT_FUNDS = "insufficient_funds"
    MANDATE_INSUFFICIENT_BALANCE = "mandate_insufficient_balance"

    # Recoverable in-session: the customer fumbled an input.
    INCORRECT_OTP = "incorrect_otp"
    INVALID_CVV = "invalid_cvv"
    INVALID_VPA = "invalid_vpa"
    OTP_NOT_ENTERED = "otp_not_entered"

    # Instrument problems — retrying the same rail cannot succeed.
    EXPIRED_CARD = "expired_card"
    CARD_DECLINED = "card_declined"
    INTERNATIONAL_NOT_ALLOWED = "international_transaction_not_allowed"
    MANDATE_REVOKED = "mandate_revoked"

    # Transient infrastructure — switch route now, or wait for the window.
    GATEWAY_TECHNICAL_ERROR = "gateway_technical_error"
    ISSUER_DOWN = "issuer_down"
    NETWORK_ERROR = "network_error"

    # Drop-off rather than decline.
    PAYMENT_TIMEOUT = "payment_timeout"
    UPI_COLLECT_EXPIRED = "upi_collect_expired"

    # Never auto-retry these.
    RISK_DECLINED = "risk_declined"
    SUSPECTED_FRAUD = "suspected_fraud"

    UNKNOWN = "unknown"


class PaymentMethod(_Permissive):
    """The rail an attempt used.

    Rail choice is one of the three levers the policy engine controls, alongside
    timing and channel, so this is not merely descriptive metadata.
    """

    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"
    PAYLATER = "paylater"
    UNKNOWN = "unknown"


# Reasons where an automated retry is forbidden outright, independent of what the
# propensity model estimates. Kept here beside the vocabulary it constrains so
# that adding a reason code forces you to think about which bucket it lands in.
#
# RISK_DECLINED and SUSPECTED_FRAUD are excluded because retrying a risk decline
# is how a merchant turns a blocked transaction into a chargeback.
#
# UNKNOWN is deliberately *not* on this list, despite "we do not know what
# happened" being a poor basis for moving money. Membership here suppresses a
# case — closes it, silently, with no human ever seeing it — and an unrecognised
# reason code is precisely the case a human should look at. It is instead caught
# by the confidence floor in recoup.policy.rules, which escalates. Escalation and
# suppression both stop the agent acting; only one of them surfaces the money.
NEVER_AUTO_RETRY: frozenset[ErrorReason] = frozenset(
    {
        ErrorReason.RISK_DECLINED,
        ErrorReason.SUSPECTED_FRAUD,
        ErrorReason.MANDATE_REVOKED,
    }
)

# Reasons where retrying the *same* rail is pointless but another rail may work.
# Distinct from NEVER_AUTO_RETRY: here the customer still wants to pay.
REQUIRES_RAIL_SWITCH: frozenset[ErrorReason] = frozenset(
    {
        ErrorReason.EXPIRED_CARD,
        ErrorReason.CARD_DECLINED,
        ErrorReason.INTERNATIONAL_NOT_ALLOWED,
    }
)
