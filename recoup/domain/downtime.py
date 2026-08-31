"""Issuer downtime as a value object.

This lives in `domain` rather than in either the gateway or the simulation
because both need it and neither owns it. In production these windows come from
Razorpay's Payment Downtime API; in the evaluation harness they are generated.
The policy engine consumes them without caring which, and that is the whole
reason the type sits down here where every layer can reach it.

It was originally defined inside the mock gateway, which created a cycle:
`simulation.generate` imported it from `gateway.mock`, while `gateway.mock`
imported `SimulatedWorld` from `simulation.world`. The layering test caught it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DowntimeWindow:
    """A period during which one issuer is degraded or unavailable.

    Half-open on purpose: `[starts_at, ends_at)`. Adjacent windows from the same
    issuer then tile without the instant at the boundary belonging to both, which
    matters because the policy engine's downtime rule defers an action while a
    window covers the clock and two overlapping windows would otherwise make the
    deferral depend on iteration order.
    """

    issuer: str
    starts_at: datetime
    ends_at: datetime

    def covers(self, at: datetime) -> bool:
        return self.starts_at <= at < self.ends_at
