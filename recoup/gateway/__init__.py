"""Payment provider adapters.

`base` defines the boundary: the `PaymentGateway` protocol and the `ActionResult`
every implementation returns. That protocol is the entire contents of this package,
and it imports only the standard library.

There is no live Razorpay adapter. An earlier version of this docstring described
one living in `razorpay_live`, imported lazily because it needed `requests` — a
module that was never written. The `[api]` extra in pyproject.toml still lists
`requests` for the shape of the eventual dependency set, but nothing here imports
it, and the README says plainly that the integration does not exist.

The deterministic in-process implementation is deliberately *not* here — it is
`recoup.simulation.mock_gateway`. It executes actions against `SimulatedWorld`, so
it depends on the simulation, and leaving it in this package made the two import
each other in a cycle.
"""

from recoup.gateway.base import ActionResult, PaymentGateway

__all__ = ["ActionResult", "PaymentGateway"]
