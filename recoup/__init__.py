"""Recoup — a bounded recovery agent for failed payments and subscription mandates.

Layering, innermost first. Each layer may import the ones above it, never below:

    domain      value objects, entities, the recovery state machine
    audit       tamper-evident decision log
    diagnosis   failure taxonomy -> named root cause
    policy      stopping rules, expected-value gate, shadow mode
    model       recovery-probability estimation and calibration   (needs numpy)
    gateway     payment provider adapter interface
    simulation  synthetic world, arms, and the evaluation harness

The first four import only the standard library. That boundary is enforced by
tests/test_core_has_no_dependencies.py, because it is the property that keeps the
policy engine cheap to test and impossible to break from a version bump. The same
test enforces the ordering above, which is how `simulation` came to be listed
where it is: it drives the gateway rather than being driven by it.

An `api` layer belongs after `simulation` and does not exist. It was listed here
before it was written, which is the kind of documentation that costs a reader
their time; the layering test knows about the name so the constraint is ready when
the package is, and the README says plainly that there is no HTTP surface.
"""

__version__ = "0.1.0"
