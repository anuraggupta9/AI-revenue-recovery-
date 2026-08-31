"""Recovery-probability estimation.

`estimator` holds the interface and the dependency-free estimators. The numpy
logistic regression and its calibration live in `logistic`, imported lazily via
`load_logistic()` so this package keeps working without numpy installed — the
policy engine and the whole core depend on this package's interface, and none of
them should acquire a numpy dependency by transitivity.
"""

from recoup.model.estimator import (
    FixedScheduleEstimator,
    HeuristicEstimator,
    ProbabilityEstimator,
)

__all__ = [
    "FixedScheduleEstimator",
    "HeuristicEstimator",
    "ProbabilityEstimator",
    "load_logistic",
]


def load_logistic():
    """Return the `recoup.model.logistic` module, or raise a legible error.

    numpy's own ImportError says nothing about why this project wants it, so it
    is translated here. The rest of the system runs on `HeuristicEstimator`
    without it, and that is a supported configuration rather than a degraded one.
    """
    try:
        from recoup.model import logistic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the calibrated logistic estimator needs numpy (pip install numpy). "
            "Everything else, including the full policy engine and all four "
            "baseline arms, runs on the standard library alone."
        ) from exc
    return logistic
