"""Building a training set that is not contaminated by the policy that will use it.

Two contamination routes matter here, and both are easy to walk into.

The first is confounding by the logging policy. The obvious way to get training
data is to run the agent and learn from its own audit log, but then every row is
an action the policy chose, and the model learns p(recovered | the policy thought
this was worth doing). It ends up unable to price the actions the policy never
tries — which are exactly the ones a better policy would need to evaluate — and
its errors correlate with the old policy's blind spots. So the rows here come
from a uniform-random exploration policy instead: every candidate action the
diagnosis proposes is equally likely to be tried, at a uniformly sampled moment
in the horizon.

The honest caveat, which belongs in the write-up rather than buried here: you
cannot run uniform exploration on real money. Retrying a card you believe is dead
costs a real fee and a real customer's patience. On production data the same
coverage has to come from a small randomised exploration budget carved out of the
holdout, or from inverse-propensity weighting of the existing logs. This module
buys unconfounded data with the one currency a simulation has and real operations
do not, and the model's offline numbers should be read with that in mind.

The second route is shared randomness. The simulated world derives its coin flips
from the case id, so training against the same cases the evaluation scores would
hand the model the exact draws it is about to be tested on. Training therefore
generates its own batch from a different seed, and samples outcomes from its own
RNG namespace rather than calling the world's `attempt_succeeds`.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Sequence

from recoup.diagnosis import Diagnosis, diagnose
from recoup.domain.case import ActionKind, Attempt, RecoveryCase
from recoup.domain.events import FailureEvent
from recoup.model.estimator import ProbabilityEstimator
from recoup.model.logistic import (
    CalibrationReport,
    LabelledAttempt,
    features,
    score_predictions,
)
from recoup.simulation.generate import GeneratedBatch, generate
from recoup.simulation.world import GroundTruth, SimulatedWorld

# Offsets applied to the evaluation seed so the training and validation batches
# are disjoint from it. Arbitrary primes; what matters is only that they differ.
TRAIN_SEED_OFFSET = 7_000_003
VALIDATION_SEED_OFFSET = 7_000_009


def _draw(seed: int, *parts: str) -> float:
    material = ":".join(("explore", str(seed), *parts)).encode()
    digest = hashlib.sha256(material).digest()[:8]
    return int.from_bytes(digest, "big") / 2**64


def _case_with_history(
    event: FailureEvent, *, attempts_used: int, at: datetime
) -> RecoveryCase:
    """A case standing at the nth attempt, for feature extraction only.

    Real `Attempt` records rather than placeholders, because `features` reads
    `attempts_used` today and could reasonably read attempt timing tomorrow; a
    case built with the wrong shape would then produce plausible-looking garbage
    instead of an error.
    """
    case = RecoveryCase.open_from(event)
    for index in range(attempts_used):
        case.attempts.append(
            Attempt(
                attempted_at=at - timedelta(days=attempts_used - index),
                action=ActionKind.RETRY_SAME_RAIL,
                idempotency_key=f"explore:{event.event_id}:{index}",
                succeeded=False,
            )
        )
    return case


@dataclass(frozen=True, slots=True)
class Sample:
    """One explored decision and its realised outcome.

    Carries the full decision context rather than only the feature vector,
    because the heuristic estimator does not consume feature vectors and the
    comparison between it and the learned model has to run on identical rows to
    mean anything.
    """

    case: RecoveryCase
    diagnosis: Diagnosis
    action: ActionKind
    at: datetime
    downtime_active: bool
    true_probability: Decimal
    succeeded: bool


def explore(
    batch: GeneratedBatch,
    *,
    truth: GroundTruth | None = None,
    samples_per_case: int = 4,
) -> list[Sample]:
    """Sample decisions uniformly at random and resolve each against the world.

    Attempt number is sampled alongside action and time, because the decay in
    success across repeated attempts is one of the things the model has to learn
    and a set drawn only from first attempts would leave it flat in that
    dimension.
    """
    world = SimulatedWorld(truth or GroundTruth(), seed=batch.seed)
    samples: list[Sample] = []

    for event in batch.events:
        diagnosis = diagnose(event)
        actions = diagnosis.candidate_actions
        if not actions:
            # Nothing to learn: the policy escalates these rather than pricing
            # them, so the model is never asked about them.
            continue

        for index in range(samples_per_case):
            tag = f"{event.event_id}:{index}"
            rng = random.Random(f"explore:{batch.seed}:{tag}")
            action = actions[rng.randrange(len(actions))]
            attempts_used = rng.randrange(3)
            offset = timedelta(
                days=rng.random() * batch.horizon_days,
                hours=rng.randrange(24),
                minutes=rng.randrange(60),
            )
            at = event.occurred_at + offset
            downtime_active = any(
                window.issuer == event.issuer and window.covers(at)
                for window in batch.downtime
            )

            probability = world.success_probability(
                cause=diagnosis.root_cause,
                action=action,
                at=at,
                attempt_number=attempts_used + 1,
                downtime_active=downtime_active,
            )
            samples.append(
                Sample(
                    case=_case_with_history(event, attempts_used=attempts_used, at=at),
                    diagnosis=diagnosis,
                    action=action,
                    at=at,
                    downtime_active=downtime_active,
                    true_probability=probability,
                    succeeded=_draw(batch.seed, tag, "outcome") < float(probability),
                )
            )

    return samples


def to_rows(samples: Sequence[Sample]) -> list[LabelledAttempt]:
    """Reduce samples to what the model is allowed to see."""
    return [
        LabelledAttempt(
            features=features(
                sample.case,
                sample.diagnosis,
                sample.action,
                at=sample.at,
                downtime_active=sample.downtime_active,
            ),
            succeeded=sample.succeeded,
        )
        for sample in samples
    ]


def score(
    samples: Sequence[Sample], estimator: ProbabilityEstimator, *, name: str = ""
) -> CalibrationReport:
    """Put any estimator through the measurement the learned model gets.

    Calls `estimate` once per sample and reads off the probability for the action
    that was actually explored, which is the number the expected-value gate would
    have used.
    """
    predictions: list[float] = []
    labels: list[bool] = []
    for sample in samples:
        estimates = estimator.estimate(
            sample.case,
            sample.diagnosis,
            at=sample.at,
            downtime_active=sample.downtime_active,
        )
        probability = estimates.get(sample.action)
        if probability is None:
            continue
        predictions.append(float(probability))
        labels.append(sample.succeeded)
    return score_predictions(predictions, labels, name=name or estimator.name)


def oracle_report(samples: Sequence[Sample]) -> CalibrationReport:
    """The world's own probabilities, scored the same way.

    This is the ceiling: no estimator can beat the number the outcomes were drawn
    from. Reporting it turns "Brier 0.10" from a bare figure into a position on a
    scale, and it is the only way to tell a good model from an easy problem.
    """
    return score_predictions(
        [float(sample.true_probability) for sample in samples],
        [sample.succeeded for sample in samples],
        name="oracle (true probabilities)",
    )


def training_and_validation(
    *,
    count: int = 6000,
    seed: int = 20260821,
    horizon_days: int = 45,
    truth: GroundTruth | None = None,
    samples_per_case: int = 4,
) -> tuple[list[Sample], list[Sample]]:
    """Two disjoint sample sets, from two batches neither of which is the evaluation batch.

    The validation set exists because `CalibratedLogisticEstimator.train` fits its
    isotonic curve on a fold that is then in-sample for calibration; the numbers
    quoted in EVALUATION.md come from scoring this set, which no stage of fitting
    has touched.
    """
    train_batch = generate(
        count=count, seed=seed + TRAIN_SEED_OFFSET, horizon_days=horizon_days
    )
    validation_batch = generate(
        count=max(count // 3, 200),
        seed=seed + VALIDATION_SEED_OFFSET,
        horizon_days=horizon_days,
    )
    return (
        explore(train_batch, truth=truth, samples_per_case=samples_per_case),
        explore(validation_batch, truth=truth, samples_per_case=samples_per_case),
    )


def positive_rate(samples: Sequence[Sample]) -> Decimal:
    if not samples:
        return Decimal("0")
    return Decimal(sum(1 for s in samples if s.succeeded)) / Decimal(len(samples))
