"""Logistic recovery model with isotonic calibration.

The estimate this module produces is multiplied by a rupee amount and compared
against a floor, so being *ranked* correctly is not sufficient — the number has
to mean what it says. A model that ranks perfectly but reports 0.45 where the
truth is 0.20 will authorise roughly twice as many actions as it should, and no
ranking metric will notice. That is why calibration is a separate fitted stage
here rather than an afterthought, and why `diagnostics()` reports Brier score and
a reliability table alongside AUC.

Three choices worth defending.

Isotonic rather than Platt scaling. Platt fits a two-parameter sigmoid, which can
only stretch and shift a curve that is already sigmoid-shaped, so it cannot fix
one end of the range without disturbing the other. Isotonic can. Whether that
freedom is worth anything is an empirical question and the answer in this project
is "barely" — see `ModelDiagnostics` and EVALUATION.md, which report the near-null
result rather than dropping the comparison.

The calibration fold is held out from fitting. Fitting isotonic on the same
predictions used to fit the weights produces a curve that looks flawless in
sample and does nothing out of sample; it is the most common way to ship a
"calibrated" model that is not.

Two interaction terms are entered by hand. A linear model with main effects only
cannot represent "timing matters for balance failures and not for dead cards" or
"switching rails rescues a dead instrument and does nothing for an empty
account". Those two facts are the entire thesis of the project, so leaving the
model unable to express them and then reporting that it beat the heuristic would
have been a rigged comparison in the other direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

import numpy as np

from recoup.diagnosis import Diagnosis, RootCause
from recoup.domain.case import ActionKind, RecoveryCase
from recoup.domain.events import Surface
from recoup.policy.timing import SALARY_DAYS, to_ist

_CAUSES: tuple[RootCause, ...] = tuple(RootCause)
_ACTIONS: tuple[ActionKind, ...] = tuple(ActionKind)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"cause={cause}" for cause in _CAUSES),
    *(f"action={action}" for action in _ACTIONS),
    "attempts_used",
    "in_salary_window",
    "downtime_active",
    "log10_amount_rupees",
    "days_since_open",
    "is_subscription",
    "contacts_customer",
    "balance_x_salary_window",
    "dead_instrument_x_rail_switch",
)


def features(
    case: RecoveryCase,
    diagnosis: Diagnosis,
    action: ActionKind,
    *,
    at: datetime,
    downtime_active: bool,
) -> tuple[float, ...]:
    """Everything observable at decision time, and nothing else.

    Deliberately excludes the world model's parameters, the true success
    probability, and anything only knowable after the attempt. The signature is
    the same information the policy engine has when it decides, which is the
    only way the offline number means anything about the online one.
    """
    cause = diagnosis.root_cause
    in_salary_window = 1.0 if to_ist(at).day in SALARY_DAYS else 0.0
    rupees = max(1.0, case.amount_at_risk.paise / 100)
    days_open = max(0.0, (at - case.opened_at).total_seconds() / 86400)

    return (
        *(1.0 if cause is candidate else 0.0 for candidate in _CAUSES),
        *(1.0 if action is candidate else 0.0 for candidate in _ACTIONS),
        float(case.attempts_used),
        in_salary_window,
        1.0 if downtime_active else 0.0,
        math.log10(rupees),
        days_open,
        1.0 if case.surface is Surface.SUBSCRIPTION_CHARGE else 0.0,
        1.0 if action.contacts_customer else 0.0,
        in_salary_window if cause is RootCause.INSUFFICIENT_BALANCE else 0.0,
        (
            1.0
            if cause is RootCause.INSTRUMENT_INVALID
            and action is ActionKind.RETRY_ALTERNATE_RAIL
            else 0.0
        ),
    )


@dataclass(frozen=True, slots=True)
class LabelledAttempt:
    """One action taken, and whether the money arrived.

    Held as a plain feature vector rather than a reference to a case, because a
    case is mutable and a training row must be a snapshot of the moment the
    decision was made.
    """

    features: tuple[float, ...]
    succeeded: bool


def _design(rows: Sequence[LabelledAttempt]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row.features for row in rows], dtype=np.float64)
    y = np.asarray([1.0 if row.succeeded else 0.0 for row in rows], dtype=np.float64)
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch on sign rather than calling exp on large positive z. Payment
    # features include a log-amount term that can push the linear predictor well
    # outside [-30, 30] early in training, and the naive form overflows there.
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass(frozen=True, slots=True)
class _Standardiser:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> _Standardiser:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A constant column has zero variance; dividing by it yields nan and
        # poisons every weight. One-hot columns for causes that never appear in
        # the training batch are exactly this case, so it is the normal path
        # rather than an edge case.
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def _fit_weights(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    epochs: int,
    learning_rate: float,
) -> tuple[np.ndarray, float]:
    """Full-batch gradient descent on the L2-penalised log-likelihood.

    Full batch, fixed epoch count, no shuffling and no early stopping, so two
    runs on identical data produce bit-identical weights. That determinism is
    worth more here than the last fraction of a point of log-loss: the whole
    evaluation is a comparison between arms, and a model that moves between runs
    makes every difference unattributable.

    The intercept is fitted separately and left unpenalised. Shrinking it towards
    zero pulls predicted probabilities towards one half, which for an outcome
    that occurs perhaps a fifth of the time is a large and entirely avoidable
    calibration error.
    """
    n, d = x.shape
    weights = np.zeros(d, dtype=np.float64)
    bias = float(np.log(max(y.mean(), 1e-6) / max(1.0 - y.mean(), 1e-6)))

    for _ in range(epochs):
        predicted = _sigmoid(x @ weights + bias)
        residual = predicted - y
        grad_w = (x.T @ residual) / n + l2 * weights
        grad_b = float(residual.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def _isotonic(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators, returning (block mean score, block value).

    Standard PAVA on the labels sorted by score. Blocks are merged while any
    adjacent pair is out of order; the result is the non-decreasing step function
    closest to the labels in squared error.

    Each block is summarised by the *mean* score inside it rather than its right
    edge, because the caller interpolates between blocks rather than stepping, and
    a block's mean is the score its fitted value is actually an estimate for.
    """
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Each block carries (sum of labels, sum of scores, count).
    label_sums: list[float] = []
    score_sums: list[float] = []
    counts: list[float] = []
    for score, label in zip(sorted_scores, sorted_labels):
        label_sums.append(float(label))
        score_sums.append(float(score))
        counts.append(1.0)
        while len(label_sums) > 1 and (
            label_sums[-2] / counts[-2] >= label_sums[-1] / counts[-1]
        ):
            # Pop first and assign second. Writing this as `sums[-2] += sums.pop()`
            # is wrong in a way that only shows up at length two: Python resolves
            # the literal index -2 after the pop has shortened the list.
            merged_labels = label_sums.pop()
            merged_scores = score_sums.pop()
            merged_count = counts.pop()
            label_sums[-1] += merged_labels
            score_sums[-1] += merged_scores
            counts[-1] += merged_count

    block_scores = np.asarray(
        [s / c for s, c in zip(score_sums, counts)], dtype=np.float64
    )
    block_values = np.asarray(
        [s / c for s, c in zip(label_sums, counts)], dtype=np.float64
    )
    # np.interp requires a strictly increasing x. Blocks are score-ordered so the
    # means are non-decreasing, but a block of entirely tied scores can equal its
    # neighbour; keep the first of any such run.
    keep = np.ones(len(block_scores), dtype=bool)
    keep[1:] = np.diff(block_scores) > 0
    return block_scores[keep], block_values[keep]


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Monotone map from raw score to calibrated probability, linearly interpolated.

    The interpolation is not cosmetic, and getting here took a wrong turn worth
    recording. A pure isotonic step function was the first implementation, and it
    collapsed the model's output to twenty-one distinct probabilities across the
    whole batch — fewer than the hand-written heuristic's fifty-seven. Any policy
    rule that compares a probability against a threshold then behaves like a cliff:
    the tuning curve for the contact-probability floor moved in jumps, and whole
    bands of the parameter were indistinguishable from each other. Calibration had
    been bought at the price of resolution, which for an estimator whose entire job
    is to be compared against thresholds is a bad trade.

    Interpolating between block means preserves monotonicity, keeps the calibration
    that PAVA fitted, and restores a continuous output. Values outside the fitted
    range are clamped rather than extrapolated: isotonic regression makes no claim
    beyond the data it saw.
    """

    block_scores: np.ndarray
    block_values: np.ndarray

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        block_scores, block_values = _isotonic(scores, labels)
        return cls(block_scores=block_scores, block_values=block_values)

    def apply(self, scores: np.ndarray) -> np.ndarray:
        if len(self.block_scores) == 1:
            return np.full_like(scores, self.block_values[0])
        return np.interp(scores, self.block_scores, self.block_values)

    @property
    def blocks(self) -> int:
        return len(self.block_values)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """How well one set of probability estimates matches what actually happened."""

    name: str
    rows: int
    positives: int
    auc: float
    brier: float
    ece: float
    reliability: tuple[tuple[float, float, float, int], ...]

    @property
    def base_rate(self) -> float:
        return self.positives / max(self.rows, 1)

    def as_text(self) -> str:
        lines = [
            f"{self.name}: rows={self.rows:,} positives={self.positives:,} "
            f"({self.base_rate:.1%})",
            f"  AUC {self.auc:.3f}   Brier {self.brier:.4f}   ECE {self.ece:.4f}",
            "",
            "| Predicted band | Mean predicted | Observed | n |",
            "|---|---|---|---|",
        ]
        for low, predicted, observed, count in self.reliability:
            lines.append(
                f"| {low:.2f}-{low + 0.1:.2f} | {predicted:.3f} | {observed:.3f} | {count:,} |"
            )
        return "\n".join(lines)


def score_predictions(
    predictions: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    *,
    name: str = "",
) -> CalibrationReport:
    """Score any estimator's output, learned or hand-written.

    Kept independent of the model class so the heuristic priors can be put through
    the identical measurement. A learned model that is not compared against the
    heuristic it replaces has not been evaluated, only described.
    """
    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(
        [1.0 if bool(label) else 0.0 for label in labels], dtype=np.float64
    )
    return CalibrationReport(
        name=name,
        rows=len(predicted),
        positives=int(observed.sum()),
        auc=_auc(predicted, observed),
        brier=float(np.mean((predicted - observed) ** 2)),
        ece=_ece(predicted, observed),
        reliability=_reliability(predicted, observed),
    )


@dataclass(frozen=True, slots=True)
class ModelDiagnostics:
    """The two stages side by side.

    Both are reported because the gap between them is the only argument for the
    calibration stage existing, and in this simulation that gap turns out to be
    small — see EVALUATION.md, where the reason is that the synthetic world's
    effects are multiplicative and therefore close to additive in log-odds, which
    is precisely the shape a logistic model is already able to fit. The stage is
    kept rather than deleted because that near-null result is a property of the
    simulation and not a property of production data, but the honest headline is
    that this evaluation cannot demonstrate its value.
    """

    raw: CalibrationReport
    calibrated: CalibrationReport

    def as_text(self) -> str:
        return "\n\n".join(
            [
                self.raw.as_text(),
                self.calibrated.as_text(),
                f"calibration stage moved Brier {self.raw.brier:.4f} -> "
                f"{self.calibrated.brier:.4f} and ECE {self.raw.ece:.4f} -> "
                f"{self.calibrated.ece:.4f}",
            ]
        )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, ties averaged. Returns 0.5 when one class is absent."""
    positives = labels == 1.0
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks within tied score groups so a model that outputs one constant
    # scores 0.5 rather than something spurious.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(unique))
    np.add.at(tie_sum, inverse, ranks)
    ranks = (tie_sum / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ece(predictions: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - observed| weighted by bin size."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predictions >= low) & (predictions < high if high < 1.0 else predictions <= 1.0)
        count = int(mask.sum())
        if count == 0:
            continue
        total += count * abs(float(predictions[mask].mean() - labels[mask].mean()))
    return total / max(len(predictions), 1)


def _reliability(
    predictions: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> tuple[tuple[float, float, float, int], ...]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[tuple[float, float, float, int]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predictions >= low) & (predictions < high if high < 1.0 else predictions <= 1.0)
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            (float(low), float(predictions[mask].mean()), float(labels[mask].mean()), count)
        )
    return tuple(out)


class CalibratedLogisticEstimator:
    """The fitted estimator. Construct with `train()`, never directly.

    Holds the fitted objects and nothing else — no data, no clock, no IO — so it
    can be dropped into the orchestrator in place of `HeuristicEstimator` with no
    other change. `diagnostics` is attached for reporting and is not consulted by
    `estimate`.
    """

    name = "calibrated_logistic"

    def __init__(
        self,
        *,
        standardiser: _Standardiser,
        weights: np.ndarray,
        bias: float,
        calibrator: IsotonicCalibrator,
        diagnostics: ModelDiagnostics | None = None,
    ) -> None:
        self._standardiser = standardiser
        self._weights = weights
        self._bias = bias
        self._calibrator = calibrator
        self.diagnostics = diagnostics

    # -- fitting ------------------------------------------------------------

    @classmethod
    def train(
        cls,
        rows: Sequence[LabelledAttempt],
        *,
        l2: float = 1e-3,
        epochs: int = 4000,
        learning_rate: float = 0.5,
        calibration_share: float = 0.3,
        seed: int = 20260821,
    ) -> CalibratedLogisticEstimator:
        """Fit weights on one fold and the calibration curve on another.

        The split is a deterministic permutation from a fixed seed rather than a
        contiguous slice, because the training rows arrive grouped by case and a
        contiguous split would put whole causes on one side of the fold.
        """
        if len(rows) < 50:
            raise ValueError(f"need at least 50 labelled attempts to fit, got {len(rows)}")

        x, y = _design(rows)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(rows))
        cut = int(len(rows) * (1.0 - calibration_share))
        fit_index, calibration_index = permutation[:cut], permutation[cut:]

        standardiser = _Standardiser.fit(x[fit_index])
        weights, bias = _fit_weights(
            standardiser.apply(x[fit_index]),
            y[fit_index],
            l2=l2,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        raw_calibration = _sigmoid(standardiser.apply(x[calibration_index]) @ weights + bias)
        calibrator = IsotonicCalibrator.fit(raw_calibration, y[calibration_index])

        model = cls(
            standardiser=standardiser,
            weights=weights,
            bias=bias,
            calibrator=calibrator,
        )
        # Diagnostics are computed on the calibration fold. That fold is out of
        # sample for the weights but in sample for the isotonic curve, so the
        # calibrated Brier and ECE reported here are optimistic by construction.
        # `evaluate()` exists to be run on a third, genuinely untouched batch,
        # and that is the number EVALUATION.md quotes.
        model.diagnostics = model.evaluate(
            [rows[i] for i in calibration_index], label="calibration fold"
        )
        return model

    # -- inference ----------------------------------------------------------

    def _raw(self, matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(self._standardiser.apply(matrix) @ self._weights + self._bias)

    def estimate(
        self,
        case: RecoveryCase,
        diagnosis: Diagnosis,
        *,
        at: datetime,
        downtime_active: bool,
    ) -> Mapping[ActionKind, Decimal]:
        actions = diagnosis.candidate_actions
        if not actions:
            return {}
        matrix = np.asarray(
            [
                features(case, diagnosis, action, at=at, downtime_active=downtime_active)
                for action in actions
            ],
            dtype=np.float64,
        )
        calibrated = self._calibrator.apply(self._raw(matrix))
        return {
            action: _to_decimal(float(probability))
            for action, probability in zip(actions, calibrated)
        }

    # -- reporting ----------------------------------------------------------

    def evaluate(self, rows: Sequence[LabelledAttempt], *, label: str = "") -> ModelDiagnostics:
        """Score a set of labelled attempts, raw and calibrated side by side."""
        x, y = _design(rows)
        raw = self._raw(x)
        calibrated = self._calibrator.apply(raw)
        suffix = f" [{label}]" if label else ""
        return ModelDiagnostics(
            raw=score_predictions(raw, y, name=f"logistic, uncalibrated{suffix}"),
            calibrated=score_predictions(calibrated, y, name=f"logistic, isotonic{suffix}"),
        )

    def coefficients(self) -> tuple[tuple[str, float], ...]:
        """Standardised weights, largest first. For the write-up, not for logic.

        Standardised, so magnitudes are comparable across features; they are not
        odds ratios and should not be read as causal.
        """
        pairs = list(zip(FEATURE_NAMES, (float(w) for w in self._weights)))
        pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return tuple(pairs)


def _to_decimal(probability: float) -> Decimal:
    """Cross back into Decimal at the boundary, clamped away from 0 and 1.

    Everything downstream of this point multiplies probabilities by money, and
    the codebase's rule is that money arithmetic never touches a float. Three
    decimal places is well past the precision the model has any claim to.
    """
    clamped = min(0.99, max(0.001, probability))
    return Decimal(str(round(clamped, 3)))


def coefficient_table(model: CalibratedLogisticEstimator, *, top: int = 12) -> str:
    rows = ["| Feature | Standardised weight |", "|---|---|"]
    for name, weight in model.coefficients()[:top]:
        rows.append(f"| {name} | {weight:+.3f} |")
    return "\n".join(rows)


def as_rows(observations: Iterable[tuple[tuple[float, ...], bool]]) -> list[LabelledAttempt]:
    return [LabelledAttempt(features=f, succeeded=s) for f, s in observations]
