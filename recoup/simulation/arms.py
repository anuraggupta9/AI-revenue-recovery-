"""The arms being compared.

Each arm is the same run loop with different capabilities removed, which is the
only way a comparison like this means anything. Four dials distinguish them:
which diagnoser is used, which probability estimator, which rule set, and which
config. Nothing else differs — same generated batch, same world, same seed, same
executor.

A note on the third arm's name. The comparison people usually want here is
"against an LLM-only agent", and this arm is deliberately *not* called that,
because no language model runs in it. What it actually removes is the policy
layer: it diagnoses correctly, estimates a probability, and then acts on that
estimate with no expected-value gate, no quiet hours, no contact cap, no
cause-aware timing and no categorical prohibitions. That is the failure mode the
comparison is about — an agent with good intentions and no bounds — and removing
the bounds demonstrates it directly, without a network call and without claiming
a result I did not produce. `no_policy` is the honest label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Callable

from recoup.diagnosis import Diagnosis, RootCause, diagnose
from recoup.domain.case import ActionKind
from recoup.domain.events import FailureEvent
from recoup.audit import AuditLog
from recoup.simulation.mock_gateway import MockGateway
from recoup.model.estimator import (
    FixedScheduleEstimator,
    HeuristicEstimator,
    ProbabilityEstimator,
)
from recoup.policy.rules import FULL_RULES, MINIMAL_RULES, PolicyConfig, RuleSet
from recoup.simulation.generate import GeneratedBatch
from recoup.simulation.orchestrator import Orchestrator, RunResult
from recoup.simulation.world import GroundTruth, SimulatedWorld


def blind_diagnose(event: FailureEvent) -> Diagnosis:
    """What a fixed retry ladder "knows": nothing.

    Returns the same plan for every failure — retry the same rail — with
    confidence pinned high so the absence of a confidence check is not silently
    doing the work of one. An expired card and a temporarily empty account get
    identical treatment, which is the entire deficiency being measured.
    """
    return Diagnosis(
        root_cause=RootCause.UNKNOWN,
        confidence=Decimal("1.0"),
        rationale="fixed ladder: cause is not consulted",
        candidate_actions=(ActionKind.RETRY_SAME_RAIL,),
        source="blind",
    )


@dataclass(frozen=True, slots=True)
class Arm:
    """One configuration of the agent, plus what it is meant to demonstrate."""

    key: str
    label: str
    claim: str
    diagnoser: Callable[[FailureEvent], Diagnosis] = diagnose
    estimator_factory: Callable[[], ProbabilityEstimator] = HeuristicEstimator
    rules: RuleSet = FULL_RULES
    config_factory: Callable[[], PolicyConfig] = PolicyConfig
    # 1.0 means the entire batch is held out, i.e. the agent never acts.
    holdout_share: Decimal = Decimal("0.20")


def _fixed_ladder_config() -> PolicyConfig:
    """The +24/48/72h ladder most merchants actually run."""
    return PolicyConfig(
        max_attempts=3,
        retry_backoff=(timedelta(hours=24), timedelta(hours=48), timedelta(hours=72)),
        # Not used by MINIMAL_RULES, but set to zero so that if this config is ever
        # paired with the full rule set the ladder's indifference to customer
        # attention is explicit rather than inherited from the defaults.
        annoyance_cost=PolicyConfig().annoyance_cost.scale(Decimal("0")),
    )


ARMS: tuple[Arm, ...] = (
    Arm(
        key="do_nothing",
        label="Do nothing",
        claim=(
            "Establishes the counterfactual. Some failed payments are completed by "
            "the customer unprompted, and every rupee of that belongs in the "
            "baseline rather than in a recovery system's results."
        ),
        holdout_share=Decimal("1"),
    ),
    Arm(
        key="fixed_ladder",
        label="Fixed retry ladder (+24/48/72h)",
        claim=(
            "The common production baseline: three retries on a clock, same rail, "
            "cause ignored. Spends attempts on dead instruments and retries empty "
            "accounts before payday."
        ),
        diagnoser=blind_diagnose,
        estimator_factory=lambda: FixedScheduleEstimator(Decimal("0.30")),
        rules=MINIMAL_RULES,
        config_factory=_fixed_ladder_config,
    ),
    Arm(
        key="no_policy",
        label="Diagnosis, no policy layer",
        claim=(
            "Correct root cause and a probability estimate, then acts on it with no "
            "bounds. Isolates how much of the result comes from the policy engine "
            "rather than from knowing why the payment failed."
        ),
        rules=MINIMAL_RULES,
    ),
    Arm(
        key="recoup",
        label="Recoup (diagnosis + policy engine)",
        claim="The full system, with every stopping rule and the expected-value gate active.",
    ),
)

ARMS_BY_KEY: dict[str, Arm] = {arm.key: arm for arm in ARMS}


def learned_arm(estimator: ProbabilityEstimator) -> Arm:
    """The full policy driven by a fitted model instead of hand-set priors.

    Takes an already-trained estimator rather than training one, so the model is
    fitted once and the same weights drive every batch size the CLI is run at.
    Training inside a factory would refit per run and make two invocations of the
    same command incomparable.

    What this arm isolates is narrow and worth stating precisely. The learned
    model and the heuristic rank actions almost identically — on the validation
    set their AUCs agree to three decimal places — so this is not a test of
    whether the model knows more. It is a test of whether being *calibrated*
    changes what the expected-value gate authorises. The heuristic's priors are
    optimistic by a fairly consistent factor, and an optimistic probability
    multiplied by a rupee amount clears a fixed floor more often than it should.
    """
    return Arm(
        key="recoup_learned",
        label="Recoup (fitted model)",
        claim=(
            "Identical policy to Recoup, with the hand-set priors replaced by the "
            "calibrated logistic model. Isolates the value of calibration alone, "
            "since the two estimators rank actions the same way."
        ),
        estimator_factory=lambda: estimator,
    )


def run_arm(
    arm: Arm,
    batch: GeneratedBatch,
    *,
    truth: GroundTruth | None = None,
    log_path: str | None = None,
) -> RunResult:
    """Run one arm against a batch.

    Every arm gets a fresh world, gateway and log seeded identically from the
    batch, so the only differences between arms are the four dials on `Arm`. In
    particular the world's seed does not vary, which means a case that would have
    self-recovered on day three does so in every arm — the counterfactual is held
    fixed across the comparison rather than resampled per arm.
    """
    world = SimulatedWorld(truth or GroundTruth(), seed=batch.seed)
    gateway = MockGateway(world, downtime=batch.downtime)
    # Truncating rather than appending. A second run against an existing log would
    # produce a chain that verifies perfectly and describes two different runs, and
    # the metrics read off it would silently double-count.
    #
    # flush_each=False because this log is closed before anything re-reads it, and
    # the per-entry flush a live deployment needs costs about forty seconds per arm
    # here for a guarantee no reader is relying on.
    log = AuditLog(log_path, truncate=True, flush_each=False)
    orchestrator = Orchestrator(
        gateway=gateway,
        world=world,
        log=log,
        config=arm.config_factory(),
        estimator=arm.estimator_factory(),
        rules=arm.rules,
        diagnoser=arm.diagnoser,
        holdout_share=arm.holdout_share,
        label=arm.label,
    )
    result = orchestrator.run(
        batch.events,
        horizon_days=batch.horizon_days,
        start=batch.events[0].occurred_at if batch.events else None,
    )
    # Release the handle so the file is complete on disk before anything tries to
    # re-read and verify it. The in-memory mirror on `result.log` stays readable.
    log.close()
    return result
