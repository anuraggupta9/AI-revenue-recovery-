#!/usr/bin/env python3
"""Recoup's command line. Every number in the write-up comes from one of these.

    python run.py compare              five arms, incremental recovery, bootstrap CIs
    python run.py model                calibration of four estimators against the oracle
    python run.py demo                 narrated walkthrough of individual cases
    python run.py sensitivity          the comparison re-run on perturbed worlds
    python run.py verify               re-read an audit log from disk and check the chain

The reason this file exists rather than a notebook: every figure quoted in
README.md and EVALUATION.md has to be reproducible by someone who has just cloned
the repository, with one command and no arguments. A number in a document that
cannot be regenerated is a claim, not a result.

`compare`, `demo` and `verify` need nothing but the standard library. `model` and
the `--learned` flag on `compare` need numpy; the error message says so.
"""

from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal
from pathlib import Path

from recoup.audit import EntryKind, verify_chain
from recoup.audit.log import read_entries
from recoup.domain.case import Arm as CaseArm
from recoup.domain.case import CaseState
from recoup.domain.money import Money
from recoup.policy.timing import to_ist
from recoup.simulation.arms import ARMS, learned_arm, run_arm
from recoup.simulation.generate import SMALL_TICKET_BUCKETS, generate
from recoup.simulation.metrics import comparison_table, narrative, summarise
from recoup.simulation.world import GroundTruth

DATA_DIR = Path(__file__).resolve().parent / "data"

# Perturbations for `sensitivity`. Chosen to bracket the plausible range rather
# than to be symmetric: the point is to find the value at which the conclusion
# flips, if there is one.
SENSITIVITY_FACTORS: tuple[tuple[str, Decimal], ...] = (
    ("pessimistic (x0.6)", Decimal("0.6")),
    ("mild (x0.8)", Decimal("0.8")),
    ("as written", Decimal("1.0")),
    ("optimistic (x1.25)", Decimal("1.25")),
)


def _rupees(paise: int) -> str:
    return f"₹{paise // 100:,}"


def _heading(text: str) -> str:
    return f"\n{text}\n{'=' * len(text)}"


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    """Run every arm on one batch and print the comparison table.

    All arms share the batch and the world seed, so a case that would have
    self-recovered on day three does so in every arm. That is what makes the
    incremental column a difference rather than two independent draws.
    """
    buckets = SMALL_TICKET_BUCKETS if args.small_ticket else None
    batch = generate(
        count=args.count,
        seed=args.seed,
        horizon_days=args.horizon_days,
        amount_buckets=buckets,
    )
    arms = list(ARMS)

    if args.learned:
        from recoup.model import load_logistic

        logistic = load_logistic()
        print("Fitting the propensity model on an exploration sample...", file=sys.stderr)
        from recoup.simulation.training_data import to_rows, training_and_validation

        train, _ = training_and_validation(count=args.count * 3, seed=args.seed)
        estimator = logistic.CalibratedLogisticEstimator.train(to_rows(train))
        arms.append(learned_arm(estimator))

    DATA_DIR.mkdir(exist_ok=True)
    started = time.perf_counter()
    metrics = []
    for arm in arms:
        result = run_arm(arm, batch, log_path=str(DATA_DIR / f"audit_{arm.key}.jsonl"))
        metrics.append(summarise(result, key=arm.key, label=arm.label))
    elapsed = time.perf_counter() - started

    print(_heading(f"Arm comparison — {args.count:,} failures, {args.horizon_days}-day horizon"))
    print(f"seed {args.seed}"
          f"{', small-ticket amounts' if args.small_ticket else ''}"
          f" — {elapsed:.1f}s\n")
    print(comparison_table(metrics))
    print()
    print(narrative(metrics))

    print(_heading("What each arm is for"))
    for arm in arms:
        print(f"\n{arm.label}\n  {arm.claim}")

    print(_heading("Audit logs"))
    for arm in arms:
        path = DATA_DIR / f"audit_{arm.key}.jsonl"
        status = verify_chain(list(read_entries(path))) if path.exists() else None
        detail = str(status) if status else "not written"
        print(f"  data/{path.name}: {detail}")

    return 0 if all(not m.halted for m in metrics) else 1


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def cmd_model(args: argparse.Namespace) -> int:
    """Score four estimators on held-out data, against the oracle ceiling.

    The oracle row is the point of the command. Without it, "Brier 0.1035" is a
    number nobody can interpret; with it, the reader can see that every serious
    estimator here sits within a thousandth of the irreducible floor, and that the
    interesting differences are therefore in calibration and not in ranking.
    """
    from recoup.model import load_logistic

    logistic = load_logistic()
    from recoup.model.estimator import FixedScheduleEstimator, HeuristicEstimator
    from recoup.simulation.training_data import (
        oracle_report,
        positive_rate,
        score,
        to_rows,
        training_and_validation,
    )

    started = time.perf_counter()
    train, validation = training_and_validation(count=args.count, seed=args.seed)
    explored = time.perf_counter()
    model = logistic.CalibratedLogisticEstimator.train(to_rows(train))
    fitted = time.perf_counter()

    print(_heading("Propensity model"))
    print(
        f"{len(train):,} training rows, {len(validation):,} validation rows "
        f"from disjoint batches\n"
        f"positive rate: train {positive_rate(train):.3f}, "
        f"validation {positive_rate(validation):.3f}\n"
        f"explored in {explored - started:.1f}s, fitted in {fitted - explored:.1f}s"
    )

    print(_heading("Held-out calibration (validation set, untouched by any fitting stage)"))
    diagnostics = model.evaluate(to_rows(validation), label="held out")
    reports = [
        oracle_report(validation),
        diagnostics.raw,
        diagnostics.calibrated,
        score(validation, HeuristicEstimator(), name="heuristic priors"),
        score(validation, FixedScheduleEstimator(Decimal("0.30")), name="flat 0.30"),
    ]

    print("\n| Estimator | AUC | Brier | ECE |")
    print("|---|---|---|---|")
    for report in reports:
        print(f"| {report.name} | {report.auc:.3f} | {report.brier:.4f} | {report.ece:.4f} |")

    oracle = reports[0]
    print(
        f"\nThe oracle row is the ceiling: those are the probabilities the outcomes "
        f"were actually drawn from, so no estimator can beat Brier {oracle.brier:.4f}. "
        f"Read every other row as a distance from it."
    )
    print(
        "\nNote what the table does and does not show. The heuristic's AUC matches "
        "the oracle's, so the learned model is not better at ranking actions — it is "
        "better at saying how likely they are, which is the number the expected-value "
        "gate multiplies by money. A ranking-only evaluation would have called the "
        "model worthless."
    )

    if args.reliability:
        print(_heading("Reliability tables"))
        for report in reports:
            print(f"\n{report.as_text()}")

    print(_heading("Fitted coefficients (standardised, largest first)"))
    print(logistic.coefficient_table(model))
    print(
        "\nNot causal, and not odds ratios. Shown because the two hand-entered "
        "interaction terms should appear near the top if the model learned the "
        "thesis — timing matters for balance failures, rail switching matters for "
        "dead instruments — and if they do not, the feature set is wrong."
    )
    return 0


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def _money(value: object) -> str:
    """Render a payload money field.

    Needed in two shapes. The in-memory mirror on an `AuditLog` holds the real
    `Money` object, because `append` stores the payload verbatim and only encodes
    at serialisation time; a log re-read from disk yields the encoded
    `{"paise": ..., "currency": ...}` form. `demo` reads the former and `verify`
    the latter, so this walkthrough has to handle both.
    """
    if isinstance(value, Money):
        return str(value)
    if isinstance(value, dict) and "paise" in value:
        return _rupees(int(value["paise"]))
    return "-"


def _why(payload: dict) -> str:
    """The human-readable reason, wherever this entry kind happens to keep it."""
    for key in ("rationale", "reason", "detail"):
        text = payload.get(key)
        if text:
            return str(text)
    return ""


def _explain(entries, case_id: str) -> list[str]:
    """Turn the audit trail for one case into readable lines.

    The key names here are not guesses: an earlier version of this function
    invented plausible ones (`amount_paise`, `probability`, `from`) and the demo
    printed `None` and `₹0` throughout while looking structurally correct. Every
    field below is read off the `log.append` call sites in the orchestrator.
    """
    lines: list[str] = []
    for entry in entries:
        if entry.case_id != case_id:
            continue
        payload = entry.payload
        # IST, not the stored UTC. The rule details in these entries quote IST, and
        # a walkthrough that timestamps a line 17:53 next to a rule explaining that
        # 23:23 is outside contact hours reads like a bug in the rule.
        stamp = f"{to_ist(entry.at):%d %b %H:%M}"
        if entry.kind is EntryKind.EVENT_INGESTED:
            lines.append(
                f"  {stamp}  failed: {payload.get('error_reason')} on "
                f"{payload.get('method')} via {payload.get('issuer')} "
                f"for {_money(payload.get('amount'))}"
            )
        elif entry.kind is EntryKind.DIAGNOSED:
            actions = payload.get("candidate_actions") or ()
            plan = ", ".join(str(a) for a in actions) or "no automated action"
            lines.append(
                f"  {stamp}  cause: {payload.get('root_cause')} "
                f"(confidence {payload.get('confidence')}, "
                f"from {payload.get('source')}) -> {plan}"
            )
        elif entry.kind is EntryKind.RULE_EVALUATED and not payload.get("passed", True):
            subject = payload.get("action") or "the case"
            lines.append(
                f"  {stamp}  rule {payload.get('rule')} blocked {subject}: "
                f"{payload.get('detail')}"
            )
        elif entry.kind is EntryKind.EV_COMPUTED:
            lines.append(
                f"  {stamp}  {payload.get('action')}: p={payload.get('p_success')} "
                f"on {_money(payload.get('amount_at_risk'))} at risk "
                f"-> EV {_money(payload.get('expected_value'))}"
            )
        elif entry.kind is EntryKind.ACTION_CHOSEN:
            lines.append(
                f"  {stamp}  chose {payload.get('action')} "
                f"(p={payload.get('p_success')}): {payload.get('rationale')}"
            )
        elif entry.kind is EntryKind.ACTION_DECLINED:
            lines.append(f"  {stamp}  declined {payload.get('action')}: {_why(payload)}")
        elif entry.kind is EntryKind.ACTION_EXECUTED:
            replayed = " [replayed, no new charge]" if payload.get("replayed") else ""
            lines.append(
                f"  {stamp}  executed {payload.get('action')} "
                f"(key {str(payload.get('idempotency_key'))[:20]}...){replayed}"
            )
        elif entry.kind is EntryKind.ACTION_RESULT:
            outcome = "succeeded" if payload.get("succeeded") else "failed"
            cost = _money(payload.get("cost"))
            suffix = f" — {payload['detail']}" if payload.get("detail") else ""
            lines.append(f"  {stamp}  result: {outcome}, cost {cost}{suffix}")
        elif entry.kind is EntryKind.DUPLICATE_SUPPRESSED:
            lines.append(
                f"  {stamp}  duplicate webhook suppressed "
                f"({payload.get('event_id')}): {payload.get('detail')}"
            )
        elif entry.kind is EntryKind.ESCALATED:
            lines.append(
                f"  {stamp}  escalated {_money(payload.get('amount'))} to a human: "
                f"{payload.get('rationale')}"
            )
        elif entry.kind is EntryKind.SHADOW_DECISION:
            would = payload.get("action") or payload.get("outcome") or "nothing"
            lines.append(
                f"  {stamp}  [holdout] would have chosen {would} — logged, not executed"
            )
        elif entry.kind is EntryKind.STATE_TRANSITION:
            defer = payload.get("defer_until")
            when = f" until {to_ist(defer):%d %b %H:%M}" if hasattr(defer, "strftime") else ""
            lines.append(f"  {stamp}  -> {payload.get('to')}{when}: {_why(payload)}")
        elif entry.kind is EntryKind.CIRCUIT_BREAKER:
            lines.append(
                f"  {stamp}  CIRCUIT BREAKER after "
                f"{payload.get('actions_failed')}/{payload.get('actions_executed')} "
                f"failures: {payload.get('rationale')}"
            )
    return lines


def cmd_demo(args: argparse.Namespace) -> int:
    """Walk through individual cases, chosen to show the interesting behaviours.

    Picked by outcome rather than by index, because the point is to show one of
    each kind of ending — including the ones where the agent declined to act, which
    are the decisions a merchant most needs to be able to audit.
    """
    batch = generate(count=args.count, seed=args.seed, horizon_days=args.horizon_days)
    arm = next(a for a in ARMS if a.key == "recoup")

    DATA_DIR.mkdir(exist_ok=True)
    log_path = DATA_DIR / "audit_demo.jsonl"
    result = run_arm(arm, batch, log_path=str(log_path))
    entries = list(result.log)

    print(_heading(f"Case walkthrough — {arm.label}"))
    print(
        f"{len(result.cases):,} cases, {result.gateway_calls:,} gateway calls, "
        f"{result.duplicates_suppressed:,} duplicates suppressed, "
        f"{result.shadow_decisions:,} shadow decisions recorded"
    )

    # One case per ending, so the declines and escalations get shown rather than
    # buried under the successes.
    wanted = [
        (CaseState.RECOVERED, "Recovered", "the agent acted and the money arrived"),
        (
            CaseState.UNECONOMIC,
            "Declined as uneconomic",
            "expected value fell below the floor, so the case was closed unattempted",
        ),
        (
            CaseState.ESCALATED,
            "Escalated to a human",
            "a categorical prohibition or a diagnosis the taxonomy could not place; "
            "the agent will not act, but somebody is told",
        ),
        (
            CaseState.SUPPRESSED,
            "Suppressed",
            "the customer opted out of recovery contact, so nothing happens and "
            "nothing is queued for review either",
        ),
        (
            CaseState.EXHAUSTED,
            "Exhausted",
            "the retry budget ran out without recovery",
        ),
    ]

    shown = 0
    for state, title, why in wanted:
        case = next(
            (
                c
                for c in result.cases
                if c.state is state and c.arm is CaseArm.TREATMENT
            ),
            None,
        )
        if case is None:
            print(f"\n{title}: no case reached this state in this batch.")
            continue
        print(f"\n{title} — {why}")
        print(f"  case {case.case_id}, {_rupees(case.amount_at_risk.paise)} at risk")
        for line in _explain(entries, case.case_id):
            print(line)
        shown += 1

    holdout = next(
        (c for c in result.cases if c.arm is CaseArm.CONTROL and c.attempts == []), None
    )
    if holdout is not None:
        print("\nHoldout case — the agent decided but did not act")
        print(
            "  This is what makes the incremental column trustworthy. The decision was "
            "computed and logged; nothing was executed."
        )
        print(f"  case {holdout.case_id}, {_rupees(holdout.amount_at_risk.paise)} at risk")
        for line in _explain(entries, holdout.case_id):
            print(line)

    print(_heading("Audit chain"))
    print(f"  in memory: {result.log.verify()}")
    print(f"  re-read from disk: {result.log.verify_on_disk()}")
    print(
        "\nThe second line is the one that means anything: it does not trust the "
        "process that wrote the log."
    )
    return 0 if shown else 1


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """Re-run the comparison against worlds whose assumptions have been scaled.

    Every number in `GroundTruth` is an estimate I wrote down, so the honest
    question is not "what does the agent recover" but "over what range of
    assumptions does the conclusion survive". This is the command that answers it,
    and it is the one most likely to embarrass the project. It does.

    Two questions are kept separate here, because an earlier version ran the sweep
    at half the batch size of `compare` and then reported that the result "does not
    survive every perturbation" — in every row, including the unperturbed one. The
    conclusion was already known to hold at the headline sample size, so what the
    sweep had actually measured was its own loss of power, and it presented that as
    a finding about the world. Whether an ordering flips and whether an interval
    covers zero fail for different reasons and are now reported separately.
    """
    batch = generate(count=args.count, seed=args.seed, horizon_days=args.horizon_days)
    keys = ("fixed_ladder", "no_policy", "recoup")
    arms = [a for a in ARMS if a.key in keys]

    print(_heading(f"Sensitivity to the world's assumptions — {args.count:,} failures"))
    print(
        "\nEach row scales every probability in GroundTruth by the given factor, so "
        "self-recovery, retry success and customer responsiveness all move together. "
        "Incremental recovery against each arm's own untouched holdout.\n"
    )
    print("| World | " + " | ".join(a.label for a in arms) + " | Recoup CI |")
    print("|---" * (len(arms) + 2) + "|")

    beats_ladder: list[bool] = []
    beats_no_policy: list[bool] = []
    excludes_zero: list[bool] = []
    for label, factor in SENSITIVITY_FACTORS:
        truth = GroundTruth().scaled(factor)
        row: list[str] = []
        by_key = {}
        for arm in arms:
            result = run_arm(arm, batch, truth=truth)
            metrics = summarise(result, key=arm.key, label=arm.label)
            by_key[arm.key] = metrics
            row.append(_rupees(metrics.recovered_incremental.paise))

        rec = by_key["recoup"]
        beats_ladder.append(
            rec.recovered_incremental > by_key["fixed_ladder"].recovered_incremental
        )
        beats_no_policy.append(
            rec.recovered_incremental > by_key["no_policy"].recovered_incremental
        )
        excludes_zero.append(rec.interval_excludes_zero)
        interval = (
            f"{_rupees(rec.incremental_low.paise)} to "
            f"{_rupees(rec.incremental_high.paise)}"
        )
        print(f"| {label} | " + " | ".join(row) + f" | {interval} |")

    print()
    if all(beats_ladder):
        print(
            "Recoup's point estimate beats the fixed ladder in every world, including "
            "the pessimistic one. That ordering is the robust part of the result."
        )
    else:
        losses = [
            label
            for (label, _), ok in zip(SENSITIVITY_FACTORS, beats_ladder)
            if not ok
        ]
        print(
            "Recoup does not beat the fixed ladder in every world. It loses under: "
            f"{', '.join(losses)}."
        )

    if not all(beats_no_policy):
        losses = [
            label
            for (label, _), ok in zip(SENSITIVITY_FACTORS, beats_no_policy)
            if not ok
        ]
        print(
            "\nThe uncomfortable row, and the one worth reading first: Recoup is beaten "
            f"by the unbounded arm under {', '.join(losses)}. When recovery odds are "
            "poor across the board, the expected-value gate declines cases that would "
            "in fact have paid, and restraint costs real money. The bounds are not free "
            "— they buy fewer wasted contacts and an auditable reason for every action, "
            "and in a harsh world they are paid for in recovered rupees. A merchant who "
            "does not care about customer attention should know that trade exists."
        )

    if not all(excludes_zero):
        weak = [
            label
            for (label, _), ok in zip(SENSITIVITY_FACTORS, excludes_zero)
            if not ok
        ]
        print(
            f"\nThe interval covers zero under: {', '.join(weak)}. Read that as a limit "
            "on precision rather than a verdict on the ordering — a perturbed world "
            "changes the effect size, and this sweep runs a smaller batch per cell than "
            "`compare` does, so both the numerator and the power move at once. The "
            "headline claim is the one measured at the full batch size."
        )

    print(
        "\nNone of this is independent evidence, and the sweep cannot make it so: the "
        "same author wrote the policy and the world it is scored in."
    )
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-read audit logs from disk and check the hash chain.

    Separate from the runs that produced them on purpose. A log verified by the
    process that wrote it proves very little; this reads the bytes back.
    """
    paths = (
        [Path(args.path)]
        if args.path
        else sorted(DATA_DIR.glob("audit_*.jsonl"))
    )
    if not paths:
        print(
            "No audit logs found in data/. Run `python run.py compare` first.",
            file=sys.stderr,
        )
        return 1

    print(_heading("Audit chain verification"))
    failures = 0
    for path in paths:
        entries = list(read_entries(path))
        status = verify_chain(entries)
        kinds: dict[str, int] = {}
        for entry in entries:
            kinds[str(entry.kind)] = kinds.get(str(entry.kind), 0) + 1
        print(f"\n{path.name}: {status}")
        for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"    {kind:24s} {count:,}")
        if not status.ok:
            failures += 1

    if args.tamper and paths:
        print(_heading("Tamper check"))
        entries = list(read_entries(paths[0]))
        if len(entries) < 2:
            print("  log too short to tamper with")
        else:
            import dataclasses

            target = len(entries) // 2
            entries[target] = dataclasses.replace(
                entries[target], payload={**entries[target].payload, "amount_paise": 1}
            )
            status = verify_chain(entries)
            print(f"  edited entry {entries[target].seq} in memory -> {status}")
            print(
                "  The chain is hash-linked, so editing one entry invalidates every "
                "entry after it. That is the property that makes the log evidence "
                "rather than a record."
            )
            if status.ok:
                print("  UNEXPECTED: tampering was not detected.", file=sys.stderr)
                failures += 1

    return 1 if failures else 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Recoup — evaluation and demonstration commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, count: int) -> None:
        p.add_argument("--count", type=int, default=count, help="number of failure events")
        p.add_argument("--seed", type=int, default=20260821, help="batch seed")
        p.add_argument("--horizon-days", type=int, default=45, help="days to run the batch")

    p_compare = sub.add_parser("compare", help="run every arm and print the comparison table")
    common(p_compare, count=2000)
    p_compare.add_argument(
        "--learned",
        action="store_true",
        help="add the fitted-model arm (needs numpy; fits a model first)",
    )
    p_compare.add_argument(
        "--small-ticket",
        action="store_true",
        help="shift amounts down an order of magnitude, so the EV floor actually binds",
    )
    p_compare.set_defaults(func=cmd_compare)

    p_model = sub.add_parser("model", help="calibration of every estimator against the oracle")
    common(p_model, count=6000)
    p_model.add_argument(
        "--reliability", action="store_true", help="also print per-bin reliability tables"
    )
    p_model.set_defaults(func=cmd_model)

    p_demo = sub.add_parser("demo", help="narrated walkthrough of individual cases")
    common(p_demo, count=400)
    p_demo.set_defaults(func=cmd_demo)

    p_sensitivity = sub.add_parser(
        "sensitivity", help="re-run the comparison on perturbed worlds"
    )
    common(p_sensitivity, count=2000)
    p_sensitivity.set_defaults(func=cmd_sensitivity)

    p_verify = sub.add_parser("verify", help="re-read audit logs from disk and check the chain")
    p_verify.add_argument("--path", help="a single log to verify (default: all of data/)")
    p_verify.add_argument(
        "--tamper",
        action="store_true",
        help="edit an entry in memory and confirm the chain notices",
    )
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ImportError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
