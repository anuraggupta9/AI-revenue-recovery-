"""Sweep min_contact_probability on the reference batch.

Answers the question recoup/policy/rules.py points at: what does raising the
contact floor above 5% actually cost in recovered revenue, and what does it buy in
contacts not made? Run for both estimators, because the floor is a threshold on a
probability and the two estimators disagree about the low end.

    python3 sweep_contact_floor.py

Writes nothing; prints two markdown tables.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from recoup.simulation.arms import ARMS, run_arm
from recoup.simulation.generate import generate
from recoup.simulation.metrics import summarise

FLOORS = (Decimal("0.00"), Decimal("0.05"), Decimal("0.10"), Decimal("0.15"), Decimal("0.20"))
BY_KEY = {arm.key: arm for arm in ARMS}


def sweep(arm, batch) -> list[tuple[Decimal, int, int, int]]:
    rows = []
    for floor in FLOORS:
        base = arm.config_factory()

        def config_factory(_floor=floor, _base=base):
            return dataclasses.replace(_base, min_contact_probability=_floor)

        result = run_arm(
            dataclasses.replace(arm, config_factory=config_factory),
            batch,
            log_path=None,
        )
        m = summarise(result, key=arm.key, label=arm.label)
        rows.append((floor, m.recovered_incremental.paise, m.contacts, m.wasted_contacts))
    return rows


def render(title: str, rows: list[tuple[Decimal, int, int, int]]) -> None:
    print(f"\n{title}\n{'=' * len(title)}")
    print("| Contact floor | ₹ incremental | Contacts | Wasted | ₹ given up vs 0% | Contacts avoided | ₹ per contact avoided |")
    print("|---|---|---|---|---|---|---|")
    base_money, base_contacts = rows[0][1], rows[0][2]
    for floor, money, contacts, wasted in rows:
        given_up = base_money - money
        avoided = base_contacts - contacts
        per = f"{given_up / avoided / 100:,.0f}" if avoided > 0 else "-"
        print(
            f"| {floor:.0%} | {money / 100:,.0f} | {contacts:,} | {wasted:,} "
            f"| {given_up / 100:,.0f} | {avoided:,} | {per} |"
        )


def main() -> int:
    batch = generate(count=2000, seed=20260821, horizon_days=45)
    render("Heuristic estimator", sweep(BY_KEY["recoup"], batch))
    try:
        from recoup.model import load_logistic
        from recoup.simulation.arms import learned_arm
        from recoup.simulation.training_data import to_rows, training_and_validation
    except ImportError:
        print("\nnumpy absent; skipped the fitted-model sweep")
        return 0
    # Same fit as `run.py compare --learned`: count * 3 exploration rows, same seed.
    train, _ = training_and_validation(count=6000, seed=20260821)
    estimator = load_logistic().CalibratedLogisticEstimator.train(to_rows(train))
    render("Calibrated logistic estimator", sweep(learned_arm(estimator), batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
