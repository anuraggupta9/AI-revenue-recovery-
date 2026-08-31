# Recoup

A bounded recovery agent for failed payments and failed subscription mandates.
Razorpay Buildathon, Track 03 — AI Revenue Recovery.

A failed payment is not one problem. An empty account, an expired card, an
issuer outage and a customer who closed the tab all arrive at the merchant as
the same thing: a payment that did not go through. The standard response is a
fixed retry ladder — try again in 24 hours, then 48, then 72 — which treats all
four identically and is wrong about three of them. Recoup reads the failure's
root cause, decides whether any action is worth taking at all, takes at most
one, and writes down why in a log that cannot be quietly edited afterwards.

The part I care most about is the second step. Most of the value here is in the
cases where the right answer is to do nothing, and a recovery agent that cannot
decline is a machine for annoying customers at scale.

## Results

Two thousand synthetic failures, 45-day horizon, seed 20260821. Every arm holds
back 20% of its cases untouched, and incremental recovery is measured against
that arm's own holdout — so the money customers would have paid unprompted is
excluded rather than claimed. Intervals are 2,000-sample bootstrap percentiles.

| Arm | Gross rate | Holdout rate | Treated rate | ₹ incremental | 95% interval | Attempts | Customer contacts | ₹ spent |
|---|---|---|---|---|---|---|---|---|
| Do nothing | 15.9% | 15.9% | 15.9% | 0 | — | 0 | 0 | 0 |
| Fixed retry ladder (+24/48/72h) | 28.7% | 21.9% | 30.4% | 708,965 | −482,286 to 1,860,466 | 4,091 | 0 | 8,182 |
| Diagnosis, no policy layer | 33.9% | 21.9% | 37.0% | 1,257,038 | 8,938 to 2,444,853 | 3,750 | 1,016 | 5,947 |
| **Recoup** | **40.1%** | 21.9% | **44.9%** | **1,902,805** | 694,191 to 3,122,269 | 3,004 | 889 | 4,754 |
| Recoup (fitted model) | 39.2% | 21.9% | 43.7% | 1,808,783 | 603,869 to 3,007,577 | 2,741 | 624 | 4,692 |

Reproduce with `python run.py compare --learned`. Every figure in this file and
in [EVALUATION.md](EVALUATION.md) comes from a command in this repository; a
number that cannot be regenerated is a claim, not a result.

Three things in that table matter more than the headline percentage.

The fixed ladder's incremental effect cannot be distinguished from zero. Its
gross recovery of 28.7% looks like most of the way to Recoup's 40.1%, and almost
all of that gap is customers who would have paid anyway. This is the single
easiest way to overstate a recovery system, and it is why every arm here has its
own control group rather than a shared baseline.

Recoup beats the same system with its policy layer removed while taking *fewer*
actions — 3,004 attempts against 3,750, and 713 wasted customer contacts against
840. The bounds are not a safety tax paid out of performance; on this data they
are where the performance comes from.

The fitted model does not recover more money than the hand-written priors
(₹1.81M against ₹1.90M, intervals overlapping almost entirely). It contacts 30%
fewer customers to get there — but not for the reason I first assumed. Set the
contact-probability floor to zero and the two estimators land within ₹321 of each
other. Calibration is not buying restraint directly; it is making a threshold that
was **very nearly inert** under the optimistic priors start firing — the 5% contact
floor blocks 2 actions under the heuristic and 585 under the calibrated model. A
policy layer full of thresholds can look thorough and barely be doing anything,
because whether a threshold fires depends on the calibration of the number fed to
it. That finding, and what it costs, is in [EVALUATION.md](EVALUATION.md) — which
is mostly about the places this project's numbers should not be trusted.

## How it works

**Diagnose.** Razorpay's failure payloads carry `error_source`, `error_step` and
`error_reason`. Those three fields already encode most of what a root-cause
classifier would try to infer, so diagnosis here is a lookup against a taxonomy
map, not a model. `error_source=bank` plus `error_reason=insufficient_funds` is
an empty account; `error_source=customer` plus `payment_timeout` is somebody who
walked away. Nine root causes, each carrying a confidence and the set of actions
that could plausibly work. An unmapped reason returns confidence zero and
candidate actions of none, which routes the case to a human rather than to a
guess.

**Decide.** `recoup.policy.engine.decide()` is a pure function: no clock, no IO,
no mutation. Five global rules can stop a case outright — circuit breaker,
customer opt-out, categorical no-retry, confidence floor, attempt cap. Then each
candidate action is walked best-first through seven per-action rules: retry
spacing, contact frequency, quiet hours (09:00–19:00 IST), the
insufficient-funds cool-off that lands retries on the salary window rather than
on a fixed +24h, known issuer downtime, a probability floor for anything that
spends customer attention, and finally the expected-value gate. At most one
action survives. A blocked action that time could unblock produces a deferral
rather than a stop, because waiting preserves the chance of recovery and
stopping forecloses it.

Purity is what makes the holdout arm nearly free: a control case runs the same
`decide()` and the same rules, and the only difference is that the orchestrator
records the decision as a shadow entry instead of executing it. The counterfactual
is a logged decision, not an assumption.

**Log.** Every rule evaluation, expected-value computation, declined action and
state transition goes into an append-only log where each entry carries the hash
of its predecessor. Editing or deleting any historical entry invalidates every
hash after it, and `python run.py verify --tamper` demonstrates exactly that.

The passes are logged as well as the blocks, which is what makes the trail an
explanation rather than a receipt. Two real lines from `data/`:

```
ev_floor   passed=True    expected value ₹82.46 for retry_alternate_rail clears the floor
ev_floor   passed=False   expected value -₹2.73 for send_payment_link is below
                          the floor ₹1.00 at p=0.043
```

Logging the passes also turned out to be load-bearing for the evaluation rather
than just for the audit: counting them is how I found that the 5% contact floor
blocks 2 actions under the hand-written priors and 585 under the calibrated model.
A log that records only blocks cannot tell a rule that is protecting you from a
rule that is barely doing anything.

Amounts are integer paise throughout. There is no float arithmetic on money
anywhere in this repository.

## Running it

The domain, audit, diagnosis and policy layers import only the standard library,
so the interesting half needs no install step:

```
python -m unittest discover -s tests -q       # 131 tests
python run.py demo                            # narrated case walkthrough
python run.py verify --tamper                 # audit chain, then break it
python run.py compare                         # the results table above
python run.py sensitivity                     # the same table across perturbed worlds
```

That zero-dependency boundary is asserted by
`tests/test_core_has_no_dependencies.py`, which parses the import graph rather
than importing it — a test that inspected `sys.modules` would pass on a machine
that simply happened not to have numpy installed.

The propensity model needs numpy:

```
pip install -e ".[model]"
python run.py model --reliability             # calibration against the oracle
python run.py compare --learned               # adds the fitted-model arm
```

## What is not here

**No live Razorpay integration.** I have no test-mode keys, so
`recoup.gateway.base` defines the adapter interface and the only implementation
is the deterministic in-process mock the simulation drives. The interface is
shaped around the real API — idempotency keys on every write, the
`error_source`/`error_step`/`error_reason` triple on reads — but "shaped around"
is not "verified against", and the first day with real keys will find things.

**No HTTP surface and no dashboard.** Both were planned; neither exists. I could
not install or run either FastAPI or npm in the environment I built this in, and
shipping a web layer I have never executed seemed worse than shipping none. The
`compare`, `demo` and `verify` commands are the interface.

**The data is synthetic, and I wrote the world.** This is the limitation that
subsumes the others. `recoup.simulation.world.GroundTruth` contains my estimates
of how often an insufficient-funds retry succeeds on payday versus mid-month, how
often a customer responds to a payment link, and so on. The agent is then scored
against those estimates. `python run.py sensitivity` re-runs everything with every
probability scaled by 0.6, 0.8 and 1.25, and the ordering survives — but that
tests robustness to my numbers being *wrong by a factor*, not to their being
wrong in *shape*. [EVALUATION.md](EVALUATION.md) is explicit about what this does
and does not establish.

## Layout

```
recoup/domain        value objects, entities, the recovery state machine
recoup/audit         hash-chained decision log
recoup/diagnosis     failure taxonomy -> named root cause
recoup/policy        stopping rules, expected-value gate, pure decide()
recoup/model         logistic propensity model, isotonic calibration  (numpy)
recoup/gateway       payment provider adapter interface
recoup/simulation    synthetic world, arms, orchestrator, metrics
```

Each layer may import the ones above it and never the ones below. The ordering is
enforced by a test, which is how it survived a genuine circular dependency
between the gateway and the simulation — see
[docs/BUILD_LOG.md](docs/BUILD_LOG.md).
