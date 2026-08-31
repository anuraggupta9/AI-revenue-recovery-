# Recoup

**A bounded recovery agent for failed payments and subscription mandates.**

**Razorpay Buildathon — Track 03: AI Revenue Recovery**

A failed payment is not one problem. An empty account, an expired card, an issuer outage, and a customer who closed the tab all arrive at the merchant as the same thing: a payment that did not go through.

The standard response is a fixed retry ladder — try again in 24 hours, then 48, then 72. That treats fundamentally different failures identically, and is wrong about most of them.

**Recoup diagnoses the failure, determines whether intervention is worthwhile, takes at most one recovery action, and records the reasoning in a tamper-evident audit log.**

The key idea is restraint. A recovery agent that cannot decide *not* to act is a machine for annoying customers at scale.

---

## Results

The evaluation uses **2,000 synthetic payment failures** over a **45-day horizon** with seed `20260821`.

Every arm holds back 20% of its cases untouched. Incremental recovery is measured against that arm's own holdout, excluding the money customers would have paid without intervention. Intervals are 2,000-sample bootstrap percentiles.

| Arm                             | Gross rate | Holdout rate | Treated rate |  ₹ incremental |               95% interval |  Attempts | Customer contacts |    ₹ spent |
| ------------------------------- | ---------: | -----------: | -----------: | -------------: | -------------------------: | --------: | ----------------: | ---------: |
| Do nothing                      |      15.9% |        15.9% |        15.9% |             ₹0 |                          — |         0 |                 0 |         ₹0 |
| Fixed retry ladder (+24/48/72h) |      28.7% |        21.9% |        30.4% |       ₹708,965 |    −₹482,286 to ₹1,860,466 |     4,091 |                 0 |     ₹8,182 |
| Diagnosis, no policy layer      |      33.9% |        21.9% |        37.0% |     ₹1,257,038 |       ₹8,938 to ₹2,444,853 |     3,750 |             1,016 |     ₹5,947 |
| **Recoup**                      |  **40.1%** |    **21.9%** |    **44.9%** | **₹1,902,805** | **₹694,191 to ₹3,122,269** | **3,004** |           **889** | **₹4,754** |
| Recoup (fitted model)           |      39.2% |        21.9% |        43.7% |     ₹1,808,783 |     ₹603,869 to ₹3,007,577 |     2,741 |               624 |     ₹4,692 |

Reproduce the results with:

```bash
python run.py compare --learned
```

Every figure in this README and in [EVALUATION.md](EVALUATION.md) comes from a command in this repository. A number that cannot be regenerated is a claim, not a result.

### What the results show

The fixed retry ladder's incremental effect cannot be distinguished from zero. Its 28.7% gross recovery looks close to Recoup's 40.1%, but much of that recovery comes from customers who would have paid anyway. This is why every arm has its own control group rather than relying on a shared baseline.

Recoup outperforms the same diagnosis system without its policy layer while taking **fewer actions**: 3,004 attempts versus 3,750, with fewer customer contacts. The stopping rules are therefore not simply a safety cost; in this evaluation, they contribute to the recovery result.

The fitted model recovers ₹1.81M versus ₹1.90M for the hand-written priors, with substantially overlapping intervals. It contacts 30% fewer customers, but the reason is more subtle. Setting the contact-probability floor to zero brings the two estimators within ₹321 of each other. Calibration changes the number entering the policy thresholds: the 5% contact floor blocks 2 actions under the heuristic and 585 under the calibrated model.

That result is documented in [EVALUATION.md](EVALUATION.md), including the limitations of the simulation and the cases where these numbers should not be trusted.

---

## How it works

### 1. Diagnose

Razorpay failure payloads provide `error_source`, `error_step`, and `error_reason`. Recoup uses these fields to map failures to a root-cause taxonomy rather than attempting to infer information that the payment provider already provides.

For example:

```text
error_source=bank
error_reason=insufficient_funds
        ↓
empty account
```

and:

```text
error_source=customer
error_reason=payment_timeout
        ↓
customer abandoned payment
```

The taxonomy contains nine root causes. Each root cause carries a confidence value and the recovery actions that could plausibly work.

An unmapped failure returns zero confidence and no candidate actions, routing the case to human review rather than making an unsupported guess.

### 2. Decide

`recoup.policy.engine.decide()` is a pure function: no clock, no IO, and no mutation.

Five global rules can stop a case immediately:

* Circuit breaker
* Customer opt-out
* Categorical no-retry rule
* Confidence floor
* Attempt cap

Candidate actions are then evaluated best-first through seven per-action rules:

* Retry spacing
* Contact frequency
* Quiet hours (`09:00–19:00 IST`)
* Insufficient-funds cool-off aligned with the salary window
* Known issuer downtime
* Probability floor for actions that spend customer attention
* Expected-value gate

At most **one action** survives.

If an action is temporarily blocked but could become valid later, Recoup defers the case instead of stopping it. Waiting preserves the possibility of recovery; stopping removes it.

The holdout arm runs the same decision function and rules. The only difference is that the resulting decision is recorded as a shadow entry rather than executed. The counterfactual is therefore a logged decision rather than an assumption.

### 3. Audit

Every rule evaluation, expected-value computation, declined action, and state transition is written to an append-only log.

Each entry contains the hash of its predecessor. Editing or deleting a historical entry invalidates every subsequent hash.

Verify the chain and demonstrate tampering with:

```bash
python run.py verify --tamper
```

Both passed and failed rule evaluations are logged. This makes the audit trail an explanation of the decision rather than simply a record that an action occurred.

For example:

```text
ev_floor   passed=True    expected value ₹82.46 for retry_alternate_rail clears the floor

ev_floor   passed=False   expected value -₹2.73 for send_payment_link is below
                          the floor ₹1.00 at p=0.043
```

Logging successful rule evaluations also supports the evaluation itself. It revealed that the 5% contact-probability floor blocks 2 actions under the hand-written priors but 585 under the calibrated model.

All monetary amounts are represented as integer paise. There is no floating-point arithmetic on money in the repository.

---

## Running it

The domain, audit, diagnosis, and policy layers use only the Python standard library.

Run the test suite:

```bash
python -m unittest discover -s tests -q
```

Expected result:

```text
131 tests
```

Run the narrated walkthrough:

```bash
python run.py demo
```

Verify the audit chain and demonstrate tampering:

```bash
python run.py verify --tamper
```

Run the main evaluation:

```bash
python run.py compare
```

Run the sensitivity analysis:

```bash
python run.py sensitivity
```

The zero-dependency boundary is asserted by `tests/test_core_has_no_dependencies.py`, which parses the import graph rather than importing the modules.

### Learned model

The propensity model requires NumPy:

```bash
pip install -e ".[model]"
```

Run model calibration:

```bash
python run.py model --reliability
```

Run the evaluation including the fitted-model arm:

```bash
python run.py compare --learned
```

---

## Scope and limitations

### No live Razorpay integration

This submission does not connect to live Razorpay payment APIs.

`recoup.gateway.base` defines the payment-provider adapter interface, while the simulation uses a deterministic in-process mock gateway.

The interface is shaped around the payment-provider boundary, including idempotency keys on writes and the `error_source` / `error_step` / `error_reason` failure fields, but it has not been verified against live Razorpay credentials.

### No HTTP surface or dashboard

There is currently no HTTP API or dashboard. The repository is intentionally focused on the recovery engine, simulation, evaluation, and audit trail.

The primary interfaces are:

```bash
python run.py demo
python run.py compare
python run.py verify
```

### Synthetic evaluation data

The evaluation uses a synthetic world defined in `recoup.simulation.world.GroundTruth`.

It contains assumptions about recovery probabilities, including how often insufficient-funds retries succeed around salary windows and how often customers respond to payment links.

The agent is evaluated against these assumptions.

`python run.py sensitivity` reruns the evaluation with probabilities scaled by `0.6`, `0.8`, and `1.25`. The ordering of the approaches survives these perturbations.

This tests robustness to the assumed probabilities being wrong by a factor; it does not establish that the assumptions are correct in shape or representative of real payment traffic.

See [EVALUATION.md](EVALUATION.md) for the detailed methodology, limitations, and cases where the results should not be trusted.

---

## Architecture

```text
recoup/domain
    Value objects, entities, recovery state machine

recoup/audit
    Hash-chained decision log

recoup/diagnosis
    Failure taxonomy → named root cause

recoup/policy
    Stopping rules, expected-value gate, pure decide()

recoup/model
    Logistic propensity model, isotonic calibration (NumPy)

recoup/gateway
    Payment-provider adapter interface

recoup/simulation
    Synthetic world, evaluation arms, orchestrator, metrics
```

Each layer may import the layers above it and never the layers below it.

The ordering is enforced by a test. This architecture also isolates the payment-provider boundary from the recovery logic, allowing the core decision engine to remain independent of the gateway implementation.

See [docs/BUILD_LOG.md](docs/BUILD_LOG.md) for the development history and architectural decisions.
