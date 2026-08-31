# Evaluation

This document exists to make the results in [README.md](README.md) falsifiable.
It is organised around the ways they could be wrong, roughly in order of how much
they would cost me if a judge found them first.

Everything here regenerates from four commands: `run.py compare --learned`,
`run.py model --reliability`, `run.py sensitivity`, `run.py verify --tamper`.

## The caveat that matters most

**I wrote both the policy and the world that scores it.**

`recoup.simulation.world.GroundTruth` holds the probabilities the outcomes are
drawn from: how often an insufficient-funds retry succeeds on payday versus
mid-month, how often a customer clicks a payment link, how long an issuer outage
lasts. `recoup.model.estimator.HeuristicEstimator` holds the priors the policy
uses to decide. Both are my estimates, written days apart, and they agree with
each other far better than either would agree with a real merchant's data.

The clearest symptom is in the calibration table below: the heuristic priors score
AUC 0.715 against an oracle that also scores 0.715 — the hand-written estimator
ranks actions exactly as well as the true probabilities do. That does not happen
with real data. It happens because the priors and the ground truth were written by
the same person reasoning the same way, so the heuristic is not estimating the
world so much as remembering it.

This does not invalidate the arm comparison, because every arm is scored in the
same world and the differences between them come from capabilities rather than
from probabilities. It does mean that any *absolute* number here — 40.1% gross
recovery, ₹1.9M incremental — is a property of my simulation and should not be
quoted as a property of payment recovery. The ordering is the claim. The
magnitudes are illustration.

The honest version of this project would run against a merchant's historical
failure data. I did not have any.

## Method

**Incremental, not gross.** Each arm holds back 20% of its cases and never acts on
them. Recovery is `treated_rate − holdout_rate`, applied to the value at risk.
Every arm gets its own holdout drawn the same way, so the comparison is between
four treatment effects rather than against a shared baseline.

This is the design decision that most changes the answer. The `do_nothing` arm
recovers 15.9% of the value at risk with no intervention at all — customers who
retry themselves, or who were going to pay on the second attempt regardless. A
recovery system reporting gross recovery claims all of that as its own. The fixed
ladder's gross rate of 28.7% collapses to an incremental effect of ₹708,965 with a
95% interval of −482,286 to 1,860,466: **not distinguishable from zero.** That arm
is the one most production systems actually run.

**Shadow decisions, not assumptions.** A holdout case runs the identical
`decide()` and the identical rule set; the orchestrator logs the result as a
`shadow_decision` rather than executing it. So the holdout is not an untouched
population we assume the agent would have handled a certain way — it is a
population where the decision is on record and the action was withheld.

Both halves of that are checkable in `data/audit_recoup.jsonl`, and here is the
check. The run assigns 391 of 2,000 cases to control. **No control case carries an
`action_executed` entry** — the withholding is total. 387 of the 391 carry a
`shadow_decision`; the remaining four self-recovered, with `attempts_used: 0`,
before the agent reached a decision point at all, so there was no decision to
shadow. That is the honest version of the claim: every holdout case the agent
formed a view about has that view on record, and four of them it never got to.

```
python run.py verify   # then grep the log yourself; the numbers above are from it
```

**Bootstrap intervals, 2,000 resamples, fixed seed.** Payment amounts are heavy
tailed; a handful of large recoveries dominate the total, so a normal
approximation would understate the spread. Percentile intervals on the difference
of means, resampling treatment and control independently.

**The world's seed is held fixed across arms.** A case that would have
self-recovered on day three does so in every arm. The counterfactual is held
constant rather than resampled, which removes a large source of between-arm
variance that has nothing to do with the policies being compared.

## Arm comparison

Two thousand failures, 45-day horizon, seed 20260821.

| Arm | Gross | Holdout | Treated | ₹ incremental | 95% interval | Attempts | Contacts | Wasted contacts | ₹ spent | Cost per ₹ |
|---|---|---|---|---|---|---|---|---|---|---|
| Do nothing | 15.9% | 15.9% | 15.9% | 0 | — | 0 | 0 | 0 | 0 | — |
| Fixed ladder | 28.7% | 21.9% | 30.4% | 708,965 | −482,286 to 1,860,466 | 4,091 | 0 | 0 | 8,182 | ₹0.012 |
| No policy layer | 33.9% | 21.9% | 37.0% | 1,257,038 | 8,938 to 2,444,853 | 3,750 | 1,016 | 840 | 5,947 | ₹0.005 |
| Recoup | 40.1% | 21.9% | 44.9% | 1,902,805 | 694,191 to 3,122,269 | 3,004 | 889 | 713 | 4,754 | ₹0.002 |
| Recoup (fitted model) | 39.2% | 21.9% | 43.7% | 1,808,783 | 603,869 to 3,007,577 | 2,741 | 624 | 477 | 4,692 | ₹0.003 |

### On the third arm's name

The comparison a judge probably wants is "against an LLM agent". This arm is
deliberately not that, and it is not called that, because no language model runs
in it. What it removes is the policy layer: it diagnoses correctly, estimates a
probability, and then acts on that estimate with no expected-value gate, no quiet
hours, no contact cap, no cause-aware timing and no categorical prohibitions.

That is the failure mode the comparison is actually about — an agent with good
intentions and no bounds — and removing the bounds demonstrates it directly,
without a network call and without my reporting a result I did not produce. The
honest label is `no_policy`.

### What the policy layer buys

Recoup against the no-policy arm: **more money from strictly fewer actions.**
₹1,902,805 against ₹1,257,038, on 3,004 attempts against 3,750 and 713 wasted
customer contacts against 840.

I expected the opposite shape of result — that the bounds would cost some recovery
and buy safety, and that the write-up would be an argument about whether the trade
was worth it. It went the other way because the unbounded arm spends its three
attempts badly: retrying a dead card, retrying an empty account before payday, and
messaging customers at 02:00 who then ignore the link. The attempt cap is shared
between the arms, so an attempt wasted on a hopeless action is an attempt
unavailable for a good one. Restraint is not competing with recovery here; it is
how the recovery budget gets spent on the cases that can move.

Escalations are 68 with the policy layer and 70 without, which is noise — both
arms share the diagnosis layer and nearly every escalation comes from a
categorical prohibition or an unmapped error code rather than from a policy
decision.

### The fitted model does not win, and what it actually does

The learned arm recovers ₹1,808,783 against the heuristic's ₹1,902,805. The
intervals overlap almost completely, so the ₹94k gap is not a real difference. But
it reaches **624 customers instead of 889** — a 30% reduction — and wastes 477
contacts instead of 713.

The obvious reading is "calibration buys restraint". That reading is wrong, and
the sweep below shows why. **Set the contact floor to 0% and the two estimators
become indistinguishable: ₹1,902,805 on 889 contacts against ₹1,902,484 on 866.**
A difference of ₹321 on ₹1.9M.

So calibration does not, by itself, change what the agent recovers or how many
people it contacts. What it does is make an existing safeguard start working. The
5% contact floor is **very nearly inert under the heuristic** — 2 blocks in 7,684
evaluations, and removing it entirely changes neither the money nor the contact
count. Those priors are optimistic in every reliability band and almost never
predict below 0.05. Calibrated probabilities fall where they honestly belong, a few
hundred land under 5%, and only then does the floor bite.

That is a more interesting result than the one I expected to report, and it
generalises past this repository: a policy layer full of thresholds can look
thorough, pass its unit tests, and be barely doing anything, because whether a
threshold fires depends on the calibration of the number fed to it. The rule was
never the safeguard on its own. The rule plus an honest probability was.

Whether the resulting trade is *good* depends on how a merchant prices customer
attention, and the price here is `PolicyConfig.annoyance_cost = ₹8.00`, which I
made up. I am reporting the result rather than tuning that number until the model
wins, which was available and would have been dishonest.

### Which rules actually bind

Blocks, and evaluations, counted off the `rule_evaluated` entries in both logs.

| Rule | Heuristic blocks | of | Fitted blocks | of |
|---|---|---|---|---|
| retry_spacing | 3,163 | 7,684 | 3,160 | 7,679 |
| balance_cooloff | 1,572 | 7,684 | 1,572 | 7,679 |
| quiet_hours | 991 | 7,684 | 991 | 7,679 |
| attempt_cap | 620 | 6,446 | 373 | 6,191 |
| contact_frequency | 228 | 7,684 | 224 | 7,679 |
| customer_opt_out | 102 | 6,637 | 102 | 6,382 |
| never_auto_retry | 49 | 6,535 | 49 | 6,280 |
| confidence_floor | 40 | 6,486 | 40 | 6,231 |
| ev_floor | 36 | 7,684 | 103 | 7,679 |
| issuer_downtime | 2 | 7,684 | 2 | 7,679 |
| **contact_probability_floor** | **2** | 7,684 | **585** | 7,679 |
| circuit_breaker | 0 | 6,637 | 0 | 6,382 |

**Read this table carefully, because I first read it wrong.** Every per-action rule
is evaluated for every candidate action, deliberately — the engine does not
short-circuit, so the trail records a complete verdict rather than stopping at the
first block. A "block" is therefore a rule returning false, not necessarily the
rule that decided the outcome; an action already dead for another reason still gets
a full set of verdicts. Block counts are a diagnostic, and the sweep in the next
section is the authoritative measure of whether a rule changes anything.

With that caveat, the row that matters is `contact_probability_floor`: **2 blocks
under the heuristic against 585 under the fitted model**, on essentially identical
evaluation counts. The 5% floor is very nearly inert when fed optimistic priors,
because those priors are optimistic in every reliability band and almost never
predict below 0.05. Calibrated probabilities land where they honestly belong, a few
hundred fall under the floor, and the rule starts doing the job it was written for.
`ev_floor` moves the same way, 36 to 103.

`attempt_cap` moves in the opposite direction — 620 blocks to 373 — because fewer
attempts are wasted early, so fewer cases reach the cap at all. The cap is doing
less work because the pricing upstream got better, which is the direction you want
that number to move.

The circuit breaker is the honest zero. It halts a batch when the failure rate
across executed actions exceeds tolerance, nothing in these runs came close, and it
has never fired outside a unit test.

## Calibration

Six thousand failures explored into 22,928 training rows and 7,524 validation
rows from **disjoint batches** — not a random split of one batch, because two
attempts on the same case are correlated and a random split would leak.

| Estimator | AUC | Brier | ECE |
|---|---|---|---|
| oracle (true probabilities) | 0.715 | 0.1031 | 0.0096 |
| logistic, uncalibrated | 0.709 | 0.1033 | 0.0115 |
| logistic + isotonic | 0.709 | 0.1035 | **0.0082** |
| heuristic priors | 0.715 | 0.1038 | 0.0261 |
| flat 0.30 | 0.500 | 0.1412 | 0.1720 |

**The oracle row is the ceiling.** Those are the probabilities the outcomes were
drawn from, so no estimator can beat Brier 0.1031. Reporting a ceiling turns
"our model scores 0.1035" from a number with no scale into a distance from the
best achievable — 0.4% off optimal. Without it I would have no way to tell a good
model from an easy problem.

The oracle's ECE is 0.0096 rather than 0.0000 purely because of finite-sample
noise in the bins, which is why the isotonic model's 0.0082 "beating" it means
nothing. The comparison that means something is isotonic 0.0082 against the
heuristic's 0.0261 — **3.2× better calibrated.**

**Why calibration and not AUC.** The heuristic's AUC matches the oracle's exactly.
On ranking, the hand-written priors are already as good as the true probabilities.
A ranking-only evaluation would have concluded the model was worthless.

But nothing in this system consumes a ranking. The expected-value gate computes
`p × amount × margin − cost` and compares it to a rupee floor; the contact floor
compares `p` to 0.05 directly. Both are thresholds on an absolute probability, and
a uniformly optimistic estimator clears them on cases it should have declined no
matter how well it ranks.

This is not a theoretical worry here — it is measured. The contact floor blocks 2
actions under the heuristic and 585 under the calibrated model; the expected-value
gate, 36 against 103. Same ranking, same rules, same data. The only thing that
changed was whether the probabilities were honest about their own magnitude, and
that took a safeguard from very nearly inert to actively protecting several hundred
customers. Discrimination and calibration are different properties, and every
threshold downstream of `decide()` depends on the second one.

The `flat 0.30` row makes the failure concrete from the other end: AUC 0.500, no
information at all, and it predicts 0.300 on 7,524 rows where the observed rate is
0.128. Every threshold in the policy layer would wave it through.

### Reliability, isotonic model

| Predicted band | Mean predicted | Observed | n |
|---|---|---|---|
| 0.00–0.10 | 0.061 | 0.063 | 3,642 |
| 0.10–0.20 | 0.137 | 0.128 | 2,396 |
| 0.20–0.30 | 0.238 | 0.252 | 1,158 |
| 0.30–0.40 | 0.362 | 0.405 | 328 |

Against the heuristic, which is optimistic in every band — the pattern the
expected-value gate is least able to survive:

| Predicted band | Mean predicted | Observed | n |
|---|---|---|---|
| 0.00–0.10 | 0.068 | 0.051 | 2,620 |
| 0.10–0.20 | 0.142 | 0.108 | 2,973 |
| 0.20–0.30 | 0.243 | 0.217 | 1,215 |
| 0.30–0.40 | 0.329 | 0.306 | 510 |
| 0.40–0.50 | 0.470 | 0.427 | 206 |

The isotonic model has only four bands because calibration compresses the tail;
there are no longer enough predictions above 0.40 to fill a bin. That is a
genuine loss of resolution, and it is why the isotonic fit interpolates linearly
between block means instead of returning the step function that
pool-adjacent-violators produces directly — the raw step function collapsed the
whole batch onto twenty-one distinct probabilities, fewer than the hand-written
heuristic's fifty-seven. For an estimator whose entire job is to be compared
against thresholds, that is calibration bought at the price of resolution.

### Did it learn the thesis

The whole argument of this project is that *when* you retry matters, and that it
matters differently depending on *why* the payment failed. Two interaction terms
were added by hand to test whether the model would find that. Standardised
coefficients, largest first:

| Feature | Weight |
|---|---|
| attempts_used | −0.470 |
| cause=insufficient_balance | −0.406 |
| cause=gateway_routing | +0.255 |
| **balance_x_salary_window** | **+0.242** |
| cause=auth_friction | +0.241 |
| cause=issuer_outage | +0.124 |
| action=retry_alternate_rail | +0.123 |
| downtime_active | −0.106 |

`balance_x_salary_window` is the fourth-largest weight in the model. The direct
behavioural consequence, on one identical ₹2,500 insufficient-funds case moved
only in time: the model prices a same-rail retry at **0.087 on 21 July and 0.284
on 1 August — 3.26× from timing alone.**

The other half of the claim is that this should *not* happen to a dead card, which
does not un-expire on payday. Same shift, expired-card case, alternate-rail retry:
0.227 to 0.212 — a move of 0.015, or 6.6% of the base, and in the opposite
direction. Timing is priced where timing is the constraint and nowhere else.

That is the thesis, learned rather than asserted, and
`test_learns_that_timing_matters_for_balance_and_not_for_dead_cards` pins both
halves so a refactor cannot quietly lose it.

The second interaction term, dead-instrument × rail-switch, does **not** appear
anywhere near the top. That is not a failure of the model — the diagnosis layer
removes same-rail retry from the candidate set for `instrument_invalid` entirely,
so the model is never shown a dead card paired with a same-rail retry and there is
no variance for the interaction to explain. The preference is enforced structurally
rather than learned statistically, which is the stronger guarantee, and it is what
the test asserts instead. I found this out by writing the test I expected to pass
and getting a `KeyError`.

## What the contact floor costs

`min_contact_probability` is the one number in `PolicyConfig` that overrides the
economics rather than expressing them, so it should be the one most exposed. A
merchant who values their customers' attention above my ₹8.00 can read off what a
higher floor would cost them. `python3 sweep_contact_floor.py` regenerates both
tables on the reference batch.

Heuristic estimator:

| Floor | ₹ incremental | Contacts | ₹ given up vs 0% | Contacts avoided | ₹ per contact avoided |
|---|---|---|---|---|---|
| 0% | 1,902,805 | 889 | 0 | 0 | — |
| **5%** | **1,902,805** | **889** | **0** | **0** | **—** |
| 10% | 1,539,143 | 305 | 363,662 | 584 | 623 |
| 15% | 1,539,143 | 305 | 363,662 | 584 | 623 |
| 20% | 1,433,430 | 94 | 469,375 | 795 | 590 |

Calibrated logistic estimator:

| Floor | ₹ incremental | Contacts | ₹ given up vs 0% | Contacts avoided | ₹ per contact avoided |
|---|---|---|---|---|---|
| 0% | 1,902,484 | 866 | 0 | 0 | — |
| **5%** | **1,808,783** | **624** | **93,701** | **242** | **387** |
| 10% | 1,539,143 | 305 | 363,341 | 561 | 648 |
| 15% | 1,433,430 | 94 | 469,054 | 772 | 608 |
| 20% | 1,433,430 | 94 | 469,054 | 772 | 608 |

Three things to read off these.

**The configured 5% is free under the heuristic and not under the fitted model.**
Identical rows at 0% and 5% in the first table; ₹93,701 and 242 contacts apart in
the second. Same rule, same threshold, opposite significance.

**The plateaus are real, not rounding.** 10% and 15% are identical under the
heuristic because no heuristic prior for a contact action lands between them —
the estimator's output is coarse, which is the resolution problem isotonic
calibration was chosen to avoid. A threshold sitting inside one of those gaps can
be moved 50% in relative terms with literally no effect, which is worth knowing
before tuning it.

**The floor is expensive in its own stated terms, and that includes the value I
chose.** Every step costs between ₹387 and ₹648 of recovery per contact avoided.
`annoyance_cost` says a contact is worth ₹8.00. I originally used this argument to
reject a 10% floor as overriding the economics by roughly eighty times while
calling itself a mild safeguard — but the 5% floor I settled on does the same
thing by a factor of about forty-eight, once calibration makes it bite at all.

I have not resolved that, and I would rather state it than quietly pick the
version that flatters the result. The consistent positions are: keep the floor and
concede that ₹8.00 is far too low a price for customer attention, or trust the
₹8.00 and drop the floor toward zero — which recovers ₹93,701 more and contacts 242
more people. The floor stays at 5% because I believe the second option is wrong and
the real error is in the ₹8.00, but that is a judgement about customer experience I
cannot support with anything in this repository, and a merchant with churn data
could settle it in an afternoon.

## Sensitivity

Every probability in `GroundTruth` scaled by a common factor, so self-recovery,
retry success and customer responsiveness all move together. Two thousand
failures per cell.

| World | Fixed ladder | No policy layer | Recoup | Recoup 95% interval |
|---|---|---|---|---|
| pessimistic (×0.6) | 353,964 | 856,746 | 1,259,677 | 70,471 to 2,449,850 |
| mild (×0.8) | 362,127 | 983,767 | 1,577,499 | 406,943 to 2,784,524 |
| as written | 708,965 | 1,257,038 | 1,902,805 | 694,190 to 3,122,268 |
| optimistic (×1.25) | 884,096 | 1,289,004 | 2,054,922 | 786,054 to 3,374,258 |

The ordering holds in all four worlds and the interval excludes zero in all four.

**What this does and does not establish.** It shows the conclusion survives my
probability estimates being wrong by a uniform factor in either direction. It does
not show the conclusion survives their being wrong in *shape* — if payday timing
does not actually matter for balance failures, scaling every probability by 0.6
will not reveal that, because the ratio between the payday and mid-month
probabilities is exactly what a common scale factor preserves. A shape-perturbing
sweep is the obvious next piece of work and it is not here.

This sweep also cost me two false findings before it was correct, which is
recorded in [docs/BUILD_LOG.md](docs/BUILD_LOG.md): run at half the batch size, it
reported that the interval covered zero in every world including the unperturbed
one, and that the unbounded arm beat Recoup under pessimistic assumptions. Both
were artefacts of the smaller sample. I nearly wrote both into this document as
limitations of the system.

## Audit trail

A 2,000-case run of the full agent emits 114,202 log entries. Each carries the
SHA-256 of its predecessor over canonically serialised JSON, so any edit,
deletion or reordering invalidates every hash after it and `verify_chain` names
the index where the break occurred.

```
python run.py verify            # re-read every log in data/ and check the chain
python run.py verify --tamper   # then edit an entry and confirm it is caught
```

Two properties worth naming. Verification re-reads from disk rather than trusting
the process that wrote the log, which is the only version of the check that means
anything. And timestamps are passed into the log, never read from the clock inside
it — a log whose contents depend on wall-clock time cannot be reproduced, and every
number in this document depends on a fixed-seed batch reproducing exactly.

What the trail is for is answering "why did you charge my customer". A single case
in `run.py demo` shows the shape: the failure and its `error_reason`, the root
cause and confidence, each rule that was evaluated *including the ones that
passed*, the expected-value computation with the actual rupee figures, the action
chosen and its idempotency key, the gateway result, and the state transition. Rules
that pass are logged because they are what demonstrates the check happened rather
than merely that nothing blocked.

## Known limits

**Synthetic data throughout, authored by me.** Covered at the top; it is the
limitation everything else is downstream of.

**No live gateway.** `recoup.gateway.base` is an interface with one in-process
implementation. Idempotency, downtime handling and the error taxonomy are modelled
on Razorpay's documented behaviour and verified against nothing.

**Mandate handling is thinner than payment handling.** `reschedule_mandate` exists
as an action and `MANDATE_INVALID` as a root cause, but the subscription lifecycle
— charge cycles, pause and resume, re-authorisation flows — is not modelled at
the depth the one-time payment path is. The Subscriptions API also needs account
activation I do not have.

**The 45-day horizon truncates slow recoveries.** Cases still in
`awaiting_window` at the horizon count as not recovered. Since the salary-window
rule can defer a balance failure to the 1st of the following month, this
systematically undercounts exactly the recoveries the agent's central mechanism
produces. The bias runs against the result being reported, which is the direction
I would rather have it, but it is a bias.

**Costs are estimates, and one of them is doing real work.** ₹2.00 per retry,
₹0.25 per contact, ₹8.00 of customer annoyance per message, an 85% margin on
recovered revenue. The first two are roughly right for Indian payment rails and the
fourth is a plausible net-of-fees figure. The third is invented, standing in for a
quantity nobody measures, and the contact-floor sweep above shows it is the number
the customer-experience half of this system turns on.
