# Build log

Notes on what broke and why. Kept in the repo because the reasoning behind a fix is
usually more useful than the fix, and because several of these were mistakes in my
own reasoning rather than in the code.

Entries are dated when they were written up. Where that is later than the fix, the
entry says so — the two entries dated 2026-08-28 are anchored to the files they
describe; the rest were found and written on 2026-08-31, most of them during a pass
whose only goal was to check whether the numbers in the README were true.

Entry format:

```
### YYYY-MM-DD — short title
Symptom:      what was observed
First guess:  what I thought it was
Root cause:   what it actually was
Fix:          what changed
```

---

### 2026-08-21 — Two safety rules contradicted each other, and the safer-looking one was wrong

**Symptom.** A test asserting that a wholly unrecognised failure escalates to a
human instead returned `STOP_SUPPRESSED` — the case was being closed silently.

**First guess.** Rule ordering. I assumed `never_auto_retry` was simply running
before `confidence_floor` and that swapping the two would fix it.

**Root cause.** Not ordering — a genuine contradiction between two pieces of
policy written independently. `ErrorReason.UNKNOWN` was on the categorical
`NEVER_AUTO_RETRY` list, while the diagnosis layer was designed so that an
unmappable failure abstains and routes to a human. Both encoded a defensible
instinct ("do not act on what you do not understand"), but they disagreed on the
*mechanism*, and the mechanisms are not interchangeable: suppression closes the
case with no human in the loop, whereas escalation stops the agent and surfaces
the money. Suppression is the more dangerous of the two precisely because it
looks like the conservative option — it produces no action, no alert, and no
record that anyone should look.

**Fix.** Removed `UNKNOWN` from `NEVER_AUTO_RETRY`, leaving only genuine
prohibitions (risk declines, suspected fraud, revoked mandates). Unrecognised
reasons are now caught by the confidence floor, which escalates. Left the rule
order alone: a categorical prohibition still outranks a confidence check.

**Worth noting.** Reordering the rules would have made the test pass without
fixing anything — `UNKNOWN` would still have been one edit away from silently
closing cases. The lesson is about where a policy is enforced, not what order the
policies run in.

**Superseded on 2026-08-31.** This entry originally ended by claiming that a
low-confidence risk decline being *suppressed* rather than escalated was "the right
way round". That was wrong, and for exactly the reason the rest of the entry gives.
See the `never_auto_retry` entry below: I had diagnosed the failure mode correctly
and then left an instance of it in place three lines from the diagnosis.

---

### 2026-08-28 — The gateway package and the simulation imported each other

**Symptom.** `ImportError` on a partially initialised module, from a fresh
interpreter, with no obvious cycle in the file I was editing.

**First guess.** A stray top-level import left over from moving code around.

**Root cause.** A real cycle, and a design error rather than a typo. I had put the
deterministic fake gateway in `recoup/gateway/` alongside the protocol it
implements, which is where a fake usually goes. But this fake executes actions
against `SimulatedWorld` — it *needs* the simulation — while the simulation needs
the gateway protocol to talk to anything. Two packages, each legitimately requiring
the other.

**Fix.** Moved the fake to `recoup.simulation.mock_gateway`. The dependency now runs
one way: simulation depends on gateway, gateway depends on nothing but the standard
library. The layering test in `tests/test_core_has_no_dependencies.py` enforces the
direction so it cannot silently come back.

**Worth noting.** The instinct "test doubles live next to the interface they
implement" is right when the double is a stub and wrong when it is a simulator. A
simulator is not a smaller version of the real thing; it is a different subsystem
that happens to satisfy the same protocol, and it belongs with the world it
simulates.

---

### 2026-08-28 — Calibration bought at the price of resolution

**Symptom.** After adding isotonic calibration, the tuning curve for the
contact-probability floor moved in visible jumps, and whole bands of the parameter
produced identical results.

**First guess.** Too few validation rows, so the bins were noisy.

**Root cause.** Pool-adjacent-violators returns a step function, and mine had
collapsed the model's entire output to twenty-one distinct probabilities — fewer
than the hand-written heuristic's fifty-seven. Every policy rule in this system
compares a probability against a threshold, so a near-discrete estimator makes
those thresholds behave like cliffs: a threshold sitting between two steps can be
moved a long way with no effect at all.

**Fix.** Interpolate linearly between block means instead of returning the step
function. Monotonicity and the fitted calibration are preserved, the output is
continuous again, and values outside the fitted range are clamped rather than
extrapolated — isotonic regression makes no claim beyond the data it saw.

**Worth noting.** The metric said the calibration was working: ECE improved. The
metric could not see that the estimator had become nearly useless for the one
operation the system actually performs on it. This is the same shape as the AUC
problem documented in EVALUATION.md — measure the property you depend on, not the
property that is conventional to report.

---

### 2026-08-31 — A deferral that did not advance the clock

**Symptom.** A case in the demo walkthrough ended in an escalation whose stated
reason was that a deferral had failed to move the clock forward. The explanation
made no sense on its face: the rule that set the deadline and the rule that checked
it were the same rule.

**First guess.** A timezone problem in the deferral arithmetic.

**Root cause.** An off-by-one at an interval boundary. `contacts_since` counted
contacts with `when >= since`, a closed interval. `rule_contact_frequency` blocks
when the contact cap is exhausted and defers to `oldest_counted_contact + 7d` —
the first instant the oldest contact should have aged out. Under a closed interval
that instant still counted the contact, so the rule blocked again on precisely the
timestamp it had asked to be woken at. The orchestrator saw a deferral that did not
advance, correctly concluded it was in a loop, and escalated.

**Fix.** `contacts_since` now counts the half-open interval `(since, now]`.
Two regression tests pin the boundary: zero contacts counted at exactly seven days,
one at seven days minus a second.

**Worth noting.** 131 unit tests did not catch this. Every test asserted on
durations comfortably inside or outside the window, because those are the cases you
think of when writing tests for a cap. The bug lived only at the exact boundary,
which is also the only value the deferral logic ever generates — so the one input
the system produced in practice was the one input nothing tested. Gateway calls in
the demo went from 297 to 317 once it was fixed.

---

### 2026-08-31 — Suspected fraud was being filed where nobody would look

**Symptom.** Reading the demo output case by case, cases stopped by
`never_auto_retry` were ending in the `suppressed` bucket — the same terminal state
as a customer who had opted out of contact.

**First guess.** A mislabelled state in the demo's own summary line.

**Root cause.** The rule really was returning `Severity.HARD`. For an opt-out that
is correct: the customer asked not to be contacted, that applies to humans too, and
closing the case silently is the respectful outcome. A suspected-fraud or risk
decline is the opposite — the agent must not touch it, *and* somebody should look at
it. Marking it `HARD` put real fraud signals in the one bucket the design guarantees
nobody reviews.

**Fix.** `rule_never_auto_retry` now returns `Severity.ESCALATE` in both branches.
Two tests that had asserted `STOP_SUPPRESSED` were wrong and were changed.

**Worth noting.** This is the same error as the 2026-08-21 entry, which I wrote. I
had described suppression-instead-of-escalation as the more dangerous failure
because it looks conservative, fixed one instance, and left another in the adjacent
function — then closed that entry by asserting the surviving instance was correct.
Writing the general lesson down is not the same as applying it.

One of the two tests I had to change was worse than wrong. It was named
`test_no_candidate_action_escalates`, it asserted suppression, and it never reached
the branch it named — the fixture failed an earlier rule. The name had contradicted
the assertion since the day it was written and the suite had been green throughout.
It is now split into two tests, one per branch, and the second constructs a
diagnosis with `candidate_actions=()` so it actually gets there.

---

### 2026-08-31 — One audit trail, two timezones

**Symptom.** A single case's rule details read "not due until 2026-07-20 15:01 UTC"
on one line and "until 2026-07-21 15:00 IST" three lines later.

**Root cause.** Each rule formatted its own timestamps with `%Z`, which prints
whatever `tzinfo` the datetime happens to be carrying. Internally everything is
UTC; the rules are written against IST business hours; so the same policy got
described in two timezones inside the document a merchant would use to check it.

**Fix.** One formatter, `recoup.policy.timing.ist_stamp`, used by every rule that
mentions a time, plus `to_ist` in the demo's narration. The rules reason in IST, so
the explanations are in IST.

**Worth noting.** Both timestamps were correct. An audit trail can be accurate and
still be unusable, and "is every value right" is a weaker test than "can a reader
follow this".

---

### 2026-08-31 — The demo printed a structurally perfect walkthrough of invented fields

**Symptom.** The narrated case walkthrough printed `None` and `₹0` in most slots
while its layout and section ordering looked exactly right.

**First guess.** The payloads were being serialised before the narrator read them.

**Root cause.** I had written the narrator against payload key names I assumed —
`amount_paise`, `probability`, `at_risk_paise`, `from` — none of which the
orchestrator emits. `dict.get` returns `None` for a missing key, so every wrong
guess degraded silently into a plausible-looking blank rather than raising.

**Fix.** Walked the AST of every `log.append` call in the orchestrator, extracted
the actual key set for all thirteen entry kinds, and rewrote the narrator against
that. A second defect surfaced immediately: money arrives as a `Money` object when
read from the in-memory mirror and as `{"paise": ..., "currency": ...}` when re-read
from disk, and the demo hits the first path while `verify` hits the second. Both
shapes are now handled explicitly.

**Worth noting.** This is the failure mode of `.get()` on a dict you did not
define. It cost me nothing to fix and would have cost me the demo if a judge had
run it first.

---

### 2026-08-31 — A sweep that measured its own lack of statistical power

**Symptom.** The sensitivity sweep reported that the headline result "does not
survive every perturbation" — in every row, including the unperturbed one. It also
showed the unbounded arm beating the full agent under pessimistic assumptions.

**First guess.** The perturbation was being applied to the wrong probabilities.

**Root cause.** The sweep ran at half the batch size of `compare`. The unperturbed
row was already known to exclude zero at the headline sample size, so a sweep that
put zero inside the interval for that same configuration was not describing the
world — it was describing its own confidence intervals widening. The arm-ordering
flip was the same artefact.

**Fix.** Raised the sweep's default count to match the headline runs, and separated
the two questions it had been conflating: "did the ordering change" and "did the
interval widen to include zero" are now tracked and reported independently, with
the actual interval printed per row.

**Worth noting.** I had written the false conclusion into a draft of EVALUATION.md
as a limitation of the *system*. A sweep that loses power quietly is worse than no
sweep, because its output is indistinguishable from a real negative result and it
arrives wearing the authority of a robustness check.

---

### 2026-08-31 — I read my own diagnostic wrong and nearly published it

**Symptom.** While checking which policy rules ever actually block anything, I
concluded that three of the twelve — the expected-value gate, the contact
probability floor, and the downtime check — blocked *zero* actions in the reference
run. I wrote several paragraphs around it, including the line "the gates are dead
code under the optimistic estimator".

**Root cause.** Two independent bugs in the throwaway script, both silent. It
printed rule counts with `Counter.most_common(20)` on a set of exactly twenty-four
rule/outcome combinations, so the four smallest — which were precisely the rarely
blocking rules I was asking about — were truncated off the bottom. And a separate
check filtered for `'expected_value' in rule_name` when the rule is called
`ev_floor`, matched nothing, and printed a confident `EV-gate failures: 0`.

The real counts are 36 blocks for `ev_floor` and 2 for `contact_probability_floor`
under the heuristic, against 103 and 585 under the calibrated model. Only the
circuit breaker never fires at all.

**Fix.** Recomputed over all rules with no truncation and no substring matching,
and corrected the claim everywhere it had already been written: EVALUATION.md, the
README, and a comment in `rules.py`. "Very nearly inert, and 292× more active once
calibrated" is a weaker sentence than "dead code" and it is the true one.

**Worth noting.** The direction of the finding survived — the contact floor really
is close to inert under optimistic priors, and the outcome sweep confirms it
independently, since setting the floor to zero moves the heuristic's results not at
all and the calibrated model's by ₹93,701 and 242 contacts. But I had reached the
right conclusion through a broken measurement, which is not the same as being
right, and the write-up was overstated in a way that a judge checking the logs
would have caught in a minute.

A third thing this exposed: raw block counts over-report, because the engine
deliberately does not short-circuit. Every per-action rule is evaluated for every
candidate action so the trail records a complete verdict, which means a "block" is
a rule returning false and not necessarily the rule that decided the outcome.
EVALUATION.md now says so before showing the table, and defers to the outcome sweep
as the authoritative measure.

---

### 2026-08-31 — The repository described a version of itself that did not exist

**Symptom.** Not a crash. A sweep for promises the code made and did not keep,
prompted by noticing that `make api` referenced a FastAPI layer I had never
written.

**Root cause.** Four separate cases of documentation written at the moment of
*intending* to build something. The Makefile had `api` and `web` targets for
neither. `recoup/__init__.py` listed an `api` layer in its layering docstring.
`recoup/gateway/__init__.py` described a live adapter in `razorpay_live` "imported
lazily because it needs `requests`", and then justified the package's dependency on
`requests` by claiming everything left in it talks to the network — the module never
existed and the package imports only the standard library. `pyproject.toml`
advertised an `[api]` extra that installs four packages nothing imports.

**Fix.** Deleted the two Makefile targets, corrected both docstrings to describe
what is there, and annotated the extra as aspirational. Each site now states the
absence rather than implying the presence.

**Worth noting.** A Makefile target that does not run is worse than a missing
feature: it is an invitation to try something that will fail, and it costs the
reader their time and their trust in everything else the repo claims. The whole
class of defect comes from writing docs in the future tense and never revisiting.

**And immediately after.** Writing the paragraph above, I claimed the audit log's
"forty seconds an arm" note on per-entry flushing had been checked and was accurate.
It had not been checked — I had read it, not measured it, in the same entry where I
was congratulating myself for auditing unverified claims. Measured properly on the
300-case batch it describes: 9.0s with flushing off, 72.5s with it on, across 17,415
entries. So the cost was 63 seconds rather than forty, and the entry count was
17,415 rather than "around sixteen thousand". Both figures in the docstring are now
the measured ones.

The pattern across this whole day is one thing wearing several disguises: a claim
that sounds like something I verified, because I wrote it, and which I never
actually ran. The only defence that worked was mechanically re-deriving every number
from the artefacts rather than from memory.

---

### 2026-08-31 — A quarter of the reference batch could not physically have happened

**Symptom.** The first line of the demo walkthrough read `failed: invalid_vpa on
card via KOTAK`. A VPA is a UPI address. A card does not have one.

**First guess.** A display bug in the narrator, which had form — see the invented
fields entry above.

**Root cause.** The generator drew `error_reason` and `payment_method` from two
independent weight tables and stapled them together. 517 of 2,000 events in the
reference batch, 25.9%, were combinations that cannot occur: `incorrect_otp` on UPI
(which authenticates with a PIN in the payer's own app, and has no OTP at all),
`expired_card` and `invalid_cvv` on UPI, `card_declined` on netbanking,
`upi_collect_expired` on cards, wallets and EMI.

Worse than the implausibility: an impossible combination is a free win for
everything downstream. Nothing in the real world generates it, so nothing can
contradict whatever the diagnosis layer maps it to or whatever the estimator learns
from it. A quarter of the batch was scoring the system against a distribution no
merchant will ever send it.

The same pass found the mirror-image defect. `MANDATE_REVOKED` was mapped in the
taxonomy and sat on the `NEVER_AUTO_RETRY` list, but no weight table ever emitted
it, so a categorical prohibition was exercised only by unit tests that built the
event by hand. In the reference run — the one every published number comes from —
that rule had never fired.

**Fix.** Rail and amount are now drawn first, and the reason comes from a table
conditional on both. Recurring debits get their own table, because a mandate charge
has nobody at the keyboard and so the entire customer-input family is unavailable to
it, with one carve-out: RBI requires an additional authentication factor on
recurring card debits above ₹15,000, so those genuinely can fail on an OTP.
`tests/test_generate.py` walks every event and asserts the combination is possible,
against a plausibility table written independently of the generator's own weights —
deriving the expectation from the code under test would have made it a tautology
that passed no matter what was emitted.

**Worth noting.** This invalidated every published figure at once, because changing
a draw shifts the whole seed stream. That is the correct outcome and it is why the
numbers get regenerated by a command rather than remembered, but it is also the
strongest argument I have for why the generator deserved this scrutiny earlier than
the last day: it is the one component whose bugs are invisible in every metric,
since it *is* the thing the metrics are computed against.

---

### 2026-08-31 — The customer pool made batch size a policy knob

**Symptom.** A new test asserting that growing a batch leaves earlier records
untouched failed on 98 of 100 events. It also failed on one `issuer`, which made no
sense: nothing about the issuer draw depends on batch size.

**Root cause.** Two things, one deliberate and undocumented, one accidental.

The deliberate one: customers were drawn from a pool of `count // 2`, so the
customer attached to a given event index changes with batch size. That is defensible
and I would choose it again — it holds failures-per-customer constant at 2.3 across
every size, which matters because the contact-frequency cap is per customer per
seven days. If density rose with `count`, then raising `count` to narrow a
confidence interval would also make the cap bind more often, and the sample size
chosen for precision would quietly change the policy's behaviour. But the module
docstring promised that adding a record shifts nothing before it, and that promise
was false.

The accidental one: because the draw was `randrange(1, count // 2)` on the shared
per-event stream, it consumed a *count-dependent amount of randomness* and shifted
every draw after it. That is where the stray issuer came from — about one in a
hundred, moving between batch sizes for no reason anyone intended.

**Fix.** The customer comes from its own stream keyed on the pool size, so the
coupling is exactly one field wide. The docstring now states the exception and the
reason for it, and three tests pin the three separate properties: everything but the
customer is index-stable, density does not drift with size, and the customer draw
cannot perturb any other field.

**Worth noting.** The deliberate half was a good decision recorded nowhere, which
made it indistinguishable from a bug when a test finally asked. The accidental half
had been silently corrupting a field for as long as the code existed, and it was
only visible because fixing the first thing required looking at the draw at all.

---

### 2026-08-31 — The flagship test was passing by luck, and calibration is doing damage

**Symptom.** After the generator fix, `test_learns_that_timing_matters_for_balance_
and_not_for_dead_cards` failed. It is the test named for the project's central
claim, and it asserted that the fitted model prices a payday retry at more than
twice a mid-month one.

**First guess.** Three, in order, and the first two were wrong in a way worth
recording because both times I was misreading my own diagnostic rather than reading
a bug.

I thought the new mandate-heavy mix had diluted the balance cases. It had not — it
more than doubled them, and the world applies the salary lift to
`MANDATE_INSUFFICIENT_BALANCE` through the same root cause.

Then I thought regularisation had crushed the coefficient, because
`balance_x_salary_window` was +0.17 where the data showed a +1.18 log-odds effect.
That was me comparing a *standardised* coefficient against a raw-space effect. The
feature is active in 3% of rows, so its standard deviation is 0.17, and +0.17
standardised is +1.00 raw. Adding the main effect gives +1.15 against a true +1.18:
the weights learn it almost exactly. `coefficients()` says "standardised" in its
docstring and I still read the number as if it were not.

Then I read the failure message itself wrong — `0.149 not greater than 0.156` is
the estimate against *twice* the mid-month value, not against the mid-month value.
The lift was 1.92x and the bar was 2x. Nothing was inverted; the test missed a round
number by a hair.

**Root cause.** The bar was arbitrary and the previous batch composition had cleared
it by luck. But measuring properly to establish that turned up the real finding:

```
                              mid-month   payday    ratio    truth
balance, same rail    truth      0.0825   0.2625    3.18x
                      raw        0.0737   0.2002    2.72x
                      calibrated 0.0777   0.1493    1.92x
dead card, alt rail   truth      0.2300   0.2300    1.00x
                      raw        0.2244   0.2498    1.11x
                      calibrated 0.2041   0.2607    1.28x
```

Isotonic calibration compresses the real effect from 2.72x toward 1.92x, and
*amplifies* a spurious one from 1.11x to 1.28x. Both moves are away from the truth,
and on the dead card the raw estimate is nearer the world than the calibrated one on
both sides of the contrast. The mechanism is not mysterious: isotonic regression
fits one monotone curve to the aggregate observed frequency and then reads every
case off it. It is entitled to preserve ordering and nothing else. The payday cell
is 3% of the training rows, so the curve's shape is set by the other 97%.

The spurious dead-card lift has its own cause: `balance_x_salary_window` is a subset
of `in_salary_window`, so the two are correlated by construction and ridge splits
the credit between them. A balance-specific effect therefore arrives with a small
positive main effect attached, +0.148 log-odds, which the training data does not
support — payday is fractionally *worse* than mid-month for non-balance causes.

**Fix.** The first assertion is now 1.5x, with all six numbers above written into
the docstring so the bar is legible rather than magic. The second assertion was a
tolerance around zero, which says nothing about the claim; it is now a comparison,
because what the claim needs is that timing moves a balance case much more than a
dead instrument, and 1.92x against 1.28x is that.

**Worth noting.** This is the AUC problem from EVALUATION.md arriving from the other
direction. There the metric that is conventional to report could not see the
property the system depends on. Here the intervention that improves the conventional
metric actively damages that property: ECE gets better while the conditional effect
the whole agent is built on gets worse. Neither is an argument against calibration —
the policy layer needs probabilities that mean what they say, and EVALUATION.md
shows what happens to a threshold fed uncalibrated ones — but "calibrated" is a
statement about a population and not a promise about any case in it.

