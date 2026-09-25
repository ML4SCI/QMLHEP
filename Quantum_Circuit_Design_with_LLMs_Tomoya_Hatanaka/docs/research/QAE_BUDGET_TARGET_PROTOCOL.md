# QAE minimum-budget-to-target study — pre-registered protocol

Frozen **before** any B = 6 or B = 10 candidate was generated, trained, or
inspected. Date: 2026-09-08 (JST).

Basis: the final slide of the 2026-09-04 meeting deck
(`20260904_llm-vqc`, presentation id `14aqubJaVVc4sxlXlY4GxwVti5ULAlQnhl82znxxQ_K8`),
which proposes *"search for the smallest budget that reaches a target
validation fidelity"*.

Retrospective input: `outputs/qae_budget_targets_20260908/REPORT.md`
(analysis branch `analysis/qae-budget-target-20260908`), a validation-only
re-analysis of `outputs/qae_robustness/` at source commit
`e846c27fc21d06aa20e026590fcc06429f8bcfb4`. That re-analysis made **no**
model calls; it is the reason the two cells below were selected, and it is
not itself an experiment.

## 1. Endpoint and decision rule (fixed, not revisable)

* **Metric**: *validation* trash fidelity
  `F_val = <0...0|rho_trash|0...0>` on the 12 held-out validation ground
  states. **Not** reconstruction fidelity. **Not** test fidelity.
* **Best-so-far**: for seed `s` and candidate index `k`,
  `F_best(s, k) = max_{i <= k} F_val(s, i)`, over the candidates of the run
  that was *configured* at budget `B`.
* **Targets**: `F_target in {0.95, 0.99}`. Both are reported. Neither may
  be changed after seeing a result.
* **Seeds**: the same 12 paired seeds `s = 0..11` as every previous QAE
  study. All 12 are reported; no seed is dropped, re-rolled, or replaced.
* **Pass rule**: a (condition, method, target) cell **passes** iff
  `#{s : F_best(s, B) >= F_target} >= 10` out of 12. Equality counts as a
  hit. A mean/median above the target is **not** a pass and may not be
  substituted for the count.
* A missing, crashed, or budget-stopped seed is reported as **missing**,
  never silently converted into a failing seed.

## 2. Budget definition and admissible grid

* **Budget `B` = the number of candidate architectures trained and
  evaluated per seed.** It is not a token count, not a call count, and not
  a wall-clock quantity.
* Admissible grid, frozen here: `B in {4, 6, 8, 10, 16}`. `B` must stay
  **even** so that the pre-registered 1:1 exploration/refinement split of
  the adaptive methods (`n_warm = B/2`) is preserved. No odd budget and no
  budget outside this grid may be added after seeing a result.
* Each configured `B` is a **separate search policy**: the initial
  exploration count, the number of proposals requested per call and the
  refinement horizon all change with `B`. Therefore:
  * a prefix of a `B = 16` run is **not** a `B = 3` experiment;
  * runs at different `B` are **never** concatenated;
  * no monotonicity in `B` is assumed, so **no bisection** and no claim of
    an exact minimum or an exact bracketing interval.
* Reporting vocabulary, fixed: **smallest verified passing budget**,
  **verified failing budgets**, **unverified budgets**. Any statement
  stronger than these three requires an explicit stated assumption.

## 3. Experiments to run (priority order) and stopping rule

**E-A (first priority). TFIM anchor, `B = 6`.**
4 qubits, open-chain TFIM, reference model snapshot
`gpt-5.4-mini-2026-03-17`, seeds 0–11, all four methods
(Random, Greedy, LLM-Open, LLM-Closed) at the same `B = 6` so the controls
are budget-matched. LLM-Closed split 3 + 3. LLM-Open draws one fresh
6-candidate pool (it is **not** a prefix of the `B = 8` pool).
Anchor for the one-factor manifest check: the study reference condition
(4 qubits / TFIM / `B = 8` / reference model). Only `budget` differs.

**E-B (second priority). XXZ anchor, `B = 10`, LLM-Closed only.**
4 qubits, XXZ chain, reference model snapshot, seeds 0–11, split 5 + 5.
Motivation: the XXZ `B = 8` LLM-Closed cell is 9/12 at 0.95, one seed short
of the rule. This is a **boundary probe anchored at XXZ**, declared as its
own protocol — *not* a one-factor change from the TFIM reference. The
existing one-factor verifier is **not** disabled or weakened; a second,
explicitly declared anchor (`hamiltonian_xxz`) is used for this cell only,
and the code records which anchor each condition was verified against.
Because only LLM-Closed is run here, **no same-budget cross-method
comparison at XXZ `B = 10` may be reported.**

**Not run (left unresolved on purpose).** 6-qubit, 8-qubit and
alternative-model boundaries, and the 0.99 target. The retrospective audit
shows every method at 0/12–2/12 there; a blind paid sweep would not
separate proposal validity, insufficient search, insufficient training and
insufficient circuit capacity. These stay open, with diagnosis prioritised
over new spend.

**Stopping rule.** Run E-A, then E-B, then stop. No further budget is
added after seeing either result. If the cost cap is reached mid-run, the
run stops at the cap and the incomplete cells are reported as *blocked*,
not as failures.

## 4. Inheritance from the anchors (no silent substitution)

Every element below is inherited **unchanged** from the corresponding
anchor condition and is fingerprinted in each run's `manifest.json`:

model snapshot (`gpt-5.4-mini-2026-03-17`; never silently substituted);
system prompt, prompt template, wording and instruction set; JSON output
schema and validator; temperature 0.7; 2 JSON-repair attempts, then
bounded capacity repair, then flagged random fill; circuit capacity
(`3n` trainable rotations + `n` CNOTs in a `4n`-gate ordered sequence),
gate set `{RX, RY, RZ, CNOT}`, arbitrary ordering and arbitrary valid
CNOT control/target; latent = first `n/2` qubits, trash = last `n/2`;
trainer (Adam, lr 0.05, 60 epochs, `uniform(-0.05, 0.05)` init,
complex128, best-validation checkpoint, loss `1 - mean trash fidelity`);
the per-architecture training-seed rule; the data split (32 train /
12 validation / 64 test parameters, `default_rng(10_000 + seed)` streams);
the RNG stream offsets of every non-API arm; the selection rule (lowest
validation loss).

Only these change, and only as `B` forces them: `n_warm = B/2`,
`n_refine = B/2`, the number of candidates requested per batch call, and
the deterministic output-token limit. Each run's manifest records the
anchor key, the changed factor(s), and the derived quantities recomputed
from the factors alone.

## 5. Test-set protection

* Selection of budgets, architectures and analysis choices uses
  **validation only**.
* The new runner **does not evaluate any candidate on the test set.** It
  passes an empty test array to the frozen trainer, so no test state is
  ever contracted with a trained circuit, and it writes candidate tables
  with a validation-only column allowlist.
* Historical test outputs already exist in `outputs/qae_robustness/`.
  They are **not** a fresh confirmation set and are not reported as one.
* No confirmatory test evaluation is performed in this study. If one is
  ever run, it must be pre-registered separately, executed after all
  selections are frozen, and never fed back into search.

## 6. Cost control

* Paid calls require `LLM_API_BUDGET_USD` to be set to an explicit
  nonzero cap by the user. The code refuses to call otherwise. This study
  runs under a user-authorised cap of **USD 2.00**, which the agent may
  not raise.
* Spend is checked **before** each request and accumulates across both
  experiments, all resumptions and any parallel execution.
* Recorded separately per condition: evaluated candidates, generated
  proposals, API calls, repair calls, input/output tokens, estimated cost,
  wall-clock time.
* Caching/resume key: results are reused **only** when the configured `B`,
  model snapshot, prompt, seed, initialisation and data hash all match
  exactly (the per-condition directory is keyed by the condition, the pool
  by the condition, and each LLM-Closed store by `(condition, seed)`).
* `invalid`, `repaired` and `fallback` are recorded distinctly. Random
  fallbacks **consume candidate budget and stay in the primary analysis.**
  Fallback-excluding numbers are auxiliary, non-causal diagnostics only.

## 7. Deliverables

A new `outputs/qae_budget_targets_v2_20260908/` directory containing this
protocol's manifest, hashed raw logs, the validation curves and attainment
counts at both targets for every condition, validity/cost tables, an
English report, a Japanese summary, and the exact reproduction commands.
