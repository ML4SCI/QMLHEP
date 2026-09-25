# Minimum budget to a target validation fidelity

Date: 2026-09-08 (JST). Protocol: `docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`,
frozen and committed before any candidate of the two new cells was generated.

Status: **executed.** Two pre-registered boundary cells were run against the
live API; seven earlier conditions were re-analysed from committed logs with
no model calls. Several boundaries were deliberately left unresolved and are
listed as such.

## 1. What this study answers

The 2026-09-04 meeting closed on an action item rather than a result: stop
ranking methods at a fixed budget, fix a target validation fidelity, and find
the smallest budget that reaches it in at least 10 of the 12 paired seeds,
reusing existing logs before making any new call.

The endpoint is **validation trash fidelity**
`F_val = <0...0|rho_trash|0...0>` on the 12 held-out validation ground
states. It is not reconstruction fidelity and it is not test fidelity. For
seed `s` and candidate index `k`, `F_best(s, k) = max_{i <= k} F_val(s, i)`
within a run *configured* at budget `B`. A cell passes a target iff at least
10 of the same 12 paired seeds (0-11) finish at or above it. Equality counts
as a hit. A mean above the target is not a pass.

Budget `B` is the number of candidate circuits trained and evaluated per
seed. It is not a token count, a call count or a wall-clock quantity.

### One deliberate departure from the closing slide

The closing slide asked for an *interval* for the minimum budget where
finding the exact value would cost too much. This study does not report an
interval. Each configured budget is a different search policy -- the number
of exploration candidates, the number of proposals requested per call and the
refinement horizon all change with `B` -- so failure at one budget and
success at a larger one does not bracket a minimum without an unproven
monotonicity assumption. Three categories are reported instead: the
**smallest verified passing budget**, the **verified failing budgets**, and
the **unverified budgets**.

## 2. Reproduction of the earlier analysis (no model calls)

The validation-only re-analysis of `outputs/qae_robustness/` at source commit
`e846c27fc21d06aa20e026590fcc06429f8bcfb4` was re-executed and reproduces the
previously reported audit **byte-for-byte** (verified by `diff -r` of the
whole output tree against a run of the unmodified script; only the script's
own self-copy differs, because the script gained an opt-in flag).

2,880 candidate records and 336 selected records across 7 conditions and 4
methods were re-checked for: seed completeness (0-11), contiguous candidate
order, finite in-range validation fidelity, no duplicate selected rows, and
agreement between each trajectory's validation maximum and its stored
selected result. All checks passed. Test columns are dropped by an explicit
input allowlist at ingestion; the historical `anytime_mean.csv` files, which
aggregate *test* fidelity, are not read and play no part in budget selection.

Every checkpoint value quoted in the request was reproduced exactly: at
target 0.95 on the 4-qubit Ising reference, LLM-Open / LLM-Closed reach
0/12 and 8/12 at B=4, 10/12 and 12/12 at B=8, and 12/12 and 11/12 at B=16;
the XXZ B=8 LLM-Closed cell is 9/12; and no cell anywhere passes at 0.99.

The re-used conditions are controls, not new experiments, and are not counted
as such anywhere below.

## 3. What was newly executed

Two cells, in the pre-registered priority order.

**E-A. 4-qubit Ising chain, reference model, B = 6, all four methods,
seeds 0-11, closed loop split 3 + 3.** Anchored at the study reference
condition (4 qubits / Ising / B = 8 / reference model); only the budget
differs. LLM-Open drew a fresh 6-candidate pool -- not a prefix of the B = 8
pool. Random and Greedy were run at the same B = 6 so the controls are
budget-matched.

**E-B. 4-qubit XXZ chain, reference model, B = 10, LLM-Closed only,
seeds 0-11, split 5 + 5.** Anchored at the XXZ B = 8 condition, which is
itself a clean one-factor change from the reference. Because only LLM-Closed
was executed, **no same-budget cross-method comparison exists at this cell
and none is reported.**

The one-factor verifier of the earlier study was not disabled or weakened.
It gained an optional explicit anchor; the default anchor is unchanged, every
old condition still verifies against the reference, a second anchor must
itself be a clean one-factor condition, and the XXZ B = 10 cell is still
rejected if checked against the Ising reference. Both new cells carry a
manifest whose frozen block is byte-identical to the reference's.

## 4. Attainment at target 0.95

Seeds (of 12) whose best-so-far validation fidelity reaches the target by the
end of the configured budget. **P** marks a cell meeting the 10-of-12 rule.
"TFIM" is the transverse-field Ising chain; the slides call it the Ising
chain. "not run" means the method was never executed at that cell, which is
not the same as reaching zero seeds.

| Condition | Configured B | Source | Random | Greedy | LLM-Open | LLM-Closed |
|---|---:|---|---:|---:|---:|---:|
| 4-qubit TFIM, reference model | 4 | re-analysed | 0/12 | 0/12 | 0/12 | 8/12 |
| 4-qubit TFIM, reference model | 6 | measured now | 0/12 | 2/12 | 12/12 **P** | 8/12 |
| 4-qubit TFIM, reference model | 8 | re-analysed | 1/12 | 1/12 | 10/12 **P** | 12/12 **P** |
| 4-qubit TFIM, reference model | 16 | re-analysed | 2/12 | 3/12 | 12/12 **P** | 11/12 **P** |
| 4-qubit XXZ, reference model | 8 | re-analysed | 2/12 | 3/12 | 0/12 | 9/12 |
| 4-qubit XXZ, reference model | 10 | measured now | not run | not run | not run | 5/12 |
| 6-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 2/12 | 0/12 |
| 8-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, alternative model | 8 | re-analysed | 1/12 | 1/12 | 7/12 | 5/12 |

## 5. Attainment at target 0.99

| Condition | Configured B | Source | Random | Greedy | LLM-Open | LLM-Closed |
|---|---:|---|---:|---:|---:|---:|
| 4-qubit TFIM, reference model | 4 | re-analysed | 0/12 | 0/12 | 0/12 | 1/12 |
| 4-qubit TFIM, reference model | 6 | measured now | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, reference model | 16 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit XXZ, reference model | 8 | re-analysed | 0/12 | 1/12 | 0/12 | 0/12 |
| 4-qubit XXZ, reference model | 10 | measured now | not run | not run | not run | 0/12 |
| 6-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 8-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, alternative model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |

## 6. What the two new cells show

**The open loop needs less budget than the closed loop on the Ising (TFIM) chain.**
At B = 6 the open loop reaches 12/12 and passes; the closed loop
reaches 8/12 and does not. Non-semantic search is nowhere near:
Random 0/12, Greedy 2/12 at the same budget and on the
same seeds. Among the budgets actually run on this ladder, the smallest
verified passing budget is therefore **6 for the open loop** and **8 for the
closed loop**. B = 4 is a verified failure for both.

The most likely reading is structural rather than mysterious: at B = 6 the closed loop
spends half its budget on only three semantic warm starts and the other half
on three feedback-driven redesigns, while the open loop spends all six on one
semantic batch. When the budget is small, splitting it in two appears to hurt more than
feedback helps. This study varied only the budget, so that reading is an
interpretation of the observed split, not an isolated cause. The B = 6 attainment curve shows this directly -- the closed
loop climbs from 5/12 to 8/12 only across the refinement half, and still ends
below the pass line.

**Raising the budget did not rescue the XXZ cell; it made the count worse.**
The XXZ closed loop was 9/12 at B = 8, one seed short of the rule.
At B = 10 it is 5/12. This is reported as measured. It is not a
collapse in fidelity: the mean final best-so-far validation moved by only
-0.0076 (0.9588 to 0.9511). The whole XXZ distribution sits
within a few thousandths of the target -- 17 of the
24 seed results across the two runs are within 0.01 of 0.95 --
so the pass count is a knife-edge statistic there and a tiny shift flips
several seeds at once. See the per-seed margin figure.

**Attainment counts are not monotone in the budget, as measured.** On the
TFIM ladder the open loop goes 0, 12, 10, 12
and the closed loop 8, 8, 12, 11 at
target 0.95. On the XXZ anchor the closed loop goes 9 then
5. Each of these is an independently executed run of a different
policy, and stochastic variation between them is real. This is the concrete
reason the study reports verified passes, verified failures and unverified
budgets rather than an interpolated minimum or a bracketing interval: the
data do not support the monotonicity such a claim would require.

**Neither new cell moves the 0.99 target.** Both are 0/12, as is every
earlier condition. Nothing here identifies whether search, training length or
circuit capacity is the binding constraint at that target.

## 7. Within-run first-hit indices for the new cells

The first candidate index at which at least 10 of 12 seeds have reached 0.95,
*within* the run configured at that budget.

| Condition | Configured B | Method | First k with 10/12 within the run |
|---|---:|---|---:|
| target_tfim_b6 | 6 | Random | not reached |
| target_tfim_b6 | 6 | Greedy | not reached |
| target_tfim_b6 | 6 | LLM-Open | 2 |
| target_tfim_b6 | 6 | LLM-Closed | not reached |
| target_xxz_b10 | 10 | LLM-Closed | not reached |

These indices describe a prefix of an executed run. A prefix is **not** a
separately executed budget: the candidate pool was requested and paid for as
a whole, and a smaller configured budget would have changed the exploration
count, the request size and the refinement horizon. Nothing here licenses
calling a prefix of length k a "B = k experiment", and nothing here refunds
the generation cost already incurred.

For the closed loop, the first half of every run is semantic exploration with
no score attached; feedback-driven redesign begins only after it. Attainment
reached before that boundary cannot be attributed to the feedback loop. The
open-loop pool is shared across the data/training seeds, so 12 successes mean
12 seeds evaluated on one pool, not 12 independent generations.

## 8. Generation validity

Invalid proposals, repaired proposals and random fallbacks are recorded
distinctly. A proposal that fails the exact gate contract is first repaired
within a bounded number of attempts and, failing that, replaced by a flagged
random draw. **Random fallbacks consume candidate budget and remain in the
primary analysis.** The fallback-excluding attainment column in the audit
tables is an auxiliary, non-causal diagnostic: filtering existing logs cannot
regenerate an adaptive search history.

| Cell | Method | Evaluated | Random fallbacks | With recorded invalid errors |
|---|---|---:|---:|---:|
| target_tfim_b6 | Random | 72 | 0 | 0 |
| target_tfim_b6 | Greedy | 72 | 0 | 0 |
| target_tfim_b6 | LLM-Open | 72 | 0 | 0 |
| target_tfim_b6 | LLM-Closed | 72 | 5 | 4 |
| target_xxz_b10 | LLM-Closed | 120 | 7 | 7 |

## 9. Cost, calls and time

| Cell | Candidate evaluations | API calls | Repair/retry calls | Input tokens | Output tokens | Cost at list price (USD) |
|---|---:|---:|---:|---:|---:|---:|
| target_tfim_b6 | 288 | 57 | 8 | 43547 | 18187 | 0.1145 |
| target_xxz_b10 | 120 | 81 | 9 | 65969 | 28059 | 0.1757 |
| **total** | **408** | **138** | **17** | **109516** | **46246** | **0.2902** |

Prices are the provider's published standard-tier list prices for the
reference model snapshot, read on 2026-09-08 from
`https://developers.openai.com/api/docs/pricing`; token counts come from the
stored per-call records. Total new spend was **USD 0.2902** across
138 model calls (17 of them repair or retry calls),
109516 input and 46246 output tokens, and 408
candidate circuits trained and evaluated. Wall-clock timings are recorded per
execution segment in `run_timeline.json` rather than as a single number: the
B = 6 cell was interrupted and resumed, and both cells were later regenerated
from stored artefacts, so no invocation measures a full cold run of both.
The one clean cold measurement is the XXZ cell: 81 calls in 227 seconds.

Spending control: paid calls are refused outright unless `LLM_API_BUDGET_USD`
is set to an explicit nonzero cap, and the remaining cap is checked *before*
each request. The user authorised a hard cap of USD 2.00 for this study. The
B = 6 run was interrupted by a transient DNS failure after 10 of 12 seeds;
it was resumed with the cap reduced by the spend already incurred, so the
cumulative bound held across the interruption. The resume re-read the 10
stored seeds and the stored pool from disk and made no new call for them --
results are reused only when the configured budget, model snapshot, prompt,
seed, initialisation and data all match exactly.

## 10. Test-set protection

No candidate was evaluated on the test set anywhere in this study. The runner
hands the frozen trainer an empty test array, so no test state is ever
contracted with a trained circuit; a guard fails the run if any recorded test
quantity turns out to be finite; and the written tables carry a
validation-only column allowlist with no test column in it. Budgets,
circuits and analysis choices were selected on validation alone.

Earlier studies did record test outputs. Those are historical descriptive
results, not a fresh untouched confirmation set, and none is reported here.
Any future confirmatory evaluation must be pre-registered separately, frozen
before it is looked at, and never fed back into search.

## 11. What is verified, what failed, what is unverified

**4-qubit Ising chain, reference model**, target 0.95:

- `Random` - smallest verified passing budget: none among those run; verified failing budgets: 4, 6, 8, 16; unverified budgets: 2, 10, 12, 14 (and every budget above 16).
- `Greedy` - smallest verified passing budget: none among those run; verified failing budgets: 4, 6, 8, 16; unverified budgets: 2, 10, 12, 14 (and every budget above 16).
- `LLM-Open` - smallest verified passing budget: **B = 6**; verified failing budgets: 4; unverified budgets: 2, 10, 12, 14 (and every budget above 16).
- `LLM-Closed` - smallest verified passing budget: **B = 8**; verified failing budgets: 4, 6; unverified budgets: 2, 10, 12, 14 (and every budget above 16).

**4-qubit XXZ chain, reference model**, target 0.95:

- `LLM-Closed` - smallest verified passing budget: none among those run; verified failing budgets: 8, 10; unverified budgets: 2, 4, 6, 12, 14, 16 (and every budget above 10).

At target 0.99 no method passes at any budget or condition tested, including the two newly executed cells. That target is unresolved, and the runs performed here do not identify whether search, training or circuit capacity is the binding constraint.

A verified passing budget is **not** evidence that a smaller untested budget fails, and a verified failing budget is not evidence that every smaller budget fails. Each budget was executed as its own policy.

## 12. Limitations

Twelve paired seeds; noiseless state-vector simulation; one circuit-capacity
rule (3 rotations + 1 CNOT per qubit); one model snapshot for the new cells;
a single XXZ anchor. Budgets were run as separate policies, so no monotonicity
in `B` is assumed and no bisection was performed. The open-loop pool is shared
across seeds. Random fallbacks are inside the operational score, so generation
validity and architecture quality are entangled in every LLM number. Nothing
here separates insufficient search from insufficient training or insufficient
circuit capacity; the 0.99 result in particular is not evidence about which of
those binds.

## 13. Reproduction

```bash
# 1. Re-run the validation-only audit over the historical logs and the two
#    new cells (no model calls, no test evaluation).
python scripts/qae/analyze_budget_targets.py \
  --root outputs/qae_robustness \
  --out outputs/qae_budget_targets_v2_20260908/audit \
  --extra-condition target_tfim_b6:6:all:outputs/qae_budget_targets_v2_20260908 \
  --extra-condition target_xxz_b10:10:LLM-Closed:outputs/qae_budget_targets_v2_20260908

# 2. Figures, report and deck.
python scripts/qae/build_budget_target_figures.py \
  --audit outputs/qae_budget_targets_v2_20260908/audit
python scripts/qae/build_budget_target_report.py \
  --audit outputs/qae_budget_targets_v2_20260908/audit \
  --runs outputs/qae_budget_targets_v2_20260908 \
  --body docs/research/templates/BUDGET_TARGET_REPORT.md.tmpl \
  --out outputs/qae_budget_targets_v2_20260908/REPORT.md
python scripts/qae/build_budget_target_deck.py \
  --audit outputs/qae_budget_targets_v2_20260908/audit \
  --runs outputs/qae_budget_targets_v2_20260908

# 3. Tests.
python -m pytest tests/test_qae_budget_targets.py -q
```

Re-running the paid cells requires an explicit cap and the stored artifacts
removed (otherwise they are correctly reused, not re-paid):

```bash
LLM_API_BUDGET_USD=2.00 python scripts/qae/run_budget_target.py --cells all
```

`--status` reports which seeds are already on disk, and `--estimate` prints
the pre-run upper bound on calls, tokens and cost.
