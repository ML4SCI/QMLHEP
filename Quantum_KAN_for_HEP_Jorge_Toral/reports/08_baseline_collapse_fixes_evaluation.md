# Experimental Evaluation of the Candidate Fixes for the Baseline QKAN Collapse

**Date:** 2026-09-16
**Scope:** the five candidate causes and fixes listed in Section 5 of
`reports/06_baseline_collapse_investigation.md`. Each cause was implemented and evaluated
experimentally, with before/after measurements and a verdict (adopted, rejected or inconclusive).
**Status:** complete. Causes #1–#4 were evaluated on isolated copies and none was adopted; cause #5 was resolved
directly. Script names refer to the code at the time of writing.

**Method.** Tests were run mainly on **seed 12**, chosen because it has usable `01_base`/`03_retrained` checkpoints
and a strongly collapsed baseline AUC. **Seed 13** was used to confirm the one finding that appeared
seed-specific. Experimental scripts were not committed to the repository.

## Summary of results

**None of the five hypothesized causes, once tested, explains most of the observed behaviour.** The main driver is
the input bottleneck created by classical pruning (a refined form of cause #3). In addition, and specific to seed 12,
a **stale `03_retrained` checkpoint**, unrelated to the five causes, made seed 12 appear considerably worse than its
structure implies.

| Cause | Verdict | Evidence |
|---|---|---|
| #1 Fixed 0.5 threshold | **Rejected** | The Youden's-J-optimal threshold (0.4974) is almost identical to 0.5; AUC is threshold-invariant and stays at 0.4457 in both cases |
| #2 Warm-start bias | **Rejected** | Zero-centering the degree-0 term of `edge_weights` changes AUC by < 0.003 (0.4457 → 0.4435); the bias term was already small (mean −0.011, std 0.032) |
| #3 Pruning reduces to 2 features | **Refined, not adopted as-is** | Confirmed and structural, but the mechanism is `node_th`/`edge_th`, not `max_fanin`; loosening the thresholds enough to matter makes ideal/noisy simulation computationally infeasible at the resulting qubit counts |
| (additional) Stale `03_retrained` checkpoint | **Confirmed for seed 12 only** | A fresh, deterministic retrain from the current `01_base` with identical production thresholds gives baseline AUC 0.6995 vs. 0.4457 for the stored checkpoint; seed 13 shows no such gap (0.7008 fresh vs. 0.7007 stored) |
| #4 Weak gradient signal | **Rejected** | Gradient norms of `edge_weights`/`output_weights` are healthy and non-vanishing (about 0.06 / 0.005) over 5 training epochs; a full training run improves AUC (0.445 → 0.65 baseline → trained, seed 12) |
| #5 Script/behaviour mismatch | **Resolved** | The seed-sweep script builds the no-mass-cut partition it declares, instead of relying on whatever partition was cached |

---

## Cause #1: fixed 0.5 threshold

**Test.** The persisted baseline-ideal probabilities of seed 12 (`qkan_eval_probs_baseline_ideal.npy` /
`qkan_eval_true_baseline_ideal.npy`)
were loaded, and the hardcoded 0.5 threshold was compared with the Youden's-J-optimal threshold of the same ROC
curve.

```
AUC (threshold-independent): 0.4457
Youden's J optimal threshold: 0.4974 (J=0.0072)
fixed 0.5             thresh=0.5000  Acc=0.4827  F1=0.5745  Precision=0.4879  Recall=0.6985
Youden J optimal      thresh=0.4974  Acc=0.5035  F1=0.6618  Precision=0.5018  Recall=0.9715
```

**Verdict: rejected.** AUC cannot change under a threshold change, and it does not. The optimal threshold is almost
exactly 0.5, and J = 0.0072 shows that the model has essentially no separating power at any threshold in this
state. Threshold calibration is not adopted as a fix for the collapse. It may still be a reasonable general
improvement of accuracy and F1 once the model has signal, but that is a separate and smaller concern.

## Cause #2: warm-start bias (zero-centering the degree-0 term)

**Test.** The production `quantum_weights.pt` of seed 12 (10 qubits, inexpensive to evaluate) was loaded into a
`QKANModel`. `edge_weights[:, -1]` (the degree-0 constant Chebyshev term of the `_qkan_edge` encoding, applied as a
standalone `RY(weights[degree])` without a data-dependent `RZ`) and the mean of `output_weights[:, 0]` were
zero-centered before running `evaluate_baseline`.

```
edge_weights degree-0 (bias) term stats before zeroing: mean=-0.0111, std=0.0321, min=-0.1111, max=0.0000
UNMODIFIED warm start:         AUC=0.4457  Recall=0.6985  Precision=0.4879
ZERO-CENTERED degree-0 term:   AUC=0.4435  Recall=0.7657  Precision=0.4900
```

**Verdict: rejected.** The bias term was already small and not systematically large. Forcing it to zero changes AUC
by less than 0.003 (noise level) and does not materially change the balance of the confusion matrix. Warm-start
bias does not drive the collapse.

## Cause #3: pruning reduces the input to two features

**Test 1 (structural dry run).** The `01_base` checkpoint of seed 12 was reloaded and `prune_and_save_kan` was rerun
with several threshold configurations, counting the raw input columns with at least one surviving edge
(`act_fun[0].mask`) after node/edge pruning and the fan-in cap:

```
current (production)          node_th=0.04  edge_th=0.06  fanin=2    -> 2 live inputs [0, 1]
                                 [prune_fanin] capped 0/19 hidden neurons  <- fanin cap did NOTHING here
fanin cap raised to 4                                       fanin=4    -> 2 live inputs [0, 1]  (unchanged)
fanin cap disabled                                          fanin=None -> 2 live inputs [0, 1]  (unchanged)
loosened node/edge 3x         node_th=0.013 edge_th=0.02   fanin=2    -> 3 live inputs [0, 1, 3]
loosened node/edge+fanin4     node_th=0.013 edge_th=0.02   fanin=4    -> 5 live inputs [0, 1, 3, 6, 17]
```

This **corrects the framing of the investigation report**: at the production thresholds, pykan's native `prune()`
(through `node_th`/`edge_th`) already reduces every hidden neuron to at most two input edges; the
`prune_max_fanin=2` hard cap never engages ("capped 0/19 hidden neurons"). Loosening `max_fanin` alone has no
effect; `node_th`/`edge_th` are the effective parameters.

**Test 2 (end-to-end, loosened `node_th`/`edge_th`).** Loosening these thresholds by a factor of 3 increases the
surviving feature set from two to three raw inputs, but the resulting quantum graph requires **21 qubits** instead
of 10 (more surviving hidden nodes and edges). On `lightning.qubit` (ideal backend), an evaluation of only 60 rows
had not completed after more than 6 minutes, compared with about 80 s for the full test set of about 9,000 rows at
10 qubits, consistent with the exponential cost of state-vector simulation in the number of qubits. The noisy
backend (already about 2,600 s per evaluation at 10 qubits) would be infeasible at 21 qubits with the current
simulation approach.

**Verdict: refined, not adopted as-is.** The reduction to two features is real and structural, and `node_th`/`edge_th`
(not `max_fanin`) are its cause. Loosening them enough to matter, however, raises the qubit count to a level at
which evaluation becomes computationally infeasible with the current one-qubit-per-accumulator circuit design. A
practical fix requires a change of circuit design or simulation approach (see
`reports/05_vqc_jax_vs_torch_benchmark.md`), or a reduction of qubits per active input; this is outside
the scope of this evaluation.

## Additional finding: stale `03_retrained` checkpoint of seed 12

While testing cause #3, a **fresh** retrain with the production configuration (same `01_base` checkpoint, same
thresholds, data and hyperparameters as Step 4 of `train_kan.py`) produced a baseline AUC very different from that of
the stored checkpoint used by the pipeline:

```
production checkpoint (outputs/top/seed_12/.../metrics_qkan_baseline_ideal.json):  AUC=0.4457
fresh retrain of the SAME 01_base, SAME thresholds (reseed=12):                      AUC=0.6995
fresh retrain, repeated with a different reseed (reseed=999):                        AUC=0.6995  (bit-identical)
```

`retrain_pruned_kan` is fully deterministic for identical inputs (two different reseed values give bit-identical
results to four decimals). The gap therefore implies that the stored `03_retrained` checkpoint of seed 12 was
produced under different conditions than the current `01_base` and hyperparameters, i.e. it is **stale**; it is not
evidence of a flaw in the circuit or the training. This is consistent with a known open issue of seed 12: its
`05_final` checkpoint fails with a NaN on a fresh `--force` retrain under the current hyperparameters (see
`reports/03_chebyshev_degree_floor_vs_fixed_degree.md`). The classical checkpoints of seed 12 are out of sync
with the current code and hyperparameters.

**Seed-specific, not general.** The same experiment on **seed 13** shows no gap: stored baseline AUC 0.7007 vs.
0.7008 for a fresh retrain. The issue is a data-hygiene problem limited to the cached checkpoints of seed 12, not a
systematic bug.

**Not applied.** The `02_pruned`/`03_retrained`/`04_symbolic`/`05_final` checkpoints of `outputs/top/seed_12` were not
overwritten, to preserve existing results. **Recommendation:** rerun `train_kan.py --seed 12 --force` from Step 3
onward (a full `--force` also retrains the base model, which is unnecessary) to refresh the classical checkpoints of
seed 12, then rerun `train_qkan.py --seed 12 --force`. This is expected to improve the baseline (and probably the
trained) AUC of seed 12 materially, once the NaN failure in symbolic fine-tuning (a separate open issue) is also
resolved.

## Cause #4: weak gradient signal

**Test.** Five training epochs (same optimizer and data-sampling logic as `QuantumKANTrainer.fit()`,
`n_train_samples_for_epoch=1000`, `qkan_batch_size=1024`) were run on the 10-qubit production graph of seed 12,
logging the norms of `edge_weights.grad`/`output_weights.grad` per epoch:

```
Epoch 1/5: loss=0.71394  edge_weights.grad norm mean=0.063122  output_weights.grad norm mean=0.007869
Epoch 2/5: loss=0.71620  edge_weights.grad norm mean=0.068501  output_weights.grad norm mean=0.005215
Epoch 3/5: loss=0.70968  edge_weights.grad norm mean=0.058337  output_weights.grad norm mean=0.002298
Epoch 4/5: loss=0.71066  edge_weights.grad norm mean=0.061679  output_weights.grad norm mean=0.009003
Epoch 5/5: loss=0.71005  edge_weights.grad norm mean=0.056725  output_weights.grad norm mean=0.004634
```

The gradient norms are well below the `clip_grad_norm_(max_norm=1.0)` ceiling (not saturating) and show no vanishing
trend over five epochs. In addition, the full `train_qkan.py` run of seed 12 (see
`reports/07_metrics_collection_audit.md`) confirms that training improves the model: baseline AUC 0.4457 →
trained AUC 0.6503 (ideal backend) after the full 50-epoch schedule.

**Verdict: rejected.** Gradients are present and training improves the model. The limit on the trained model's
performance is much better explained by the two-input bottleneck (cause #3) than by a gradient problem.

## Cause #5: script/behaviour mismatch

**Finding.** The header of `scripts/run_seeds.sh` stated `apply_mass_cut=False` defaults, but its preprocessing call
was disabled, so the canonical partition used depended on whatever was already cached and was not necessarily the
no-mass-cut regime.

**Verdict: resolved.** This is a consistency issue with a single correct answer rather than an experimental
question. The sweep now builds the no-mass-cut partition it declares before training.
