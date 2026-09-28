# Root Cause of the QKAN Baseline AUC Regression

**Scope:** baseline (pre-training) AUC of the warm-started QKAN on the top-tagging task.
**Status:** historical. The regression was resolved by fitting every edge at a fixed Chebyshev degree 4 without the
R² gate; see `reports/03_chebyshev_degree_floor_vs_fixed_degree.md` and
`reports/04_chebyshev_fixed_degree_full_seed_sweep.md`. Line numbers and the 15-way split refer to the code at the
time of writing.

## Summary

The QKAN **baseline evaluation** is the AUC computed right after `SymbolicWarmStartExtractor.extract_and_save`
warm-starts the quantum circuit from the B-splines of the classical KAN, before any quantum fine-tuning
(`QuantumKANTrainer.fit()`). Its value used to be consistently above 0.5. After commit `38f6900` (brute-force
Chebyshev-degree search) and commit `7717d38` (canonical 15-way balanced subsample split), it became consistently
**below 0.5** (about 0.26–0.30) on every checkpoint trained after the split.

The result was verified empirically rather than inferred from the diffs. The extraction and baseline evaluation
(extractor → `QKANModel(backend_mode="ideal")` → sigmoid → `roc_auc_score`) were run directly, varying one factor
at a time, without running `scripts/train_qkan.py` or any quantum training loop.

**Root cause: an interaction between two changes, not a single bug.** The brute-force minimum-degree Chebyshev fit
of `38f6900` is not incorrect in itself: applied to a classical model trained on the previous, larger dataset it
reproduces the previous behaviour almost exactly (AUC 0.81 → 0.81). Applied to any classical model trained on the
new, much smaller 15-way subsamples (`7717d38`), it consistently collapses the baseline AUC to about 0.26–0.30. The
previous fixed degree-4 fit does **not** collapse on the same smaller-data checkpoints (0.81). Neither the smaller
per-fold training set nor the new extractor is sufficient alone; the combination causes the regression.

## Method

For each seed's `03_retrained` classical checkpoint (the same checkpoint for the previous and the current
extractor), two factors were held fixed and one was varied:

- **Fixed:** the classical checkpoint (`HEPKAN.loadckpt`) and the test split
  (`processor_top.load_and_preprocess_data(seed=...)`).
- **Varied:** the Chebyshev extraction logic. Either the *previous* method (reconstructed verbatim from
  `git show 38f6900^:src/architectures/extractor.py`: always `chebfit(x_vals, y_vals, deg=4)`), or the *current*
  method at the time (`extractor.py:106-132`: brute-force search over `degree ∈ 1..4`, accepting the first degree
  with R² ≥ 0.95, falling back to the best R², then zero-padding to a uniform graph-wide degree).
- **Measured:** baseline AUC from a direct `QKANModel(graph_path=..., backend_mode="ideal")` forward pass,
  `torch.sigmoid` and `sklearn.roc_auc_score` on the fixed test split. This is the computation performed by
  `QuantumKANTrainer.evaluate_baseline`, called directly so that no quantum training runs.

The diagnostic script (`compare_chebyshev_extraction.py`) was not committed to the repository.

## Results

| Checkpoint | Classical training data | Extraction logic | Hidden nodes (sum/mult) | Baseline AUC |
|---|---|---|---|---|
| `seed_12` (post-split) | 1/15 balanced subsample | **previous** (fixed deg=4) | 0/4 | **0.811** |
| `seed_12` (post-split) | 1/15 balanced subsample | **current** (brute-force) | 0/4 | **0.259** |
| `seed_5` (pre-split) | full balanced dataset | current (brute-force) | 6/0 | **0.806** |
| `seed_3` (post-split) | 1/15 balanced subsample | current (brute-force) | 4/0 | 0.298 |
| `seed_13` (post-split) | 1/15 balanced subsample | current (brute-force) | 0/3 | 0.290 |
| `seed_14` (post-split) | 1/15 balanced subsample | current (brute-force) | 0/3 | 0.293 |
| `seed_42` (post-split) | — | — | — | *excluded: no `03_retrained` checkpoint on disk* |

Comparison of both extractors on the same model and test data (`seed_12`):

- **Output-edge `coefs[0]`** (the only coefficient used by the hidden → output "Stage 2" readout,
  `qkan_model.py:189,191`) changed very little, e.g. `-0.286 → -0.298`, `-0.107 → -0.121`, `0.272 → 0.280`, with no
  sign changes.
- **Input-edge (Stage 1) coefficients changed substantially**, because the current extractor lets each edge select
  its own minimum degree instead of always fitting degree 4. Several edges of `seed_12` settled at **degree 1**. For
  example, wire 0 / input 0 gives `[0.098, -0.936, 0, 0]`, whereas the fixed degree-4 fit of the same edge gives
  `[0.114, -0.886, 0.058, 0.085, -0.045]`. The current fit discards curvature captured by the previous fit, because a
  straight line through the 500 sampled points of that edge already reaches R² ≥ 0.95. Since `chebfit` is an
  ordinary discrete least-squares fit (not an orthogonal projection), coefficients at different degrees are not
  nested: an early truncation does not only drop the high-order terms but also **redistributes energy across the
  low-order terms**. The modified per-wire rotation angles feed a Chebyshev-basis data re-uploading circuit
  (`_qkan_edge`, `qkan_model.py:158-167`) whose output depends on the exact angle sequence, so small per-edge
  coefficient shifts compound over the circuit into a large change in AUC.

Causes ruled out:
- **PennyLane parameter broadcasting (batched `forward()`, commit `3597aec`)**: numerically identical to the previous
  per-sample loop according to `verify_broadcasting.py` and `verify_mult_edge.py` (both `Equal? True`, exact
  `torch.allclose`).
- **IsingZZ approximation of multiplication nodes**: the pruned graph of `seed_3` has **no** multiplication nodes
  (all 4 hidden nodes are `sum`), yet its baseline AUC still collapses to 0.298 under the current extractor. The
  regression therefore occurs without any IsingZZ gate, and the multiplication approximation is not the deciding
  factor.
- **Stale `quantum_weights.pt` cache / train-serve skew**: no `quantum_weights.pt` existed under `outputs/` at the
  time of the investigation, so every extraction in this report used the current code.

## Does the 15-way split make baseline AUC vary between folds?

No. Under the current extractor, every post-split fold tested (`seed_3`, `seed_12`, `seed_13`, `seed_14`) falls in a
narrow band of **AUC ≈ 0.26–0.30**. The effect is systematic, not fold-dependent. The only checkpoint that does not
collapse (`seed_5`, AUC 0.806) is the only classical model trained on the previous, much larger dataset. The split
changed the classical training data enough to expose a latent fragility of the brute-force degree search; it does
not introduce variation between folds within the new regime.

## Conclusions

1. **Not a correctness bug** (no sign flip, label inversion or index scrambling). The regression is a **fragility of
   the brute-force minimum-degree Chebyshev fit** (`extractor.py:106-132`) on B-splines learned from much smaller
   (1/15) training subsamples. Noisier or flatter splines pass the `r2_threshold=0.95` gate at very low degrees
   (often degree 1), discarding curvature that the fixed degree-4 fit preserves. Since the discrete least-squares
   fit is not nested by degree, the surviving low-order coefficients are also perturbed.
2. **Candidate fix:** a more conservative degree search, i.e. a higher `chebyshev_max_degree` and/or
   `chebyshev_r2_threshold`, or a *minimum* accepted degree (e.g. degree ≥ 3), rather than accepting degree 1 on a
   noisy spline. This keeps the fit-quality gate of `38f6900` while addressing the collapse.
3. **Possible second contributor:** 1/15 of the balanced dataset may be too small for the classical KAN to learn
   splines clean enough for a reliable downstream degree search. A comparison of the classical retrain AUC/loss of
   post-split and pre-split checkpoints is proposed as follow-up.
4. Batching and the IsingZZ approximation are confirmed not to be responsible.

The candidate fixes of item 2 were evaluated in `reports/02_chebyshev_degree_ceiling_comparison.md` and
`reports/03_chebyshev_degree_floor_vs_fixed_degree.md`.
