# Chebyshev Degree Ceiling vs. the Historical Post-Symbolic Extraction: Baseline AUC Comparison

**Scope:** baseline (pre-training) AUC of the warm-started QKAN, ideal backend, top-tagging task.
**Status:** historical. Superseded by `reports/03_chebyshev_degree_floor_vs_fixed_degree.md` and
`reports/04_chebyshev_fixed_degree_full_seed_sweep.md`.

## Background

`reports/01_qkan_baseline_auc_regression.md` explained why the QKAN **baseline evaluation** (AUC
computed right after the Chebyshev warm-start extraction, before any quantum fine-tuning) dropped from consistently
above 0.5 to about 0.26–0.30 on every classical checkpoint trained after the 15-way balanced subsample split. The
cause was the brute-force *minimum-degree* Chebyshev search in `SymbolicWarmStartExtractor._fit_edge`: on the noisy
or flat splines obtained from the smaller per-fold training data, many edges pass the R² ≥ 0.95 gate at a very low
degree, discarding curvature that a fixed higher-degree fit would retain. That report ruled out a sign-flip or
label-inversion bug and proposed a more conservative search (e.g. a higher `chebyshev_max_degree`) as the next test.

This report performs that test: baseline (pre-training) AUC on the ideal backend for two `chebyshev_max_degree`
settings (3 and 4), and for a reconstruction of the historical extraction method used before commit `3597aec`
(fixed degree 4, no search, fitted on the checkpoint obtained **after** symbolic simplification and fine-tuning
instead of the pre-symbolic checkpoint).

## Method

Seeds with an on-disk classical checkpoint were used (3, 5, 12, 13, 14). Seed 42 was excluded because no
checkpoint was trained for it after the split. Baseline AUC was measured by building
`QKANModel(graph_path=..., backend_mode="ideal")` from each extracted graph, running a forward pass over the
held-out test set, applying `torch.sigmoid` and computing `roc_auc_score`. This is the computation of
`QuantumKANTrainer.evaluate_baseline`, called directly so that no quantum training loop runs. Four configurations
were compared:

- **`max_degree=3`** and **`max_degree=4`**: the extractor at the time (`extractor.py`, reading `act_fun` from the
  `03_retrained` checkpoint) with `chebyshev_max_degree` set to 3 or 4, keeping its R² ≥ 0.95-gated minimum-degree
  search (starting at degree 2).
- **Previous approach (as written):** the extraction logic before commit `3597aec`, reconstructed verbatim from
  commit `146e021`: fixed `degree=4`, no R² search, fitted on `symbolic_fun` of the **`05_final`** checkpoint (after
  symbolic simplification and fine-tuning).
- **Previous approach (subnode-bias fixed):** the same logic with one patch, replacing `node_bias`/`node_scale` by
  `subnode_bias`/`subnode_scale`. This is the same shape correction used by the current extractor. It separates the
  question "does fixed-degree post-symbolic extraction work once an unrelated shape bug is fixed?" from the
  question "does the minimum-degree search truncate?".

No committed file was modified. The comparison driver and both reconstructed extractor variants were not committed
to the repository.

## Results

| Seed | Structure (sum/mult, qubits) | `max_degree=3` | `max_degree=4` | Previous approach (as written) | Previous approach (subnode-bias fixed) |
|---|---|---|---|---|---|
| 3 (post-split) | 4 sum / 0 mult, 4 qubits | 0.3653 | 0.3653 | 0.8271 | 0.8271 |
| 5 (pre-split) | 6 sum / 0 mult, 6 qubits | 0.6022 | 0.8403 | 0.8358 | 0.8358 |
| 12 (post-split) | 0 sum / 4 mult, 7 qubits | 0.3057 | 0.3057 | **ERROR** (shape mismatch, tensor a=14 vs b=9) | 0.4837 |
| 13 (post-split) | 0 sum / 3 mult, 6 qubits | 0.3492 | 0.3492 | **ERROR** (shape mismatch, tensor a=14 vs b=9) | 0.8425 |
| 14 (post-split) | 0 sum / 3 mult, 6 qubits | 0.3647 | 0.3647 | **ERROR** (shape mismatch, tensor a=14 vs b=9) | 0.1463 |

The as-written previous approach fails on every seed whose surviving hidden layer consists only of multiplication
nodes (12, 13, 14). Its use of `node_bias`/`node_scale` assumes that the widths before and after the
multiplication collapse are equal, which only holds when no multiplication node survives (seeds 3 and 5). This is
the same shape issue that the current extractor documents and fixes with `subnode_bias`/`subnode_scale`.

## Discussion

1. **Raising `chebyshev_max_degree` from 3 to 4 does not fix the collapse on the post-split seeds.** For seeds 3,
   12, 13 and 14 both settings give byte-identical AUC: the R²-gated search already settles at degree ≤ 3 for every
   edge, so the ceiling is not the active constraint on this data. The ceiling only matters for seed 5 (0.602 →
   0.840), the one checkpoint trained on the previous, larger pre-split dataset, where several edges require degree
   4. Raising the ceiling alone is therefore insufficient in the post-split regime; the search would need a **higher
   minimum degree**.
2. **The historical fixed-degree post-symbolic approach is not a clean replacement.** With its shape bug patched, it
   gives healthy AUC (> 0.8) for seeds 3, 5 and 13, but stays near or below chance for seeds 12 (0.484) and 14
   (0.146), both post-split checkpoints with multiplication-only hidden layers. The degree-search truncation is
   therefore not the only source of fragility: the `05_final` symbolic fit itself, or its interaction with the
   `IsingZZ`-chained multiplication readout for arity > 2, is inconsistent across seeds with the same qualitative
   structure. This is left as a separate open issue.
3. **No sign-flip bug in either extractor.** The as-written previous approach fails with an explicit shape-mismatch
   `RuntimeError`, not with silently wrong output, and neither reconstructed variant shows a sign-inversion pattern
   in its coefficients.

## Conclusions

Neither `chebyshev_max_degree` setting (3 or 4) resolves the baseline-AUC collapse on the post-split checkpoints:
the binding constraint is the willingness of the R² gate to accept degree 2, not the ceiling. For the current
(`act_fun` / `03_retrained`) extraction path, the more promising lever is the **minimum** degree the search may
accept (e.g. degree ≥ 3). Reverting to the historical fixed-degree post-symbolic approach is not a safe substitute:
it requires the `subnode_bias`/`subnode_scale` fix to run at all, and it is inconsistent across the post-split seeds
with multiplication-only hidden layers (healthy for seed 13, poor for seeds 12 and 14).

The minimum-degree hypothesis is tested in `reports/03_chebyshev_degree_floor_vs_fixed_degree.md`.
