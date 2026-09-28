# Chebyshev Degree Floor vs. Fixed Degree: Effect on QKAN Baseline AUC

**Scope:** baseline (pre-training) AUC of the warm-started QKAN, ideal backend, top-tagging task, seeds 12, 13 and 14.
**Status:** the recommended configuration (fixed degree 4, no R² gate) was subsequently adopted; its verification on
every available seed is reported in `reports/04_chebyshev_fixed_degree_full_seed_sweep.md`. Line numbers refer to
the code at the time of writing.

## Summary

Imposing a minimum Chebyshev degree of 3 does not recover the baseline AUC of seeds 12, 13 and 14, because the R²
gate still accepts the first degree that passes, which is always the new floor. Removing the R² gate and fitting
every edge at degree 4 recovers AUC ≈ 0.80–0.81 on all three seeds for the B-spline extraction path.

## Background

- `reports/01_qkan_baseline_auc_regression.md`: the brute-force minimum-degree Chebyshev search in the
  extractor collapsed the baseline AUC (AUC of the warm-started QKAN right after extraction, before quantum
  fine-tuning) from above 0.5 to about 0.26–0.36 on post-split seeds (3, 12, 13, 14). Root cause: `_fit_edge`
  (`src/architectures/extractor.py:105-145`) accepts the *first* degree whose R² exceeds 0.95. On the smaller,
  noisier 1/15 subsamples many edges settle at a low degree, and because `chebfit` is a discrete least-squares fit
  (not nested by degree), the early acceptance redistributes coefficient energy across *all* retained terms, not
  only the dropped high-order ones.
- `reports/02_chebyshev_degree_ceiling_comparison.md`: raising the degree *ceiling* (`max_degree` 3 → 4) had
  **no effect**; degree 3 and degree 4 gave byte-identical AUC on seeds 3, 12, 13 and 14, showing that the search
  already settles below degree 3 regardless of the ceiling. That report also reconstructed the historical
  extraction (commit `146e021`: fixed degree 4, no search, fitted on `symbolic_fun` of the post-symbolic-fine-tuning
  `05_final` checkpoint instead of the numeric B-spline branch of `03_retrained`) with the `node_bias`/`node_scale` →
  `subnode_bias`/`subnode_scale` shape fix. It was inconsistent on seeds with multiplication-only hidden layers:
  0.8425 (seed 13), 0.4837 (seed 12, near chance) and 0.1463 (seed 14, below chance). Both reports pointed to a
  **minimum accepted degree** as the missing lever.

## Method

Two standalone scripts (`tests/test_chebyshev_min_degree_bspline.py` and
`tests/test_chebyshev_min_degree_symbolic_final.py`; both removed from `tests/`; available in git history at commit `0b7c53b`) subclass `SymbolicWarmStartExtractor` without modifying
`src/architectures/extractor.py`. For seeds 12, 13 and 14 of task `top` (the only task with cached checkpoints for
these seeds), each script runs **two configurations** on its extraction branch:

1. **Gated search, degree ∈ [3, 4]:** degrees below 3 are never accepted; the same R² ≥ 0.95 threshold as the
   production extractor gates the search.
2. **Fixed degree 4, no gate:** every edge is always fitted at degree 4, with no early-acceptance check. This
   configuration was added after configuration 1 produced no change (see Results). It separates two previously
   conflated factors: the degree floor, and the policy of accepting the first degree that passes the R² threshold.

**Approach A** fits the numeric B-spline branch (`act_fun`) of the `03_retrained` checkpoint (the committed
extraction path). **Approach B** reconstructs the historical path (`symbolic_fun` on `05_final`, with the
`subnode_bias`/`subnode_scale` fix) but applies the same two configurations instead of the historical fixed
degree 4, so that both approaches are compared on equal terms.

All classical checkpoints (`03_retrained` for the three seeds, `05_final` for seeds 13 and 14) were cached and
current. **The `05_final` checkpoint of seed 12 predates the current `hyperparams.py`:** a later `--force` rerun of
`train_kan.py` failed with `Fatal error: Input contains NaN` during symbolic fine-tuning under the current
hyperparameters. This is a separate open issue, not addressed here because fixing it requires changes to
`hyperparams.py`/`classic_kan.py`. The Approach-B results for seed 12 are reported with this caveat.

Baseline AUC is computed directly (`QKANModel(backend_mode="ideal")` forward pass → sigmoid → `roc_auc_score`),
bypassing the training loop of `scripts/train_qkan.py`, as in both previous reports. No plots or metrics are
written to the production `outputs/.../results` tree, and the extracted graphs are written to a temporary directory,
never to the `quantum_weights.pt` cache. The **sign-flip diagnostic** is `roc_auc_score(y_test, 1 - probs)`,
checked against the identity `1 - AUC` (`assert abs(flipped_auc - (1 - auc)) < 1e-6`) for every result below.

## Results

| Seed | Approach | Config | Final degree | Degree histogram | Qubits | Hidden nodes | AUC | Flipped AUC |
|---|---|---|---|---|---|---|---|---|
| 12 | A (B-spline) | gated [3,4] | 3 | {3: 16} | 7 | 0 sum / 4 mult | 0.3069 | 0.6931 |
| 12 | A (B-spline) | fixed 4 | 4 | — | 7 | 0 sum / 4 mult | **0.8107** | 0.1893 |
| 13 | A (B-spline) | gated [3,4] | 3 | {3: 13} | 6 | 0 sum / 3 mult | 0.3455 | 0.6545 |
| 13 | A (B-spline) | fixed 4 | 4 | — | 6 | 0 sum / 3 mult | **0.8032** | 0.1968 |
| 14 | A (B-spline) | gated [3,4] | 3 | {3: 13} | 6 | 0 sum / 3 mult | 0.3546 | 0.6454 |
| 14 | A (B-spline) | fixed 4 | 4 | — | 6 | 0 sum / 3 mult | **0.8057** | 0.1943 |
| 12* | B (symbolic) | gated [3,4] | 3 | {3: 16} | 7 | 0 sum / 4 mult | 0.4015 | 0.5985 |
| 12* | B (symbolic) | fixed 4 | 4 | — | 7 | 0 sum / 4 mult | 0.4837 | 0.5163 |
| 13 | B (symbolic) | gated [3,4] | 3 | {3: 13} | 6 | 0 sum / 3 mult | 0.6525 | 0.3475 |
| 13 | B (symbolic) | fixed 4 | 4 | — | 6 | 0 sum / 3 mult | **0.8425** | 0.1575 |
| 14 | B (symbolic) | gated [3,4] | 3 | {3: 13} | 6 | 0 sum / 3 mult | 0.4697 | 0.5303 |
| 14 | B (symbolic) | fixed 4 | 4 | — | 6 | 0 sum / 3 mult | 0.1463 | **0.8537** |

\* The `05_final` checkpoint of seed 12 is the stale checkpoint described in Method; its Approach-B results carry
that caveat.

The Approach-B fixed-degree-4 results (0.4837 / 0.8425 / 0.1463 for seeds 12 / 13 / 14) reproduce exactly the
"previous approach (subnode-bias fixed)" column of `reports/02_chebyshev_degree_ceiling_comparison.md`, which
cross-validates the reimplementation used here.

## Discussion

1. **The gated degree-3/4 search changes nothing.** For every seed and both approaches, the degree histogram of the
   gated search shows all edges at degree 3: R² already exceeds 0.95 there, so degree 4 is never tried. The
   resulting AUCs (0.31–0.65) are within noise of the degree-3 results of the previous report. The previous
   hypothesis was therefore only partly correct: the floor matters, but raising it from 2 to 3 does not help
   because the R² gate keeps accepting the new floor and never reaches the ceiling.
2. **Removing the R² gate (fixed degree 4) recovers Approach A.** All three seeds improve from about 0.31–0.35 to
   about 0.80–0.81, above the 0.75 threshold used to consider the regression resolved and in the same range as the
   healthy pre-split seed 5 (0.84, from the previous report) and the historical-approach result of seed 13 (0.84).
   The mechanism matches the root-cause explanation of the first report: `chebfit` is a discrete least-squares refit
   at each degree, not a truncation of one fit, so accepting degree 3 instead of 4 produces different low-order
   coefficients, not only fewer high-order ones. The gate was hiding degree-4 fits that work.
3. **The same fix does not reliably rescue Approach B.** Fixed degree 4 improves seed 13 (0.65 → 0.84) but not seed
   12 (stale checkpoint, near chance at 0.48) nor seed 14, which becomes worse (0.47 → 0.15). The fixed-degree-4
   symbolic fit of seed 14 is confidently wrong, not merely uninformative.
4. **The sign-flip diagnostic characterizes seed 14 / Approach B.** The raw AUC (0.1463) and the flipped AUC (0.8537)
   sum to 1.0 and are both far from 0.5: this circuit is a well-separated classifier with an inverted output, not an
   uninformative one. The inversion is limited to Approach B at seed 14. It does not appear in Approach A with the
   fixed-degree-4 configuration, where the flipped AUCs (0.19–0.20) are as far below 0.5 as the raw AUCs are above
   it, as expected for an ordinary, non-inverted classifier.

## Resolved metrics

| Seed | Resolved | Configuration | Baseline AUC | Comparable to |
|---|---|---|---|---|
| 12 | **Yes** | Approach A, B-spline (`03_retrained`), fixed Chebyshev degree 4, no R² gate | **0.8107** | seed 5 (0.84), historical-approach seed 13 (0.84) |
| 13 | **Yes** | Approach A, B-spline (`03_retrained`), fixed Chebyshev degree 4, no R² gate | **0.8032** | same (Approach B with fixed degree 4 also resolves this seed, at 0.8425) |
| 14 | **Yes** | Approach A, B-spline (`03_retrained`), fixed Chebyshev degree 4, no R² gate | **0.8057** | same |

All three seeds are resolved by the **same configuration**: Approach A (the committed B-spline extraction path) with
the R²-gated minimum-degree search replaced by an unconditional degree-4 fit on every edge. No seed required the
sign flip to reach a valid metric; the flip was only used as a diagnostic for the Approach-B seed-14 case
(Discussion, item 4).

## Conclusions and recommendations

- **Adopt Approach A with fixed degree 4 (no R² gate) as the extraction default.** In practice, `_fit_edge` in
  `src/architectures/extractor.py` should no longer accept the first degree that exceeds
  `chebyshev_r2_threshold`, and should always fit at `chebyshev_max_degree`. The change is small and well isolated
  (see `FixedMaxDegreeExtractor` in `tests/test_chebyshev_min_degree_bspline.py`). The production extractor was not
  modified in this experiment.
- **Do not adopt Approach B** (symbolic / `05_final`), even with the fix. It is inconsistent across seeds with the
  same qualitative structure (multiplication-only hidden layer) and includes a confidently wrong case (seed 14).
  This confirms, without resolving, the fragility in the interaction between the post-fine-tuning symbolic fit and
  the IsingZZ-chained multiplication readout reported in `reports/02_chebyshev_degree_ceiling_comparison.md`.
- **Open issue:** why removing the R² gate fixes Approach A uniformly but not Approach B. Candidate explanations are
  the approximation error of the `symbolic_fun`/`fix_symbolic` layer compounding with the Chebyshev refit (the
  "fit of a fit" described in the extractor docstring, `extractor.py:19-28`), or a property of the pruned graph of
  seed 14. A clean seed-12 Approach-B comparison requires rerunning the classical pipeline of seed 12 once the NaN
  in symbolic fine-tuning (see Method) is fixed, to obtain a current `05_final` checkpoint.
