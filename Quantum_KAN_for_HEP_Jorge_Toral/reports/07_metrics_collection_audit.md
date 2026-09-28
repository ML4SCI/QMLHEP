# Metrics Collection Audit

**Date:** 2026-09-16
**Scope:** `src/utils/reporting.py::compute_run_statistics` (the function behind `scripts/collect_metrics.py`) and
every upstream training and evaluation script that feeds it.
**Status:** complete. All gaps identified below were resolved. Script names refer to the code at the time of writing.

## Summary

Before this audit, the collection was incomplete. The read loop itself (`compute_run_statistics`) was correct with
respect to its registry (`METRIC_REGISTRY`): it iterates over every entry and performs collection only (no mean or
standard deviation), as intended. The gaps were all **upstream** of the loop: two registry entries were never
populated, the metrics and probability files of three architectures could become stale when a checkpoint already
existed, and the collector was never invoked automatically. All four gaps described below were resolved.

## Gaps found

### 1. Unpopulated registry entries: `qkan_shots` / `qkan_baseline_shots`

`METRIC_REGISTRY` (`src/utils/reporting.py`) maps `qkan_shots`/`qkan_baseline_shots` to `metrics_qkan_shots.json` /
`metrics_qkan_baseline_shots.json`. However, the two evaluation loops of `scripts/train_qkan.py` were hardcoded to
`("ideal", "noisy")`. The `"shots"` backend (the third `backend_mode` of `QKANModel`, fully implemented in
`qkan_model.py`) was never evaluated, so these files were never written and both rows were empty in every collected
table.

**Resolution.** The `shots` backend is evaluated before and after training, together with `ideal` and `noisy`. A
`shots` evaluation completes (about 1.5 s for a 200-row slice at seed 12, and within the full pipeline run on the
complete test set) and writes both metrics files, so the collected table contains non-empty rows for both entries.

### 2. Stale metrics after a checkpoint-exists skip

`classical_base`/`classical_retrained`/`classical_symbolic` (`train_kan.py`, `train_kan_qg.py`, `train_kan_top.py`)
and `random_forest` (`train_rf.py`) wrote their metrics JSON and probability arrays only **inside** the
`not os.path.exists(checkpoint) and not args.force` branch. On a rerun without `--force` with an existing
checkpoint, evaluation was skipped entirely, and the collector read whatever JSON was on disk, with no indication
that it could be stale relative to newer code (e.g. the new efficiency-metric fields) or hyperparameters.

**Resolution.** Training remains skip-if-checkpoint-exists, but evaluation always runs, so the metrics JSON and
probability arrays always reflect the current code. Training-specific outputs (history JSON, loss/AUC plots, spline
plots) are still produced only when training actually runs.

**Verification.** `train_rf.py --seed 12` and `train_kan.py --seed 12` were rerun without `--force`. Both skipped
every training stage ("Random Forest model found... Skipping training.", "Skipping retraining. Model found", etc.)
but re-evaluated every stage (the full classical KAN pipeline finished in 3.58 s with all four stages
re-evaluated), and both metrics JSON files contain the new signal-efficiency and background-rejection fields
without a `--force` retrain.

### 3. Collector never run automatically

`scripts/collect_metrics.py` had no callers in the repository: neither `scripts/run_seeds.sh`,
`scripts/run_all_classic.sh`, nor any training script invoked it. This was confirmed empirically: the
`outputs/top/aggregate/metrics_table.parquet` table contained a single `random_forest` row although 5
`rf_eval_metrics.json` files existed on disk (seeds 10–14), because the collector had not been rerun after most of
those seeds finished.

**Resolution.** The collector runs automatically at the end of every seed sweep, so the aggregate Parquet table is
always refreshed after new runs.

### 4. Related data-integrity bug: symbolic-stage key mismatch

The symbolic stage of `train_kan_qg.py` and `train_kan_top.py` saved the true labels with
`np.save(CONFIG["symbolic_model_eval_data"], ...)`, where that config key (`workspace.py`) ends in `.json`. Since
`np.save` appends `.npy` to a file name that does not already end in it, the file written was
`symbolic_model_eval_data.json.npy`, not the file referenced by `symbolic_eval_data_true` (the key used correctly
by `train_kan.py`). This did not affect the metrics-JSON collector, but the true-label array of the symbolic stage
of the quark-gluon and legacy top-tagging scripts was not reachable under its intended name.

**Resolution.** All training scripts store the symbolic-stage arrays under the same `symbolic_eval_data_true` /
`symbolic_model_eval_probs` / `symbolic_model_eval_binary` keys.

## Intentional exclusions

Training histories (`base_train_history_data`, etc.), `rf_feature_importance_data`, `hyperparams_report_path` and
the pruning metadata are deliberately excluded from `METRIC_REGISTRY`. The registry is limited to
`*_eval_metrics`-type JSON files that are directly comparable across model types, not every numeric artifact in the
output tree. These exclusions remain correct.

## Verification

1. `train_rf.py --task top --seed 12` (without `--force`): training skipped, evaluation run,
   `rf_eval_metrics.json` refreshed with the new fields.
2. `train_kan.py --seed 12` (without `--force`): all four stages (base, retrained, symbolic, final) skipped training
   but were re-evaluated; the pipeline finished in 3.58 s.
3. `QuantumKANTrainer.evaluate(..., eval_backend="shots")` and `eval_backend="noisy"`: both exercised directly (no
   error, correct metrics and probability files written), independently of the full `train_qkan.py --seed 12` run
   used in `reports/08_baseline_collapse_fixes_evaluation.md`.
4. Full test suite (`pytest tests/`, excluding the two non-pytest benchmark scripts): 33 passed, 2 skipped, and one
   pre-existing failure
   (`test_baseline_quantum_collapse.py::test_baseline_metrics_show_systematic_positive_bias`). The failure was
   reproduced identically before the resolution of the gaps and is caused by drift in the pinned baseline recall of
   seed 42, not by the audit.
