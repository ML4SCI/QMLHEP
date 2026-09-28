# src/utils/reporting.py
# Collects the per-run/per-stage metrics JSON files that classic_kan.py /
# quantum_kan.py already write (unchanged) into one long-format table, tagged by
# task/seed/variant/subset_id/model, for later Parquet export and cross-run tabular
# extraction/plotting. Pure collection -- no statistics (mean/std, ...) computed
# here; that analysis is deliberately left for later, separate work.
import json
import os

import numpy as np
import pandas as pd

from src.utils import workspace

# model/stage name -> the get_config() key holding that stage's metrics JSON path.
# Extend this registry (not compute_run_statistics itself) to add a future
# architecture's metrics to the collected table.
METRIC_REGISTRY = {
    "classical_base": "base_eval_metrics",
    "classical_retrained": "retrain_eval_metrics",
    "classical_symbolic": "symbolic_eval_metrics",
    "classical_final": "final_eval_metrics",
    "qkan_ideal": "metrics_qkan_ideal",
    "qkan_noisy": "metrics_qkan_noisy",
    "qkan_shots": "metrics_qkan_shots",
    # Untrained ideal-device init comparison (scripts/eval_sine_baseline.py):
    # qkan_baseline_ideal (Chebyshev warm start, written by train_qkan.py) vs.
    # qkan_sine_baseline_ideal (SineKAN warm start) vs. qkan_baseline_random_ideal
    # (random N(0,1) weights). qkan_random_ideal below is the *trained* random init.
    "qkan_baseline_ideal": "metrics_qkan_baseline_ideal",
    "qkan_sine_baseline_ideal": "metrics_qkan_sine_baseline_ideal",
    "qkan_baseline_random_ideal": "metrics_qkan_baseline_random_ideal",
    "qkan_baseline_noisy": "metrics_qkan_baseline_noisy",
    "qkan_baseline_shots": "metrics_qkan_baseline_shots",
    "qkan_random_ideal": "metrics_qkan_random_ideal",
    "random_forest": "rf_eval_metrics",
}


def _eval_keys(true, probs, binary, history=None):
    return {"true": true, "probs": probs, "binary": binary, "history": history}


def _qkan_eval_keys(suffix, history=None):
    return _eval_keys(*(f"qkan_eval_data_{kind}_{suffix}" for kind in ("true", "probs", "binary")), history)


# model/stage name (same names as METRIC_REGISTRY) -> the get_config() keys holding
# that stage's saved test-set arrays and, for trained stages, its training history JSON.
EVAL_REGISTRY = {
    "classical_base": _eval_keys("base_eval_data_true", "base_eval_data_probs",
                                 "base_eval_data_binary", "base_train_history_data"),
    "classical_retrained": _eval_keys("retrain_eval_data_true", "retrain_eval_data_probs",
                                      "retrain_eval_data_binary", "retrain_history_data"),
    "classical_symbolic": _eval_keys("symbolic_eval_data_true", "symbolic_model_eval_probs",
                                     "symbolic_model_eval_binary", "symbolic_history_data"),
    "classical_final": _eval_keys("final_eval_data_true", "final_eval_data_probs",
                                  "final_eval_data_binary", "final_history_data"),
    "qkan_ideal": _qkan_eval_keys("ideal", "history_ideal_loss"),
    "qkan_noisy": _qkan_eval_keys("noisy", "history_noisy_loss"),
    "qkan_shots": _qkan_eval_keys("shots", "history_shots_loss"),
    "qkan_baseline_ideal": _qkan_eval_keys("baseline_ideal"),
    "qkan_sine_baseline_ideal": _qkan_eval_keys("sine_baseline_ideal"),
    "qkan_baseline_random_ideal": _qkan_eval_keys("baseline_random_ideal"),
    "qkan_baseline_noisy": _qkan_eval_keys("baseline_noisy"),
    "qkan_baseline_shots": _qkan_eval_keys("baseline_shots"),
    "qkan_random_ideal": _qkan_eval_keys("random_ideal", "history_random_ideal_loss"),
    "random_forest": _eval_keys("rf_eval_data_true", "rf_eval_data_probs", "rf_eval_data_binary"),
}

# Training-history curves copied per row (QKAN histories have no train_auc -> None).
HISTORY_CURVES = ("train_loss", "val_loss", "train_auc", "val_auc")


def _iter_runs(task):
    """Yields (tags, config, run_path) for every run directory under outputs/<task>/
    (see workspace.iter_run_dirs), with config rebuilt for that run's data regime."""
    for run in workspace.iter_run_dirs(task):
        seed = run["seed"]
        if run["legacy"]:
            config = workspace.get_config(task, seed)
        else:
            config = workspace.get_config(
                task, seed, apply_mass_cut=run["apply_mass_cut"], n_subsets=run["n_subsets"]
            )

        tags = {
            "task": task,
            "seed": seed,
            "variant": run["variant"],
            "apply_mass_cut": run["apply_mass_cut"],
            "n_subsets": run["n_subsets"],
            "subset_id": run["subset_id"],
        }
        yield tags, config, run["path"]


def _resolve(path, config, run_path):
    """get_config's paths are rooted at the variant it was built for; re-root onto the
    directory actually found on disk (identical except for legacy runs)."""
    if not path:
        return path
    return os.path.join(run_path, os.path.relpath(path, config["run_dir"]))


def _load_array(path):
    """Loads a saved eval .npy as a flat Python list, or None if it doesn't exist."""
    if not path or not os.path.exists(path):
        return None
    return np.load(path).ravel().tolist()


def _flatten_metrics_file(path, tags):
    """Reads one metrics JSON file if it exists and merges in the given tag
    columns (e.g. task/seed/subset_id/model). List/dict values (e.g. "Confusion
    Matrix") are JSON-serialized to strings so every cell stays a scalar. Returns
    None if the file doesn't exist -- pure pass-through, no computation."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        metrics = json.load(f)

    row = dict(tags)
    for key, value in metrics.items():
        row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return row


def compute_run_statistics(task):
    """Walks every run directory actually present under outputs/<task>/ (see
    workspace.iter_run_dirs), and for each one, for every model/stage in
    METRIC_REGISTRY, collects that stage's metrics JSON (if it exists) into one row
    tagged with task/seed/variant/apply_mass_cut/n_subsets/subset_id/model.

    Despite the name, this is a COLLECTION function, not a statistical one: it
    does not compute mean/std or any other aggregate -- it only gathers and tags
    whatever per-run metrics already exist on disk into a single long-format
    DataFrame (one row per (run, model) pair found), ready for Parquet export and
    later analysis elsewhere. Missing seeds/stages (partial sweeps) are silently
    skipped, not errors.

    The data regime tags come from each run's directory name (workspace's variant
    layout), not from the current hyperparams.py, so a full-dataset run is never
    mislabeled as one of n_subsets partitions. Pre-variant-layout runs are tagged
    variant="legacy" with apply_mass_cut/n_subsets/subset_id left empty.
    """
    rows = []
    for tags, config, run_path in _iter_runs(task):
        for model_name, config_key in METRIC_REGISTRY.items():
            path = _resolve(config.get(config_key), config, run_path)
            row = _flatten_metrics_file(path, tags={**tags, "model": model_name})
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def collect_eval_data(task):
    """Same walk as compute_run_statistics, but for each (run, model) in EVAL_REGISTRY
    collects the raw test-set arrays (y_true / y_probs / y_binary, flattened to lists)
    and the training history curves (train_loss / val_loss / train_auc / val_auc) into
    one row. A (run, model) pair is kept only if its probs .npy exists; any other
    missing array/history/curve is left as None. Pure collection -- no computation
    besides the n_events / n_epochs convenience lengths."""
    rows = []
    for tags, config, run_path in _iter_runs(task):
        for model_name, keys in EVAL_REGISTRY.items():
            probs = _load_array(_resolve(config.get(keys["probs"]), config, run_path))
            if probs is None:
                continue

            history = {}
            history_path = _resolve(config.get(keys["history"]), config, run_path)
            if history_path and os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)

            row = {**tags, "model": model_name}
            row["y_true"] = _load_array(_resolve(config.get(keys["true"]), config, run_path))
            row["y_probs"] = probs
            row["y_binary"] = _load_array(_resolve(config.get(keys["binary"]), config, run_path))
            for curve in HISTORY_CURVES:
                row[curve] = history.get(curve)
            row["n_events"] = len(probs)
            row["n_epochs"] = len(row["val_loss"]) if row["val_loss"] is not None else None
            rows.append(row)

    return pd.DataFrame(rows)
