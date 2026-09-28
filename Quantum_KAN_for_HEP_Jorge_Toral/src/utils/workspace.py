# workspace.py
import os
import re
import json
import datetime
import numpy as np
import torch
import random
from pathlib import Path

from src.utils.hyperparams import get_hyperparams

# Tasks whose preprocessing applies an invariant-mass cut (only top-tagging today);
# for every other task the "cut" level is omitted from the directory layout.
TASKS_WITH_MASS_CUT = {"top"}

# Directory name under outputs/<task>/ (and data/processed/<task>/) holding runs whose
# regime could not be identified from the path (pre-variant-layout results).
LEGACY_DIR = "legacy"
_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
_SUBSET_RUN_RE = re.compile(r"^n(\d+)_subset(\d+)$")


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This is useful for constructing paths to data, models, and reports in a way that is independent of the current working directory.
    """
    return Path(__file__).parent.parent.parent.resolve()

# Set random seeds for reproducibility
def set_seed(seed_value=42, purpose=None):
    suffix = f" ({purpose})" if purpose else ""
    print(f"Setting global random seed {seed_value} for reproducibility{suffix}.")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_dirs(config):
    """
    Creates all necessary directories based on the provided configuration dictionary.
    """
    print("Ensuring directory structure exists...")
    for key, value in config.items():
        if isinstance(value, (str, Path)) and str(config['root']) in str(value):
            path=Path(value)
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

def resolve_variant(task, apply_mass_cut, n_subsets, seed):
    """
    Single source of truth for how a run's data regime is encoded in directory names.

    - cut: "mass_cut" / "no_mass_cut" (None for tasks without a mass cut, e.g. quark-gluon)
    - data_label: "full" when n_subsets == 1 (the entire dataset, unpartitioned), else "n{N}".
      Names the seed-independent canonical cache directory.
    - run_label: "full" when n_subsets == 1, else "n{N}_subset{seed % N}". Names the
      per-run outputs directory (the selected subset depends on the seed).
    """
    if n_subsets < 1:
        raise ValueError(f"n_subsets must be >= 1, got {n_subsets}")
    cut = None
    if task in TASKS_WITH_MASS_CUT:
        cut = "mass_cut" if apply_mass_cut else "no_mass_cut"
    subset_id = seed % n_subsets
    data_label = "full" if n_subsets == 1 else f"n{n_subsets}"
    run_label = "full" if n_subsets == 1 else f"n{n_subsets}_subset{subset_id}"
    return {
        "cut": cut,
        "n_subsets": n_subsets,
        "subset_id": subset_id,
        "data_label": data_label,
        "run_label": run_label,
        "variant": "/".join(p for p in (cut, run_label) if p),
    }


def _parse_run_parts(parts):
    """Inverse of resolve_variant's directory naming: path parts between
    outputs/<task>/ and seed_<N>/ -> (apply_mass_cut, n_subsets, subset_id, variant),
    or None if the parts do not match the variant layout (legacy/unknown)."""
    parts = list(parts)
    apply_mass_cut = None
    if parts and parts[0] in ("mass_cut", "no_mass_cut"):
        apply_mass_cut = parts.pop(0) == "mass_cut"
    if len(parts) != 1:
        return None
    label = parts[0]
    if label == "full":
        n_subsets, subset_id = 1, 0
    else:
        m = _SUBSET_RUN_RE.match(label)
        if not m:
            return None
        n_subsets, subset_id = int(m.group(1)), int(m.group(2))
    cut = None if apply_mass_cut is None else ("mass_cut" if apply_mass_cut else "no_mass_cut")
    variant = "/".join(p for p in (cut, label) if p)
    return apply_mass_cut, n_subsets, subset_id, variant


def iter_run_dirs(task):
    """
    Yields one dict per outputs/<task>/**/seed_<N>/ directory on disk:
    {path, seed, apply_mass_cut, n_subsets, subset_id, variant, legacy}.
    Variant-layout runs are tagged from their path; anything else (e.g. the
    pre-variant outputs/<task>/seed_<N>/ dirs or outputs/<task>/legacy/seed_<N>/)
    is yielded with legacy=True and variant="legacy" so it can still be collected.
    """
    task_dir = Path(get_project_root()) / "outputs" / task
    if not task_dir.is_dir():
        return
    for path in sorted(task_dir.rglob("seed_*")):
        m = _SEED_DIR_RE.match(path.name)
        if not m or not path.is_dir() or ".ipynb_checkpoints" in path.parts:
            continue
        seed = int(m.group(1))
        parsed = _parse_run_parts(path.relative_to(task_dir).parts[:-1])
        if parsed is None:
            yield {"path": path, "seed": seed, "apply_mass_cut": None, "n_subsets": None,
                   "subset_id": None, "variant": LEGACY_DIR, "legacy": True}
        else:
            apply_mass_cut, n_subsets, subset_id, variant = parsed
            yield {"path": path, "seed": seed, "apply_mass_cut": apply_mass_cut,
                   "n_subsets": n_subsets, "subset_id": subset_id, "variant": variant,
                   "legacy": False}


def write_hyperparams_snapshot(config, extra=None):
    """
    Serializes this run's resolved hyperparameters to config["hyperparams_report_path"]
    as JSON: task, seed, a timestamp, the subset of `config` matching get_hyperparams()'s
    keys (so it reflects what this config object actually resolved to), and an optional
    `extra` dict of script-identifying info (e.g. parsed CLI args) under 'run_args'.
    """
    hp_keys = get_hyperparams().keys()
    snapshot = {
        "task": config.get("task"),
        "seed": config.get("seed"),
        "variant": config.get("variant"),
        "full_dataset": config.get("full_dataset"),
        "subset_id": config.get("subset_id"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {k: config[k] for k in hp_keys if k in config},
        "run_args": extra or {},
    }
    path = Path(config["hyperparams_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================
# Define all hyperparameters and paths in one place.
# This makes it easy to modify and experiment.
def get_config(task, seed, full_dataset=False, apply_mass_cut=None, n_subsets=None):
    """
    Returns a tight configuration dictionary isolating the raw data path,
    the processed multiscale tensors, and specific KAN 2.0 / VQC output targets.

    The data regime is resolved here, once, and encoded in every path:
      - full_dataset=True forces apply_mass_cut=False and n_subsets=1 (the entire,
        unpartitioned dataset; train/val/test stay separate splits).
      - Otherwise apply_mass_cut / n_subsets fall back to hyperparams.py's values.
        Explicit apply_mass_cut / n_subsets arguments override those defaults (used
        by tooling that reconstructs the config of an already-existing run).

    Layout (see resolve_variant):
      data/processed/<task>/[<cut>/]<full|n{N}>/               canonical cache (seed-independent)
      outputs/<task>/[<cut>/]<full|n{N}_subset{k}>/seed_<seed>/   one run
      outputs/<task>/aggregate/                                 cross-run metrics table
    """
    root = get_project_root()
    hp = get_hyperparams()

    if full_dataset:
        hp["apply_mass_cut"] = False
        hp["n_subsets"] = 1
    else:
        if apply_mass_cut is not None:
            hp["apply_mass_cut"] = apply_mass_cut
        if n_subsets is not None:
            hp["n_subsets"] = n_subsets

    variant = resolve_variant(task, hp["apply_mass_cut"], hp["n_subsets"], seed)
    seed_dir = f"seed_{seed}"
    cut_parts = [variant["cut"]] if variant["cut"] else []

    # Core directories
    canonical_dir = os.path.join(root, "data", "processed", task, *cut_parts, variant["data_label"])
    outputs_dir = os.path.join(root, "outputs", task, *cut_parts, variant["run_label"], seed_dir)
    quantum_dir = os.path.join(outputs_dir, "models", "quantum_weights")

    # Task-level (variant-independent) locations: the cross-run metrics collection
    # table (Parquet, tagged by variant/seed/model) and the pipeline-wide shell logs.
    aggregate_dir = os.path.join(root, "outputs", task, "aggregate")
    pipeline_logs_dir = os.path.join(root, "outputs", task, "pipeline_logs")

    CONFIG = {
        # Base Engine Paths
        "root": root,
        "task": task,
        "seed": seed,

        # Data regime (see resolve_variant)
        "full_dataset": bool(full_dataset),
        "variant": variant["variant"],
        "subset_id": variant["subset_id"],
        "run_dir": outputs_dir,

        # Origin and Destination of Data
        "raw_data_dir": os.path.join(root, "data", "raw"),

        # Canonical (seed-independent) disjoint subset partition -- built once by
        # scripts/run_preprocessing.py / run_preprocessing_qg.py, only ever read
        # (never rebuilt) by the training scripts. One directory per regime.
        "canonical_data_dir": canonical_dir,
        "canonical_cache_file": os.path.join(canonical_dir, "preprocessed_subsets.pt"),
        "canonical_scaler_path": os.path.join(canonical_dir, "global_scaler.pkl"),

        # Cross-run metrics collection (Parquet table, task-level)
        "aggregate_dir": aggregate_dir,
        "metrics_table_path": os.path.join(aggregate_dir, "metrics_table.parquet"),
        "eval_table_path": os.path.join(aggregate_dir, "eval_table.parquet"),
        "sine_comparison_summary_path": os.path.join(aggregate_dir, "sine_vs_chebyshev_vs_random_baseline.json"),
        "pipeline_logs_dir": pipeline_logs_dir,

        # Output targets for models and reports
        "models_dir": os.path.join(outputs_dir, "models", "01_base"),
        "plots_dir": os.path.join(outputs_dir, "plots"),
        "results_dir": os.path.join(outputs_dir, "results"),
        "logs_dir": os.path.join(outputs_dir, "logs"),
        "hyperparams_report_path": os.path.join(outputs_dir, "hyperparameters.json"),

        # reports
        "base_train_history_data": os.path.join(outputs_dir, "results", "01_base", "base_train_history.json"),
        "base_eval_data_true": os.path.join(outputs_dir, "results", "01_base", "base_eval_true.npy"),
        "base_eval_data_probs": os.path.join(outputs_dir, "results", "01_base", "base_eval_probs.npy"),
        "base_eval_data_binary": os.path.join(outputs_dir, "results", "01_base", "base_eval_binary.npy"),
        "base_eval_metrics": os.path.join(outputs_dir, "results", "01_base", "base_eval_metrics.json"),

        # plots
        "base_train_loss_plot": os.path.join(outputs_dir, "plots", "01_plot_base", "base_train_loss.png"),
        "base_train_auc_plot": os.path.join(outputs_dir, "plots", "01_plot_base", "base_train_auc.png"),
        "base_eval_cm": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_cm.png"),
        "base_eval_cm_normalized": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_cm_normalized.png"),
        "base_eval_roc": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_roc.png"),
        "base_eval_pr": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_pr.png"),

        "base_model_plot_folder": os.path.join(outputs_dir, "plots", "01_plot_base", "splines"),
        "base_model_plot_save_path": os.path.join(outputs_dir, "plots", "01_plot_base", "base_model.png"),
        
        # Pruned
        "pruned_model_path": os.path.join(outputs_dir, "models", "02_pruned"),

        # Re-trained
        "retrained_model_path": os.path.join(outputs_dir, "models", "03_retrained"),
        "retrain_history_data": os.path.join(outputs_dir, "results", "03_retrained", "retrain_history.json"),
        "retrain_loss_plot": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_loss.png"),
        "retrain_auc_plot": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_auc.png"),
        "retrain_eval_cm": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_cm.png"),
        "retrain_eval_cm_normalized": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_cm_normalized.png"),
        "retrain_eval_roc": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_roc.png"),
        "retrain_eval_pr": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_pr.png"),
        "retrain_eval_data_true": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_true.npy"),
        "retrain_eval_data_probs": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_probs.npy"),
        "retrain_eval_data_binary": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_binary.npy"),
        "retrain_eval_metrics": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_metrics.json"),
        "retrained_model_plot_folder": os.path.join(outputs_dir, "plots", "03_retrained", "splines"),
        "retrained_model_plot_save_path": os.path.join(outputs_dir, "plots", "03_retrained", "retrained_model.png"),

        # symbolic simplification
        "symbolic_model_path": os.path.join(outputs_dir, "models", "04_symbolic"),
        "symbolic_model_plot_folder": os.path.join(outputs_dir, "plots", "04_symbolic", "splines"),
        "symbolic_model_plot_save_path": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_model.png"),
        "symbolic_model_eval_data": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_model_eval_data.json"),
        "symbolic_eval_cm": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_cm.png"),
        "symbolic_eval_cm_normalized": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_cm_normalized.png"),
        "symbolic_eval_roc": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_roc.png"),
        "symbolic_eval_pr": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_pr.png"),
        "symbolic_eval_data_true": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_true.npy"),
        "symbolic_model_eval_probs": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_probs.npy"),
        "symbolic_model_eval_binary": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_binary.npy"),
        "symbolic_eval_metrics": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_metrics.json"),
        "symbolic_history_data": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_history.json"),
        "symbolic_loss_plot": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_loss.png"),
        "symbolic_auc_plot": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_auc.png"),

        # final fine-tuning
        "final_model_path": os.path.join(outputs_dir, "models", "05_final"),
        "final_eval_cm": os.path.join(outputs_dir, "plots", "05_final", "final_eval_cm.png"),
        "final_eval_cm_normalized": os.path.join(outputs_dir, "plots", "05_final", "final_eval_cm_normalized.png"),
        "final_eval_roc": os.path.join(outputs_dir, "plots", "05_final", "final_eval_roc.png"),
        "final_eval_pr": os.path.join(outputs_dir, "plots", "05_final", "final_eval_pr.png"),
        "final_eval_data_true": os.path.join(outputs_dir, "results", "05_final", "final_eval_true.npy"),
        "final_eval_data_probs": os.path.join(outputs_dir, "results", "05_final", "final_eval_probs.npy"),
        "final_eval_data_binary": os.path.join(outputs_dir, "results", "05_final", "final_eval_binary.npy"),
        "final_eval_metrics": os.path.join(outputs_dir, "results", "05_final", "final_eval_metrics.json"),
        "final_formula_path": os.path.join(outputs_dir, "results", "05_final", "final_formula.txt"),
        "final_model_plot_folder": os.path.join(outputs_dir, "plots", "05_final", "splines"),
        "final_model_plot_save_path": os.path.join(outputs_dir, "plots", "05_final", "final_model.png"),
        "final_history_data": os.path.join(outputs_dir, "results", "05_final", "final_history.json"),
        "final_loss_plot": os.path.join(outputs_dir, "plots", "05_final", "final_loss.png"),
        "final_auc_plot": os.path.join(outputs_dir, "plots", "05_final", "final_auc.png"),

        # ----------------------------
        # --- Report quantum Paths ---
        # ----------------------------
        # Chebyshev coefficients txt paths
        "Chebyshev_coefficients_path": os.path.join(outputs_dir, "results", "chebyshev_coefficients.txt"),
        "circuit_plot": os.path.join(outputs_dir, "plots", "quantum-circuit.png"),
        "qkan_metadata_path": os.path.join(outputs_dir, "results", "qkan_metadata.json"),

        # Reports - QKAN evaluation - Noisy
        "roc_qkan_noisy": os.path.join(outputs_dir, "plots", "qkan","noisy", "roc_qkan_noisy.png"),
        "pr_qkan_noisy": os.path.join(outputs_dir, "plots", "qkan","noisy", "pr_qkan_noisy.png"),
        "cm_qkan_noisy": os.path.join(outputs_dir, "plots", "qkan","noisy", "cm_qkan_noisy.png"),
        "cm_qkan_noisy_normalized": os.path.join(outputs_dir, "plots", "qkan","noisy", "cm_qkan_noisy_normalized.png"),
        "metrics_qkan_noisy": os.path.join(outputs_dir, "results", "qkan","noisy", "metrics_qkan_noisy.json"),
        "qkan_eval_data_true_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "qkan_eval_true_noisy.npy"),
        "qkan_eval_data_probs_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "qkan_eval_probs_noisy.npy"),
        "qkan_eval_data_binary_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "qkan_eval_binary_noisy.npy"),
        "history_noisy_loss": os.path.join(outputs_dir, "results", "qkan","noisy", "history_loss.json"),
        "history_noisy_loss_plot": os.path.join(outputs_dir, "plots", "qkan","noisy", "history_loss.png"),
        "history_noisy_auc_plot": os.path.join(outputs_dir, "plots", "qkan","noisy", "history_auc.png"),

        # Reports - QKAN evaluation - Ideal
        "roc_qkan_ideal": os.path.join(outputs_dir, "plots", "qkan","ideal", "roc_qkan_ideal.png"),
        "pr_qkan_ideal": os.path.join(outputs_dir, "plots", "qkan","ideal", "pr_qkan_ideal.png"),
        "cm_qkan_ideal": os.path.join(outputs_dir, "plots", "qkan","ideal", "cm_qkan_ideal.png"),
        "cm_qkan_ideal_normalized": os.path.join(outputs_dir, "plots", "qkan","ideal", "cm_qkan_ideal_normalized.png"),
        "metrics_qkan_ideal": os.path.join(outputs_dir, "results", "qkan","ideal", "metrics_qkan_ideal.json"),
        "qkan_eval_data_true_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "qkan_eval_true_ideal.npy"),
        "qkan_eval_data_probs_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "qkan_eval_probs_ideal.npy"),
        "qkan_eval_data_binary_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "qkan_eval_binary_ideal.npy"),
        "history_ideal_loss": os.path.join(outputs_dir, "results", "qkan","ideal", "history_loss.json"),
        "history_ideal_loss_plot": os.path.join(outputs_dir, "plots", "qkan","ideal", "history_loss.png"),
        "history_ideal_auc_plot": os.path.join(outputs_dir, "plots", "qkan","ideal", "history_auc.png"),

        # Reports - QKAN evaluation - Shots
        "roc_qkan_shots": os.path.join(outputs_dir, "plots", "qkan","shots", "roc_qkan_shots.png"),
        "pr_qkan_shots": os.path.join(outputs_dir, "plots", "qkan","shots", "pr_qkan_shots.png"),
        "cm_qkan_shots": os.path.join(outputs_dir, "plots", "qkan","shots", "cm_qkan_shots.png"),
        "cm_qkan_shots_normalized": os.path.join(outputs_dir, "plots", "qkan","shots", "cm_qkan_shots_normalized.png"),
        "metrics_qkan_shots": os.path.join(outputs_dir, "results", "qkan","shots", "metrics_qkan_shots.json"),
        "qkan_eval_data_true_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "qkan_eval_true_shots.npy"),
        "qkan_eval_data_probs_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "qkan_eval_probs_shots.npy"),
        "qkan_eval_data_binary_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "qkan_eval_binary_shots.npy"),
        "history_shots_loss": os.path.join(outputs_dir, "results", "qkan","shots", "history_loss.json"),
        "history_shots_loss_plot": os.path.join(outputs_dir, "plots", "qkan","shots", "history_loss.png"),
        "history_shots_auc_plot": os.path.join(outputs_dir, "plots", "qkan","shots", "history_auc.png"),

        # Reports - QKAN evaluation - Baseline (untrained, warm-start only, pre-training)
        "roc_qkan_baseline_noisy": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "roc_qkan_baseline_noisy.png"),
        "pr_qkan_baseline_noisy": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "pr_qkan_baseline_noisy.png"),
        "cm_qkan_baseline_noisy": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "cm_qkan_baseline_noisy.png"),
        "cm_qkan_baseline_noisy_normalized": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "cm_qkan_baseline_noisy_normalized.png"),
        "metrics_qkan_baseline_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "baseline", "metrics_qkan_baseline_noisy.json"),
        "qkan_eval_data_true_baseline_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "baseline", "qkan_eval_true_baseline_noisy.npy"),
        "qkan_eval_data_probs_baseline_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "baseline", "qkan_eval_probs_baseline_noisy.npy"),
        "qkan_eval_data_binary_baseline_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "baseline", "qkan_eval_binary_baseline_noisy.npy"),

        "roc_qkan_baseline_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "roc_qkan_baseline_ideal.png"),
        "pr_qkan_baseline_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "pr_qkan_baseline_ideal.png"),
        "cm_qkan_baseline_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "cm_qkan_baseline_ideal.png"),
        "cm_qkan_baseline_ideal_normalized": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "cm_qkan_baseline_ideal_normalized.png"),
        "metrics_qkan_baseline_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "baseline", "metrics_qkan_baseline_ideal.json"),
        "qkan_eval_data_true_baseline_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "baseline", "qkan_eval_true_baseline_ideal.npy"),
        "qkan_eval_data_probs_baseline_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "baseline", "qkan_eval_probs_baseline_ideal.npy"),
        "qkan_eval_data_binary_baseline_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "baseline", "qkan_eval_binary_baseline_ideal.npy"),

        "roc_qkan_baseline_shots": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "roc_qkan_baseline_shots.png"),
        "pr_qkan_baseline_shots": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "pr_qkan_baseline_shots.png"),
        "cm_qkan_baseline_shots": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "cm_qkan_baseline_shots.png"),
        "cm_qkan_baseline_shots_normalized": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "cm_qkan_baseline_shots_normalized.png"),
        "metrics_qkan_baseline_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "baseline", "metrics_qkan_baseline_shots.json"),
        "qkan_eval_data_true_baseline_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "baseline", "qkan_eval_true_baseline_shots.npy"),
        "qkan_eval_data_probs_baseline_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "baseline", "qkan_eval_probs_baseline_shots.npy"),
        "qkan_eval_data_binary_baseline_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "baseline", "qkan_eval_binary_baseline_shots.npy"),

        # Reports - QKAN evaluation - Random VQC init ablation (Ideal only)
        "roc_qkan_random_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "random", "roc_qkan_random_ideal.png"),
        "pr_qkan_random_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "random", "pr_qkan_random_ideal.png"),
        "cm_qkan_random_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "random", "cm_qkan_random_ideal.png"),
        "cm_qkan_random_ideal_normalized": os.path.join(outputs_dir, "plots", "qkan", "ideal", "random", "cm_qkan_random_ideal_normalized.png"),
        "metrics_qkan_random_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "random", "metrics_qkan_random_ideal.json"),
        "qkan_eval_data_true_random_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "random", "qkan_eval_true_random_ideal.npy"),
        "qkan_eval_data_probs_random_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "random", "qkan_eval_probs_random_ideal.npy"),
        "qkan_eval_data_binary_random_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "random", "qkan_eval_binary_random_ideal.npy"),
        "history_random_ideal_loss": os.path.join(outputs_dir, "results", "qkan", "ideal", "random", "history_loss.json"),
        "history_random_ideal_loss_plot": os.path.join(outputs_dir, "plots", "qkan", "ideal", "random", "history_loss.png"),
        "history_random_ideal_auc_plot": os.path.join(outputs_dir, "plots", "qkan", "ideal", "random", "history_auc.png"),

        "init_weights": quantum_dir,

        # Quantum model and weights
        "polynomial_weights_dir": quantum_dir,
        "coef_n_path": os.path.join(quantum_dir, "w_n.npy"),
        "coef_q_path": os.path.join(quantum_dir, "w_q.npy"),
        "coef_z_path": os.path.join(quantum_dir, "w_z.npy"),
        "coef_dr_path": os.path.join(quantum_dir, "w_dr.npy"),
        "coef_out_path": os.path.join(quantum_dir, "w_out.npy"),

        "qkan_noisy_path": os.path.join(quantum_dir, "qkan_noisy.pth"),
        "qkan_ideal_path": os.path.join(quantum_dir, "qkan_ideal.pth"),
        "qkan_shots_path": os.path.join(quantum_dir, "qkan_shots.pth"),
        "qkan_random_ideal_path": os.path.join(quantum_dir, "qkan_random_ideal.pth"),

        # ----------------------------------
        # --- Random Forest baseline paths ---
        # ----------------------------------
        "rf_model_path": os.path.join(outputs_dir, "models", "rf", "rf_model.joblib"),
        "rf_eval_data_true": os.path.join(outputs_dir, "results", "rf", "rf_eval_true.npy"),
        "rf_eval_data_probs": os.path.join(outputs_dir, "results", "rf", "rf_eval_probs.npy"),
        "rf_eval_data_binary": os.path.join(outputs_dir, "results", "rf", "rf_eval_binary.npy"),
        "rf_eval_metrics": os.path.join(outputs_dir, "results", "rf", "rf_eval_metrics.json"),
        "rf_feature_importance_data": os.path.join(outputs_dir, "results", "rf", "rf_feature_importance.json"),
        "rf_eval_cm": os.path.join(outputs_dir, "plots", "rf", "rf_eval_cm.png"),
        "rf_eval_cm_normalized": os.path.join(outputs_dir, "plots", "rf", "rf_eval_cm_normalized.png"),
        "rf_eval_roc": os.path.join(outputs_dir, "plots", "rf", "rf_eval_roc.png"),
        "rf_eval_pr": os.path.join(outputs_dir, "plots", "rf", "rf_eval_pr.png"),
        "rf_feature_importance_plot": os.path.join(outputs_dir, "plots", "rf", "rf_feature_importance.png"),
    }

    # Untrained (baseline) ideal-device evals for the basis comparison: SineKAN warm
    # start ("sine_baseline") and untrained random VQC init ("baseline_random").
    # Same key scheme as the blocks above: {metric}_qkan{suffix}_ideal.
    for suffix, sub in (("_sine_baseline", "sine_baseline"), ("_baseline_random", "baseline_random")):
        plots = os.path.join(outputs_dir, "plots", "qkan", "ideal", sub)
        results = os.path.join(outputs_dir, "results", "qkan", "ideal", sub)
        name = f"qkan{suffix}_ideal"
        CONFIG.update({
            f"roc_{name}": os.path.join(plots, f"roc_{name}.png"),
            f"pr_{name}": os.path.join(plots, f"pr_{name}.png"),
            f"cm_{name}": os.path.join(plots, f"cm_{name}.png"),
            f"cm_{name}_normalized": os.path.join(plots, f"cm_{name}_normalized.png"),
            f"metrics_{name}": os.path.join(results, f"metrics_{name}.json"),
            f"qkan_eval_data_true{suffix}_ideal": os.path.join(results, f"qkan_eval_true{suffix}_ideal.npy"),
            f"qkan_eval_data_probs{suffix}_ideal": os.path.join(results, f"qkan_eval_probs{suffix}_ideal.npy"),
            f"qkan_eval_data_binary{suffix}_ideal": os.path.join(results, f"qkan_eval_binary{suffix}_ideal.npy"),
        })

    # SineKAN warm-start graph (kept apart from the Chebyshev quantum_weights.pt)
    CONFIG["quantum_graph_sine_filename"] = "quantum_weights_sine.pt"

    CONFIG.update(hp)
    return CONFIG
