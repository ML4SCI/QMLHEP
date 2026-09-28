# src/utils/hyperparams.py
# Model/training hyperparameters, isolated from workspace.py's path/output-tree logic.
# None of these currently vary by task or seed -- only paths do (see workspace.get_config).

features_globales = ['m', 'n']
features_locales = []
for i in range(1, 11):
    features_locales.append(rf'DR_{i}')
    features_locales.append(rf'pT_{i}')

TOTAL_FEATURES = features_globales + features_locales


def get_hyperparams():
    """
    Returns all model/training hyperparameters as a single flat dict.
    Merged into workspace.get_config()'s output alongside the path/output-tree keys.
    """
    return {
        # -----------------------------
        # --- Classic KAN ----
        # -----------------------------
        "features": TOTAL_FEATURES,
        "width": [22, [9, 9], 1],  # Architecture [input, hidden, output]
        "grid": 5,
        "k": 3,
        "num_workers": 0,

        # --- Base Training Hyperparameters ---
        "base_lr": 5e-3,
        "base_epochs": 60,
        "base_batch_size": 4096,
        "base_patience": 7,
        "base_early_stop_delta": 1e-2,
        "base_lamb": 0.01,  # Regularization weight
        "base_lamb_l1": 0.01,
        "base_lamb_entropy": 0.01,
        "base_lamb_coef": 0.005,
        "base_lamb_coefdiff": 0.01,
        "base_update_grid_freq": 60,

        # --- Pruning Hyperparameters ---
        "prune_input_th": 1e-2,
        "prune_node_th": 4e-2,
        "prune_edge_th": 6e-2,
        "prune_max_fanin": 2,  # hard cap on active input->hidden edges per hidden neuron, additive on top of prune_node_th/prune_edge_th

        # --- Re-training Hyperparameters ---
        "retrain_lr": 1e-3,
        "retrain_epochs": 20,
        "retrain_batch_size": 2048,
        "retrain_patience": 6,
        "retrain_early_stop_delta": 5e-5,
        "retrain_lamb_l1": 0.1,
        "retrain_lamb_entropy": 0.2,
        "retrain_lamb_coef": 0.005,
        "retrain_lamb_coefdiff": 0.01,

        # --- Simplification (Fitting) Hyperparameters ---
        "symbolic_r2_threshold": 0.85,
        "symbolic_weight_simple": 0.5,

        # --- Fine-Tuning Hyperparameters ---
        "finetune_lr": 5e-5,  # Lower learning rate
        "finetune_epochs": 15,
        "finetune_batch_size": 2048,
        "finetune_patience": 5,
        "finetune_early_stop_delta": 1e-6,

        # ---------------------
        # QKAN Hyperparameters
        # ---------------------
        "qkan_batch_size": 1024,
        "qkan_learning_rate": 5e-3,
        "qkan_epochs": 50,
        "qkan_patience": 8,
        "qkan_early_stop_delta": 5e-3,

        "n_train_samples_for_epoch": 1000,
        "n_val_samples": 2000,

        # --- Chebyshev Extraction Hyperparameters ---
        # Fixed degree every edge is fit at (no R2-gated search) -- see
        # reports/01_* to reports/04_* for why an R2-gated minimum-degree search was
        # tried and reverted.
        "chebyshev_max_degree": 4,
        "qkan_dynamic_range_threshold": 1e-3,

        # --- Statistical-replicate subset splitting ---
        # After transform + balance, each split (train/val/test) is partitioned into
        # n_subsets mutually disjoint, class-balanced chunks. A training run's --seed
        # selects which chunk it trains on via seed % n_subsets; subset_split_seed
        # seeds ONLY the one-time canonical partition itself, decoupled from --seed,
        # so the partition is stable regardless of which seed later selects a subset.
        "n_subsets": 5,
        "subset_split_seed": 37,

        # ---------------------------
        # Random Forest Hyperparameters
        # ---------------------------
        # Fixed, tuned defaults (no in-script hyperparameter search) -- reused
        # as-is by every seed/task run. random_state is intentionally NOT set
        # here: RandomForestTrainer passes the run's own --seed (config["seed"])
        # so the forest's internal randomness stays tied to the selected fold.
        "rf_n_estimators": 300,
        "rf_max_depth": 35,
        "rf_min_samples_split": 10,
        "rf_min_samples_leaf": 5,
        "rf_max_features": 0.35,
        "rf_class_weight": "balanced",
        "rf_n_jobs": -1,

        # --- Symbolic-regression warm-up sample sizing ---
        # X_train_sample is drawn as `symbolic_sample_fraction` of a subset's own
        # train chunk; if that falls under `symbolic_sample_min_floor` (a 1/15 subset
        # is much smaller than the old full-dataset pool), `symbolic_sample_fallback_size`
        # is used instead (capped to the subset's own size). See balance.resolve_sample_size.
        "symbolic_sample_fraction": 0.05,
        "symbolic_sample_min_floor": 500,
        "symbolic_sample_fallback_size": 5000,

        # --- Top-tagging invariant-mass cut ---
        # Applied in processor_top._compute_physics_features. run_preprocessing.py
        # is the only caller allowed to override apply_mass_cut via CLI; every other
        # script (train_kan.py, train_kan_top.py, train_qkan.py, ...) uses this default.
        "apply_mass_cut": True,
        "mass_cut_lo": 145.0,
        "mass_cut_hi": 205.0,
    }
