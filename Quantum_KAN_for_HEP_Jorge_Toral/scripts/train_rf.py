# train_rf.py

import sys
import os
import argparse
import traceback
import time
import json

import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import src.utils.workspace as workspace
import src.preprocessing.processor_top as processor_top
import src.preprocessing.processor_qg as processor_qg
from src.architectures.random_forest import RandomForestTrainer

PROCESSORS = {"top": processor_top, "quark-gluon": processor_qg}


class TeedLog:
    """Clone sys.stdout to log messages to both the console and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(args):
    workspace.set_seed(args.seed)

    CONFIG = workspace.get_config(task=args.task, seed=args.seed, full_dataset=args.full_dataset)
    workspace.make_dirs(CONFIG)

    log_file_path = os.path.join(CONFIG["logs_dir"], f"rf_training_{args.task}_seed_{args.seed}.log")
    sys.stdout = TeedLog(log_file_path)
    workspace.write_hyperparams_snapshot(CONFIG, extra={"script": "train_rf.py", "task": args.task, "force": args.force})

    start_time = time.time()
    print(f"Starting the Random Forest baseline pipeline. Task: {args.task}. Seed: {args.seed}.")

    # ============================================================================
    # STEP 1: DATA LOADING AND PREPROCESSING (selects the seed % n_subsets fold;
    # never builds the canonical partition itself)
    # ============================================================================
    print("\n--- Step 1: Loading and Preprocessing Data ---")
    processor = PROCESSORS[args.task]
    data_dir = os.path.join(CONFIG["raw_data_dir"], args.task)
    X_train, y_train, \
    X_val, y_val, \
    X_test, y_test, \
    X_sample, scaler = processor.load_and_preprocess_data(
        data_dir=data_dir,
        task=CONFIG["task"],
        seed=args.seed,
        force_process=False,
        full_dataset=args.full_dataset
    )

    # sklearn expects numpy arrays, not torch tensors
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy().ravel()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy().ravel()

    trainer = RandomForestTrainer(CONFIG)

    # ============================================================================
    # STEP 2: TRAIN + EVALUATE
    # ============================================================================
    print("\n--- Step 2: Training Random Forest ---")
    rf_model_path = CONFIG["rf_model_path"]

    rf_trained_this_run = args.force or not os.path.exists(rf_model_path)
    if not rf_trained_this_run:
        print(f"Random Forest model found at {rf_model_path}. Skipping training.")
    else:
        trainer.train_rf_model(
            X_train=X_train_np,
            y_train=y_train_np,
            model_save_path=rf_model_path
        )

    # Evaluation always runs, even when training was skipped this run, so the
    # metrics JSON/probability arrays feeding scripts/collect_metrics.py never
    # go stale relative to a --force-free re-run (evaluate_rf_model loads the
    # checkpoint from disk itself -- only training is worth skipping).
    print("\n--- Evaluation of Random Forest Model ---")
    model, eval_data, metrics = trainer.evaluate_rf_model(
        model_save_path=rf_model_path,
        X_test=X_test_np,
        y_test=y_test_np,
        conf_matrix_save_path=CONFIG["rf_eval_cm"],
        conf_matrix_normalized_save_path=CONFIG["rf_eval_cm_normalized"],
        save_path_roc_curve=CONFIG["rf_eval_roc"],
        save_path_pr_curve=CONFIG["rf_eval_pr"]
    )

    np.save(CONFIG["rf_eval_data_true"], eval_data[0])
    np.save(CONFIG["rf_eval_data_probs"], eval_data[1])
    np.save(CONFIG["rf_eval_data_binary"], eval_data[2])

    pipeline_total_time = time.time() - start_time
    metrics['total_pipeline_time_seconds'] = pipeline_total_time

    with open(CONFIG["rf_eval_metrics"], 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n--- Feature Importance Report ---")
    trainer.feature_importance_report(
        feature_names=CONFIG["features"],
        save_path_plot=CONFIG["rf_feature_importance_plot"],
        save_path_json=CONFIG["rf_feature_importance_data"]
    )

    print(f"\nRandom Forest pipeline completed in {pipeline_total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Baseline Training Pipeline")
    parser.add_argument('--task', type=str, choices=['top', 'quark-gluon'], default='top', help='Task to train on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (selects the seed %% n_subsets fold)')
    parser.add_argument("--force", action='store_true', help='Overwrite existing model')
    parser.add_argument("--full-dataset", dest="full_dataset", action="store_true",
                        help="Use the entire dataset: forces no mass cut and n_subsets=1 (train/val/test stay separate). "
                             "Reads/writes the no_mass_cut/full variant directories. Default: off.")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
