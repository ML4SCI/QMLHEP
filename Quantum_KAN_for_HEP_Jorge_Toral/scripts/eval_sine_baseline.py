# eval_sine_baseline.py
"""
Untrained (pre-fine-tuning), ideal-device evaluation of three QKAN initializations
on the SAME pruned graph and SAME test set, per seed:

  1. sine      : SineKAN-basis warm start (SineWarmStartExtractor)
  2. chebyshev : Chebyshev warm start. NOT re-evaluated here: it is the ideal baseline
                 train_qkan.py already wrote (metrics_qkan_baseline_ideal.json), which is
                 read back for the summary.
  3. random    : Chebyshev graph structure, weights ~ N(0, 1), no training

No training happens here. Sine and random outputs land in the run's outputs dir
(outputs/top/<cut>/<run>/seed_<N>/results/qkan/ideal/{sine_baseline,baseline_random}/) and are
picked up by collect_metrics.py. Requires the run's 03_retrained checkpoint and
quantum_weights.pt (both produced by train_kan.py / train_qkan.py). All paths come from
workspace.get_config; pass --full-dataset for the full-dataset run (e.g. --seeds 42).
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
from src.architectures.extractor_sine import SineWarmStartExtractor
from src.architectures.quantum_kan import QuantumKANTrainer
import src.preprocessing.processor_top as processor_top

SUMMARY_METRICS = ("Test AUC", "Test Accuracy", "Test F1 Score",
                   "Test Precision", "Test Recall", "Confusion Matrix")


def run_seed(seed, task, full_dataset):
    CONFIG = workspace.get_config(task=task, seed=seed, full_dataset=full_dataset)
    workspace.make_dirs(CONFIG)

    _, _, _, _, X_test, y_test, _, _ = processor_top.load_and_preprocess_data(
        data_dir=os.path.join(CONFIG["raw_data_dir"], task),
        task=task, force_process=False, seed=seed, full_dataset=full_dataset,
    )
    print(f"[seed {seed}] {CONFIG['variant']}: test set {len(X_test)} samples, subset {CONFIG['subset_id']}")

    classic_ckpt = os.path.join(CONFIG["retrained_model_path"], "03_retrained")
    quantum_dir = CONFIG["polynomial_weights_dir"]
    cheb_graph = os.path.join(quantum_dir, "quantum_weights.pt")
    sine_graph = os.path.join(quantum_dir, CONFIG["quantum_graph_sine_filename"])
    if not os.path.exists(cheb_graph):
        raise FileNotFoundError(f"{cheb_graph} not found; run train_qkan.py for seed {seed} first.")

    SineWarmStartExtractor(CONFIG).extract_and_save(
        classic_ckpt, sine_graph, os.path.join(CONFIG["results_dir"], "sine_coefficients.txt"))

    out = {}

    trainer = QuantumKANTrainer(CONFIG, train_backend="ideal", graph_filename=CONFIG["quantum_graph_sine_filename"])
    out["sine"] = trainer.evaluate_baseline(X_test, y_test, eval_backend="ideal", sine=True)

    with open(CONFIG["metrics_qkan_baseline_ideal"], "r") as f:
        out["chebyshev"] = json.load(f)

    workspace.set_seed(seed, purpose="untrained random VQC init")
    trainer = QuantumKANTrainer(CONFIG, train_backend="ideal", random_init=True)
    out["random"] = trainer.evaluate_baseline(X_test, y_test, eval_backend="ideal", random_init=True)
    return out, CONFIG


def main(args):
    all_out = {}
    config = None
    for seed in args.seeds:
        all_out[seed], config = run_seed(seed, args.task, args.full_dataset)

    summary = {s: {k: {m: v[m] for m in SUMMARY_METRICS} for k, v in d.items()}
               for s, d in all_out.items()}
    summary_path = args.summary or config["sine_comparison_summary_path"]
    if args.full_dataset and not args.summary:
        summary_path = summary_path.replace(".json", "_full.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Untrained ideal-device QKAN baseline: sine vs chebyshev vs random.")
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    p.add_argument("--task", default="top")
    p.add_argument("--full-dataset", dest="full_dataset", action="store_true",
                   help="Evaluate the full-dataset run (no mass cut, n_subsets=1), as in train_qkan.py.")
    p.add_argument("--summary", default=None,
                   help="Override the summary JSON path (default: get_config['sine_comparison_summary_path']).")
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
