# train_qkan.py
import argparse
import traceback
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
from src.architectures.extractor import SymbolicWarmStartExtractor
from src.architectures.quantum_kan import QuantumKANTrainer
import src.preprocessing.processor_top as processor_top
import src.preprocessing.processor_qg as processor_qg

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
    # Configuration and workspace setup
    CONFIG = workspace.get_config(task=args.task, seed=args.seed, full_dataset=args.full_dataset)
    processor = PROCESSORS[args.task]
    workspace.make_dirs(CONFIG)
    print(f"Selected backend mode (Training): {args.train_backend} \n")

    log_suffix = "random_init" if args.random_init else args.train_backend
    log_file_path = os.path.join(CONFIG["logs_dir"], f"train_qkan_{args.task}_seed_{args.seed}_{log_suffix}.log")
    sys.stdout = TeedLog(log_file_path)
    workspace.write_hyperparams_snapshot(CONFIG, extra={
        "script": "train_qkan.py",
        "task": args.task,
        "train_backend": args.train_backend,
        "random_init": args.random_init,
        "force": args.force,
    })

    if args.random_init:
        workspace.set_seed(args.seed, purpose="random VQC init")

    # Load classical data
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = processor.load_and_preprocess_data(
        data_dir=os.path.join(CONFIG["raw_data_dir"], args.task),
        task=args.task,
        force_process=False,
        seed=args.seed,
        full_dataset=args.full_dataset
    )

    # Automatic Extraction (Agnostic Warm-Start)
    extractor = SymbolicWarmStartExtractor(CONFIG)
    
    # Extraction now reads the post-retrain, pre-symbolic-fit checkpoint
    # fitting Chebyshev polynomials against the numeric spline
    classic_model_path = os.path.join(CONFIG['retrained_model_path'], "03_retrained")
    output_weights_path = os.path.join(CONFIG["polynomial_weights_dir"], "quantum_weights.pt")
    report_path = CONFIG.get("Chebyshev_coefficients_path", os.path.join(CONFIG["results_dir"], "chebyshev_report.txt"))
    
    # If forced or not existing, extract classical model weights
    if args.force or not os.path.exists(output_weights_path):
        extractor.extract_and_save(classic_model_path, output_weights_path, report_path)
    else:
        print(f"Initial Weights exist in {output_weights_path}. Skipping extraction.")

    # Quantum Initialization and Training
    train_backend = "ideal" if args.random_init else args.train_backend
    q_trainer = QuantumKANTrainer(CONFIG, train_backend=train_backend, random_init=args.random_init)

    # Plot the dynamically generated circuit before training
    q_trainer.model.plot_circuit(CONFIG.get("circuit_plot", os.path.join(CONFIG["plots_dir"], "quantum-circuit.png")))

    if args.random_init:
        # Random-VQC-init ablation: no classical warm start to baseline
        # against, and ideal backend only.
        history = q_trainer.fit(X_train, y_train, X_val, y_val, resume=True, force=args.force)
        q_trainer.evaluate(X_test, y_test, eval_backend="ideal", random_init=True)
        return

    # Baseline evaluation: warm-started QKAN before any quantum fine-tuning, on
    # all three simulation backends, to measure how much predictive signal
    # survives the classical->quantum extraction alone.
    for backend in ("ideal", "noisy", "shots"):
        q_trainer.evaluate_baseline(X_test, y_test, eval_backend=backend)

    # Quantum optimization loop (trains on args.train_backend, e.g. 'ideal')
    history = q_trainer.fit(X_train, y_train, X_val, y_val, resume=True, force=args.force)

    # Final evaluation AFTER training, on all three backends, so the
    # baseline/final x ideal/noisy/shots grid can be compared directly.
    for backend in ("ideal", "noisy", "shots"):
        q_trainer.evaluate(X_test, y_test, eval_backend=backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QKAN with architecture inferred from the classical model.")
    parser.add_argument('--seed', type=int, default=42, help='Global seed')
    parser.add_argument('--train_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='ideal')
    # evaluation always runs on both 'ideal' and 'noisy' backends, before and after training.
    parser.add_argument('--force', action='store_true', help='Force extraction and retraining')
    parser.add_argument('--task', type=str, choices=['top', 'quark-gluon'], default='top')
    parser.add_argument("--full-dataset", dest="full_dataset", action="store_true",
                        help="Use the entire dataset: forces no mass cut and n_subsets=1 (train/val/test stay separate). "
                             "Must match the regime the classical KAN was trained under. Default: off.")
    parser.add_argument('--random_init', action='store_true',
                         help='Initialize the VQC with random weights instead of the '
                              'KAN-extracted warm start. Ideal backend only.')
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)