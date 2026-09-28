import sys
from pathlib import Path
import os
import argparse
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.workspace import get_config
from src.preprocessing.processor_top import load_and_preprocess_data

def main(task="top", seed=42, apply_mass_cut=None, force_process=False, balance=True, full_dataset=False):
    # Load global configuration (task and seed); full_dataset forces no mass cut + n_subsets=1
    config = get_config(task=task, seed=seed, full_dataset=full_dataset, apply_mass_cut=apply_mass_cut)
    print(f"Preprocessing regime: {config['variant']} "
          f"(apply_mass_cut={config['apply_mass_cut']}, n_subsets={config['n_subsets']}) "
          f"-> {config['canonical_data_dir']}")
    top_path = os.path.join(config["raw_data_dir"], "top")

    # Run the pipeline
    # This automatically checks the cache; if it doesn't exist, it processes and saves
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = load_and_preprocess_data(
        data_dir=top_path,
        task=task,
        force_process=force_process,
        apply_mass_cut=apply_mass_cut,
        seed=seed,
        balance=balance,
        full_dataset=full_dataset,
    )

    print("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-tagging raw-to-canonical preprocessing pipeline.")
    parser.add_argument("--task", type=str, default="top", help="Task name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mass-cut",
        dest="apply_mass_cut",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply the invariant-mass cut (bounds from hyperparams.py's "
            "mass_cut_lo/mass_cut_hi). Default: hyperparams.py's apply_mass_cut "
            "value. Pass --no-mass-cut to use all data, no mass cut."
        ),
    )
    parser.add_argument("--force",
        dest="force_process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force reprocessing of the data, ignoring any cached results."
    )
    parser.add_argument(
        "--balance",
        dest="balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance the classes in the dataset."
    )
    parser.add_argument(
        "--full-dataset",
        dest="full_dataset",
        action="store_true",
        help=(
            "Use the entire dataset: forces no invariant-mass cut and n_subsets=1 "
            "(train/val/test stay separate splits). Takes precedence over --mass-cut. "
            "Cached under data/processed/<task>/no_mass_cut/full/. Default: off."
        ),
    )
    args = parser.parse_args()

    main(task=args.task, seed=args.seed, apply_mass_cut=args.apply_mass_cut,
        force_process=args.force_process, balance=args.balance,
        full_dataset=args.full_dataset)
