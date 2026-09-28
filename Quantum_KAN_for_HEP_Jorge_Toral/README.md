# Quantum Sine-Kolmogorov-Arnold-Networks for High Energy Physics

This repository contains the code developed for **QKAN**, a project for Google Summer of Code 2026 at [ML4SCI](https://ml4sci.org/).

The project builds a classical **Kolmogorov-Arnold Network (KAN)** for High Energy Physics (HEP) jet classification, prunes it down to a small, interpretable topology, and extracts each surviving edge into a compact basis function representation, either **Chebyshev polynomials** (default) or a fixed-frequency **sine basis** ("SineKAN", [Reinhardt et al. 2024](References.md#ref-reinhardt-2024)). That extracted graph warm-starts a **Variational Quantum Circuit (QKAN)** built with [PennyLane](https://pennylane.ai/), which is then fine-tuned and evaluated on ideal, shot-noise, and noisy quantum backends. A classical Random Forest is trained alongside as a fast, strong reference point.

A full technical summary of the method, results and conclusions is given in [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md). A report of the project is also available on the [author's web page](https://jorge-1501.github.io/en/investigacion/interdisciplinarios/qkan-jets/). The full bibliography is in [`References.md`](References.md).

---

## Background / Method

**Task.** Binary jet classification on HEP dataset:
- **Top tagging** (main pipeline): [Zenodo record 2603256](https://zenodo.org/records/2603256), with an invariant-mass cut (145-205 GeV) isolating top-quark jets from background.

Each jet is represented by 22 features: total jet mass `m`, particle multiplicity `n`, and per-particle `DR_i`/`pT_i` for the 10 leading particles.

**Data regime.** The balanced dataset is split once into **5 disjoint, class-balanced subsets**, and each seed runs the *entire* pipeline on one of them, so several seeds give independent end-to-end replicates (see [Data regimes and directory layout](#data-regimes-and-directory-layout)).

**Pipeline.**
1. **Base KAN training**. `src/architectures/classic_kan.py`, the base [pykan](https://github.com/KindXiaoming/pykan)-derived KAN trainer.
2. **HEP-specific model + pruning**. `src/architectures/hep_kan.py`. Pruning combines pykan's native attribution-threshold pruning (`node_th`/`edge_th`) with an additive hard fan-in cap (`prune_fanin`) that keeps only the top-2 highest-attribution input edges per hidden neuron.
3. **Retraining** the pruned topology.
4. **Warm-start extraction**. `src/architectures/extractor.py` (`SymbolicWarmStartExtractor`, Chebyshev basis, fit against the retrained numeric spline branch at a fixed degree) or `src/architectures/sine_basis.py` + `extractor_sine.py` (fixed-frequency SineKAN basis), each surviving edge's isolated response is fit into a small set of coefficients, producing a structured "quantum graph" of sum/multiplication nodes.
5. **Quantum circuit**. `src/architectures/qkan_model.py` (`QKANModel`): classical inputs are angle encoded via repeated RY/RZ "data re-uploading" per edge, multiplication nodes use `IsingZZ` + `CNOT`, and the prediction is a single-qubit `PauliZ` expectation. Three backends: `ideal` (`lightning.qubit`), `shots` (`default.qubit` with 1024 shots), and `noisy` (Qiskit Aer with the `FakeManilaV2` noise model).
6. **Quantum fine-tuning + evaluation**. `src/architectures/quantum_kan.py` (`QuantumKANTrainer`): Adam + `ReduceLROnPlateau`, early stopping, `BCEWithLogitsLoss`. Evaluates the model both as a warm-started baseline (pre-training) and after fine tuning, on the `ideal`, `shots` and `noisy` backends.
7. **Classical baseline**. `src/architectures/random_forest.py` (`RandomForestTrainer`), trained on all 22 features via `scripts/train_rf.py`, for a fast, strong classical comparison point.

---

## Current status / findings

- **Warm-start comparison** (seeds 10-14, untrained circuit, ideal backend): Chebyshev mean AUC **0.698 ± 0.004** vs. SineKAN **0.637 ± 0.015** vs. random init **0.488 ± 0.021**. Both real warm starts clearly beat chance; the gap is attributed to Chebyshev's much better per-edge fit quality (mean R² ≈ 0.998 vs. a mean of ≈ 0.56 and a median of ≈ 0.84 for the sine basis). See `notebooks/Analysis_of_results.ipynb` and `reports/09_warm_start_sine_vs_chebyshev_vs_random.md`.
- **Baseline-AUC regression, found and fixed**: an adaptive minimum-degree Chebyshev search (accepting the first fit degree whose R² cleared a threshold) silently collapsed baseline AUC from ~0.80 to ~0.26-0.36 once training data shrank to 1/15 of the initial full pool. Root-caused and fixed by reverting to an unconditional fixed-degree-4 fit, restoring AUC to ≈0.80-0.81 uniformly across seeds. See `reports/01_qkan_baseline_auc_regression.md` to `reports/04_chebyshev_fixed_degree_full_seed_sweep.md`.
- **Quantum-baseline collapse investigation**: of 5 hypothesized causes for a separate confusion-matrix-collapse pattern (fixed 0.5 decision threshold, warm-start bias, aggressive pruning, weak gradients, a script/regime mismatch), 4 were tested and rejected. The confirmed driver is that classical pruning collapses most runs down to only ~2 surviving input features, a structural bottleneck that can't be loosened without pushing the qubit count past what the current simulator can evaluate in reasonable time. See `reports/06_baseline_collapse_investigation.md` and `reports/08_baseline_collapse_fixes_evaluation.md`.

---

## Structure

The repository includes:
* `notebooks/`: exploratory data analysis (`EDA_top.ipynb`), a step-by-step walkthrough of the training process (`Training_process.ipynb`), and the analysis of the results over seeds 10-14 (`Analysis_of_results.ipynb`).
* `src/`: code for the project.
  * `architectures/`: model implementations.
    * `classic_kan.py`: base KAN trainer, extended by HEP-KAN.
    * `hep_kan.py`: HEP-KAN model (pykan subclass) and pruning.
    * `extractor.py`: Chebyshev warm-start extraction from the pruned classical KAN into a quantum-ready graph (`sine_basis.py`/`extractor_sine.py` for the SineKAN basis).
    * `qkan_model.py`: the PennyLane variational quantum circuit (`QKANModel`).
    * `quantum_kan.py`: quantum training/evaluation loop (`QuantumKANTrainer`).
    * `random_forest.py`: classical Random Forest baseline.
  * `preprocessing/`: `balance.py` (class balancing + `n_subsets`-way subset split, 5 by default), `processor_top.py`, `processor_qg.py`.
  * `utils/`: `hyperparams.py`, `workspace.py` (path/config resolution), `metrics.py`, `reporting.py` (aggregation of metrics and per-event evaluation data), `evaluate_qkan.py`.
* `scripts/`: CLI entry points for preprocessing, training, and evaluation (`train_kan.py` for HEP-KAN, `train_qkan.py` for warm-start extraction + quantum training/evaluation, `collect_metrics.py` and `collect_eval_data.py` for the aggregate tables; see below).
* `tests/`: pytest suite.
* `outputs/`, `data/`: generated/cached artifacts (see "Output files" below).
* `reports/`: technical reports on individual experiments and investigations.
* `PROJECT_SUMMARY.md`: technical summary of the project (data, architecture, warm-start strategies, results, conclusions).
* `References.md`: bibliography.

---

## Environment

Python **3.11 or 3.12** is required (`requires-python = ">=3.11, <3.13"`). Two equivalent ways to set up the environment are provided; use one of them.

### Option A: uv (recommended, reproducible from `uv.lock`)

```bash
pip install uv
uv sync                       # create/sync the virtual environment (.venv)
uv sync --extra cuda          # ...with the CUDA (cu121) torch build
uv sync --extra cpu           # ...or with the CPU-only torch build (mutually exclusive with cuda)
```

### Option B: classic pip + venv (from `requirements.txt`)

`requirements.txt` is a frozen snapshot (`pip freeze`) of the project virtual environment, with every version pinned. It targets the **CUDA 12.1** build of torch (`torch==2.4.1+cu121`) and already carries the PyTorch `--extra-index-url` line at the top.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a **CPU-only** machine, change the `--extra-index-url` at the top of `requirements.txt` to `https://download.pytorch.org/whl/cpu` and the torch pin to `torch==2.4.1+cpu`. `pykan` is installed from GitHub (pinned to a commit in `requirements.txt`), so `git` must be available.

To regenerate the file after changing the environment, run this with the venv activated:

```bash
pip freeze > requirements.txt      # then re-add the header lines (--extra-index-url ...)
```

---

## Download datasets
Downloads use `aria2` (installed automatically via `apt` if missing):

```bash
./download_data.sh             # top tagging only (train/val/test.h5 -> data/raw/top)
./download_data.sh --with-qg   # also the quark-gluon dataset (-> data/raw/quark-gluon)
```

The quark-gluon download is optional: the reported results only use top tagging. If a download is interrupted, run the same command again; aria2 resumes partial files and skips complete ones.

---

## Running the pipeline

### Data regimes and directory layout

Preprocessing/training can run in two regimes; each one has its own directories, so runs never overwrite each other:

* **Default (partitioned):** invariant-mass cut (145-205 GeV, top tagging) and `n_subsets` disjoint, class-balanced partitions per split (`hyperparams.py`). `--seed` selects subset `seed % n_subsets`.
* **`--full-dataset` (optional, off by default):** no mass cut and `n_subsets=1`, i.e. the entire dataset. Train/val/test stay separate splits.

```
data/processed/<task>/<cut>/<full|n{N}>/                      # canonical cache: preprocessed_subsets.pt, global_scaler.pkl
outputs/<task>/<cut>/<full|n{N}_subset{k}>/seed_<seed>/       # one run: models/ plots/ results/ logs/ hyperparameters.json
outputs/<task>/aggregate/metrics_table.parquet                # all runs, tagged by `variant`
outputs/<task>/aggregate/eval_table.parquet                   # per-event predictions + training histories of all runs
```

`<cut>` is `mass_cut` or `no_mass_cut` (top tagging only; quark-gluon has no mass cut). A run with `n_subsets=1` is labeled `full`; otherwise the directory name carries the partition count and the selected subset (e.g. `n5_subset3`). Only `scripts/run_preprocessing.py` builds the canonical cache; every other script only selects from it and raises if it doesn't exist yet.

### Top-tagging pipeline

```bash
python scripts/run_preprocessing.py [--full-dataset] --force   # build the canonical cache for a regime
python scripts/train_kan.py --seed 10 [--full-dataset]
python scripts/train_qkan.py --seed 10 [--full-dataset]        # must match the regime of the classical run
python scripts/train_rf.py --task top --seed 10 [--full-dataset]
python scripts/collect_metrics.py --task top
python scripts/collect_eval_data.py --task top
```

### Basis comparison

```bash
python scripts/eval_sine_baseline.py --seeds 10 11 12 13 14     # Chebyshev vs. SineKAN vs. random warm-start comparison
```

### Reproducing all results (seeds 10-14)

```bash
scripts/reproduce_results.sh [--full-dataset] [--force] [--foreground]
```

This runs preprocessing, then for each seed: the classical KAN, the QKAN from the KAN warm start, the random-init QKAN ablation and the Random Forest baseline. It then runs the Chebyshev/sine/random comparison and rebuilds the metrics and eval-data tables. Stages whose checkpoint already exists are skipped unless you pass `--force`. By default the script runs in the background via nohup and logs to `outputs/top/pipeline_logs/`. Override the seeds with `SEEDS="10 11" scripts/reproduce_results.sh`.

All paths are produced by `src/utils/workspace.py` (`get_config(task, seed, full_dataset=...)`); don't hardcode them.

---

## Output files

Each run writes its checkpoints (`models/`), plots, evaluation results and logs into its own run directory under `outputs/` (layout in [Data regimes and directory layout](#data-regimes-and-directory-layout)); subdirectories are created automatically. `scripts/collect_metrics.py` gathers the metrics of all runs into `outputs/<task>/aggregate/metrics_table.parquet`, and `scripts/collect_eval_data.py` gathers their per-event predictions and training histories into `outputs/<task>/aggregate/eval_table.parquet`. The author's current results are available in [this Drive folder](https://drive.google.com/drive/folders/1elF53g99h0OQvAfiQd8k2qCoYObbV1o6?usp=sharing).

---

## Testing

```bash
uv run pytest tests/ -v            # with uv
python -m pytest tests/ -v         # with an activated pip/venv environment
```

The suite covers pruning fan-in behavior, Chebyshev/sine warm-start extraction, quantum-baseline evaluation behavior, efficiency-metric computation, metrics reporting/workspace path utilities, and dataset balancing/subset splitting.

---

## License

MIT License, © 2026 Jorge Toral. See [LICENSE](LICENSE).

---

## Acknowledgements

* Kolmogorov-Arnold Networks: [pykan](https://github.com/KindXiaoming/pykan) (installed from GitHub; the project's modifications live in a subclass in `src/architectures/hep_kan.py`, and pykan itself is left unmodified).
* [ML4SCI](https://ml4sci.org/) / Google Summer of Code 2026.

Full bibliography: [`References.md`](References.md).
