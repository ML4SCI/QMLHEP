# Test Suite: Structure, Coverage and Current Status

**Scope:** every file in `tests/`. All of them are pytest unit tests.
**Status:** current as of 2026-09-22 (after commit `0b7c53b`). The results are in Section 5.

## 1. Purpose and scope

The suite checks the parts of the classical → quantum KAN pipeline that can be tested without a full training run:
- subset balancing and splitting;
- output-path resolution;
- classical KAN training, pruning and plotting;
- the call contract between the training scripts and `ClassicKANTrainer`;
- Chebyshev and sine warm-start extraction;
- routing of QKAN evaluation outputs;
- metrics computation and collection.

The tests use synthetic tensors, very small models or duck-typed fakes. They need no GPU, no raw dataset, no
`outputs/` artifacts and no quantum-circuit training. Circuit execution is limited to single forward passes on a few
qubits.

The folder holds no standalone experiment or benchmark scripts. The scripts behind `reports/` were removed and remain
in git history at commit `0b7c53b`.

## 2. Running the suite

```bash
python -m pytest tests
# or, inside the uv environment
uv run pytest tests
```

`conftest.py` puts the repository root on `sys.path`. This matches the `sys.path.append(... parent.parent)`
pattern used by every entry point in `scripts/`, so `src` can be imported as a top-level package.

### Which tests to run for a change

| If you change | Run |
|---|---|
| `src/preprocessing/balance.py` | `test_balance_subsets.py` |
| `src/utils/workspace.py` (paths, variants) | `test_variant_layout.py`, `test_reporting.py` |
| `src/architectures/classic_kan.py` | `test_classic_kan_trainer_device.py`, `test_trainer_call_contract.py` |
| `src/architectures/hep_kan.py` | `test_hep_kan_plot.py`, `test_prune_fanin.py`, `test_extractor_numeric_branch.py` |
| `src/utils/hyperparams.py` (`features`, `width`) | `test_hep_kan_plot.py` |
| `scripts/train_kan.py`, `scripts/train_rf.py`, `processor_*.py` seeding | `test_trainer_call_contract.py` |
| `src/architectures/extractor.py` | `test_extractor_numeric_branch.py` |
| `sine_basis.py`, `extractor_sine.py`, `qkan_model.py` | `test_extractor_sine.py` |
| `src/architectures/quantum_kan.py` (evaluation) | `test_qkan_baseline_eval.py` |
| `src/utils/metrics.py` | `test_metrics_efficiency.py` |
| `src/utils/reporting.py`, `scripts/collect_metrics.py` | `test_reporting.py` |

The whole suite takes about 10 s, so running all of it before a commit is cheap.

## 3. Unit tests by component

### 3.1 Preprocessing

**`test_balance_subsets.py`** (7 tests) covers `src/preprocessing/balance.py`, using synthetic tensors only:
- `balance_classes` undersamples every class to the minority count;
- `split_into_subsets` produces disjoint subsets that together cover every input row, including when the row count
  is not divisible by the number of subsets, and each subset stays class-balanced;
- `resolve_sample_size` uses the fractional sample above the floor, falls back to the fixed size below it, and never
  exceeds the pool size.

### 3.2 Workspace and output layout

**`test_variant_layout.py`** (13 tests) covers `workspace.resolve_variant`, `workspace.get_config(full_dataset=...)`
and `workspace.iter_run_dirs`, the single place where a run's data regime (mass cut, number of subsets, subset
index) is encoded in directory names. It checks:
- the variant labels;
- that `full_dataset=True` forces no mass cut and a single subset;
- that the quantum weights are stored under the run directory;
- that the full-dataset, partitioned and no-mass-cut regimes never share an output path, except the aggregate keys
  listed in `SHARED_KEYS` (`aggregate_dir`, `metrics_table_path`, `sine_comparison_summary_path`, ...);
- that `get_config` defines every key requested by the quantum evaluation for each combination of baseline,
  random-init and backend;
- that `iter_run_dirs` parses the variant layout and flags legacy runs.

The tests involve only path logic, with no data or models. A new `get_config` path key that is meant to be shared
by every regime has to be added to `SHARED_KEYS`.

### 3.3 Classical KAN

- **`test_classic_kan_trainer_device.py`** (2 tests): `ClassicKANTrainer` selects the expected device (derived from
  `torch.cuda.is_available()`) and keeps the model on that device during `train_kan_model`. It runs the real code
  path on a tiny synthetic model and dataset, so it passes on both CPU-only and CUDA machines.
- **`test_hep_kan_plot.py`** (6 tests): input labelling of `HEPKAN.plot` when the number of feature names differs
  from the input width. It plots a model with more inputs than the configured `features` list, and checks that
  `_resolve_in_vars` pads missing names, truncates extra names, selects the surviving input ids of a pruned model
  and ignores stale ids. It also checks that `width[0]` in `get_hyperparams()` matches the length of `features`.
- **`test_prune_fanin.py`** (2 tests): `hep_kan.prune_fanin` on a deterministic duck-typed model with hand-picked
  attribution scores. Capped hidden neurons keep exactly the two highest-scoring input edges (not the first by
  index or a random subset). Neurons already at or below the cap are untouched, and other layers are never modified.
- **`test_trainer_call_contract.py`** (4 tests): a static (AST) check of `scripts/train_kan.py` and
  `scripts/train_rf.py`. It checks that `train_kan.py` creates exactly one `ClassicKANTrainer` and calls
  `prune_and_save_kan` only with keyword arguments that exist in its signature. For both scripts, it checks that
  the runtime RNG and subset selection use the run seed (`args.seed`), and that `processor_top.py` and
  `processor_qg.py` seed the canonical split with `subset_split_seed`. A sanity test makes sure the reflected set
  of stage methods is not empty.

### 3.4 Warm-start extraction

- **`test_extractor_numeric_branch.py`** (2 tests): `SymbolicWarmStartExtractor._evaluate_isolated_edges` on a
  real, very small `HEPKAN` (`width=[3,[1,1],1]`). This width keeps a multiplication node, so
  `width_out[1] != width_in[1]`. The tests check that the isolated-edge evaluation reads the numeric spline branch
  (`act_fun`) and returns a curve with non-zero dynamic range, applying `subnode_bias`/`subnode_scale` without a
  shape error. They also check that the symbolic branch is still the all-zero placeholder on a fresh model.
- **`test_extractor_sine.py`** (4 tests): `src/architectures/sine_basis.py` and the sine branch of `QKANModel`. It
  checks the shapes of the sine grid and the first-layer frequencies, that `sinefit` recovers a known sine edge, a
  forward pass of a sine-basis QKAN, and that the default basis is Chebyshev.

### 3.5 Quantum evaluation

**`test_qkan_baseline_eval.py`** (4 tests) covers `QuantumKANTrainer.evaluate` and `evaluate_baseline`. The trainer
is built without `__init__` and given a lightweight fake model, and `pennylane.QNode` is monkeypatched, so no
circuit is simulated. The tests check that:
- baseline evaluations write to the `*_baseline_{backend}` configuration keys, and post-training evaluations to the
  plain keys;
- `evaluate_baseline` restores the training backend after evaluating on another one;
- random-initialization evaluations write to the `*_random` keys and save the raw evaluation arrays.

### 3.6 Metrics and collection

- **`test_metrics_efficiency.py`** (3 tests): `src/utils/metrics.compute_efficiency_metrics` on hand-constructed ROC
  curves whose values are known exactly. With perfect separation, the background efficiency is zero at every
  working point. A known interleaved-score case gives the exact background efficiency and rejection at two working
  points. The function returns only flat scalar values.
- **`test_reporting.py`** (4 tests): `src/utils/reporting.py`. `_flatten_metrics_file` is tested on a real temporary
  JSON file: it returns `None` for a missing path, merges the tag columns, and serializes nested lists and
  dictionaries. `compute_run_statistics` is tested on a synthetic output tree built from the path keys of
  `workspace.get_config`, with some stages present and others missing. The tests check the row count, the tagging
  by model, seed and subset, the tagging of full-dataset and legacy runs, and that no aggregate (mean or standard
  deviation) column is produced.

## 4. Coverage gaps

These parts of the pipeline have no test:
- the raw-data loaders in `processor_top.py` and `processor_qg.py` (they need the downloaded datasets);
- the circuit construction in `QKANModel` for Chebyshev graphs, beyond the sine forward pass;
- `scripts/train_qkan.py`, `scripts/eval_sine_baseline.py` and `scripts/reproduce_results.sh` end to end;
- `src/utils/evaluate_qkan.py`, which is stale (see `CLAUDE.md`, "Known inconsistency").

## 5. Current results

**Environment:** Python 3.11.16, PyTorch 2.4.1+cu121 (CUDA available), PennyLane 0.45.1, pytest 9.1.1.
**Command:** `python -m pytest tests`.

| Outcome | Count |
|---|---:|
| Collected | 51 |
| Passed | 51 |
| Failed | 0 |
| Skipped | 0 |
| Wall-clock time | about 11 s |

**Warnings (4):** `sklearn` `UndefinedMetricWarning` in `test_qkan_baseline_eval.py`. The fake model predicts a
constant class, so precision is undefined. This is expected for the fake model and does not affect the assertions.
