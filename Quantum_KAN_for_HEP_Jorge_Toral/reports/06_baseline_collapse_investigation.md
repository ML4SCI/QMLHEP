# Investigation of the Baseline QKAN Confusion-Matrix Collapse

**Scope:** confusion matrices and threshold-dependent metrics of the warm-started (baseline) and trained QKAN,
top-tagging task.
**Status:** historical. The five candidate causes listed here were tested experimentally in
`reports/08_baseline_collapse_fixes_evaluation.md`. Line numbers, the branch name and the script
names (e.g. `scripts/run_seeds.sh`) refer to the code at the time of writing.

## 1. Summary

The warm-started **baseline** QKAN (the circuit evaluated in `evaluate_baseline()`, before any quantum fine-tuning)
predicts a single class for essentially every test sample in several seeds. For example, seed 3 (ideal and noisy)
and seed 10 (ideal) produce confusion matrices of the form `[[0, N], [0, N]]`, i.e. every sample predicted positive.
Seeds that do not collapse completely (11, 12, 13) show a bias in the same direction: recall at 0.95–1.0 with
precision around 0.50–0.55, meaning almost every prediction is positive regardless of the true label. Quantum
fine-tuning improves AUC by a few points but does **not** remove the pattern; the confusion matrices of the trained
models are similar (recall ≈ 0.94–1.0, precision ≈ 0.50–0.55).

AUC is consistently **0.63–0.75** across all seeds, backends and stages, clearly above 0.5. The circuit carries real
separating signal, but the scores are not thresholded correctly. The collapse therefore appears to be a
**calibration or bias problem of the decision boundary**, not an absence of learning, although a weak, nearly
one-dimensional signal (Section 3.5) probably makes the miscalibration easy to reach and hard for training to
correct.

## 2. Evidence: per-seed metrics

Values read from `outputs/top/seed_*/results/qkan/{ideal,noisy}/{baseline/,}metrics_qkan*.json`.

| seed | backend | stage    | confusion matrix `[[TN,FP],[FN,TP]]` | acc    | prec   | recall | AUC    | F1     |
|-----:|--------:|----------|---------------------------------------|-------:|-------:|-------:|-------:|-------:|
| 2    | noisy   | trained  | `[[874,2247],[72,5553]]`               | 0.735  | 0.712  | 0.987  | 0.898  | 0.827  |
| 3    | ideal   | baseline | `[[0,2367],[0,2367]]`                  | 0.500  | 0.500  | 1.000  | 0.674  | 0.667  |
| 3    | noisy   | baseline | `[[0,2367],[0,2367]]`                  | 0.500  | 0.500  | 1.000  | 0.626  | 0.667  |
| 10   | ideal   | baseline | `[[0,4733],[0,4733]]`                  | 0.500  | 0.500  | 1.000  | 0.700  | 0.667  |
| 10   | ideal   | trained  | `[[115,4618],[10,4723]]`               | 0.511  | 0.506  | 0.998  | 0.746  | 0.671  |
| 10   | noisy   | baseline | `[[142,4591],[20,4713]]`               | 0.513  | 0.507  | 0.996  | 0.698  | 0.672  |
| 10   | noisy   | trained  | `[[729,4004],[198,4535]]`              | 0.556  | 0.531  | 0.958  | 0.739  | 0.683  |
| 11   | ideal   | baseline | `[[863,3870],[224,4509]]`              | 0.568  | 0.538  | 0.953  | 0.699  | 0.688  |
| 11   | ideal   | trained  | `[[885,3848],[196,4537]]`              | 0.573  | 0.541  | 0.959  | 0.736  | 0.692  |
| 11   | noisy   | baseline | `[[754,3979],[271,4462]]`              | 0.551  | 0.529  | 0.943  | 0.693  | 0.677  |
| 11   | noisy   | trained  | `[[1003,3730],[276,4457]]`             | 0.577  | 0.544  | 0.942  | 0.731  | 0.690  |
| 12   | ideal   | baseline | `[[847,3886],[234,4499]]`              | 0.565  | 0.537  | 0.951  | 0.699  | 0.686  |
| 12   | ideal   | trained  | `[[935,3798],[182,4551]]`              | 0.580  | 0.545  | 0.962  | 0.741  | 0.696  |
| 12   | noisy   | baseline | `[[848,3885],[284,4449]]`              | 0.560  | 0.534  | 0.940  | 0.693  | 0.681  |
| 12   | noisy   | trained  | `[[1109,3624],[284,4449]]`             | 0.587  | 0.551  | 0.940  | 0.736  | 0.695  |
| 13   | ideal   | baseline | `[[345,4388],[36,4697]]`               | 0.533  | 0.517  | 0.992  | 0.701  | 0.680  |

(The only saved QKAN metrics file for seed 2 is a post-training noisy run; seeds 5, 14 and 42 had no saved QKAN
metrics at the time.)

Every row has recall ≥ 0.94 and precision ≤ 0.71: the model is biased toward the positive class in every
seed, backend and stage sampled. The pattern is systematic.

## 3. Pipeline walkthrough

### 3.1 Model and circuit: `src/architectures/qkan_model.py`

- `QKANModel.__init__` (lines 40–133) loads a pruned, warm-started graph (`torch.load(graph_path...)`) and
  initializes `edge_weights`/`output_weights` directly from the Chebyshev coefficients extracted from the classical
  KAN (`SymbolicWarmStartExtractor`, Section 3.2), not randomly.
- Angle encoding, `_qkan_edge` (lines 158–167): `theta = acos(clamp(x, -0.9999, 0.9999))`, followed by `degree`
  (= 4) repetitions of `RY(weight_i); RZ(theta)` and a final `RY(weight[degree])`. Several edges writing to the same
  wire accumulate additively (RZ rotations about the same axis compose), by design (class docstring, lines 20–22).
- Circuit (`_circuit`, lines 169–194): stage 1 writes all input edges onto their accumulator wires and applies
  `IsingZZ`/`CNOT` chains for multiplication nodes; stage 2 entangles every surviving hidden wire into a single
  `_output_wire` through further `IsingZZ`/`CNOT` gates. The measurement is
  `qml.expval(qml.PauliZ(self._output_wire))`, a **single-qubit expectation value bounded to [-1, 1]**.
- `forward()` (lines 196–204) applies no bias or scale after the QNode call; the bounded expectation value is used
  directly as the logit.

### 3.2 Warm start: `src/architectures/extractor.py`

- `_evaluate_isolated_edges` (lines 60–105) fits each edge in isolation by comparing the numeric (spline) output of
  the classical layer for a variable input (`y_var`) with the same layer evaluated at an all-zero input (`y_zero`),
  and returns `y_var - y_zero + y_zero/in_dim` (line 105). Each of the `in_dim` edges feeding a node thus carries an
  equal share `y_zero/in_dim` of the node's constant term, in addition to its own variable response. This correctly
  reconstructs the total offset of the node when the edges are summed on the same wire, but it also means that
  **any constant offset of the classical layer at zero input (`subnode_bias`, lines 93–94) is included in every
  edge's Chebyshev fit (`_fit_edge`, lines 107–144), and therefore in the initial `edge_weights`** used before any
  quantum training.
- This is the most direct mechanistic candidate for the collapse of the **baseline** circuit to one class in seeds 3
  and 10: if the zero-input offset of the classical model is systematically positive at the hidden layer, the
  warm-started circuit inherits a constant positive bias on its `PauliZ` readout before the first training step.

### 3.3 Training and evaluation: `src/architectures/quantum_kan.py`

- `QuantumKANTrainer.__init__` (lines 17–34): `criterion = nn.BCEWithLogitsLoss()` (line 29, no `pos_weight`),
  `optimizer = Adam(lr=config.get("qkan_learning_rate", 5e-3))` (line 30),
  `ReduceLROnPlateau(factor=0.5, patience=6, threshold=1e-3)` (lines 31–33).
- `fit()` (lines 47–147): resamples only `n_train_samples_for_epoch=1000` training rows per epoch
  (`src/utils/hyperparams.py:79`), batch size `qkan_batch_size=1024` (`hyperparams.py:73`), gradient clipping
  `max_norm=1.0`, early stopping with `qkan_patience=8` / `qkan_early_stop_delta=5e-3` (`hyperparams.py:76-77`).
- `evaluate()` (lines 149–259): applies `sigmoid` to the `[-1, 1]`-bounded logit only at evaluation time (line 194)
  and thresholds at a **hardcoded** `test_preds_binary = (test_probs > 0.5).astype(int)` (line 207), without
  calibration on the validation set. Because the logit is bounded to `[-1, 1]`, the sigmoid output is bounded to
  **`[0.269, 0.731]`**: the model can never be highly confident in either direction, and a modest systematic offset
  of the expectation value is enough to push most or all samples above 0.5.
- `evaluate_baseline()` (called from `scripts/train_qkan.py`) calls the same `evaluate()` on the warm-started,
  **untrained** model, which produces the baseline metrics of Section 2.

### 3.4 Training dynamics

`outputs/top/seed_10/results/qkan/ideal/history_loss.json` (25 epochs, early stopped): `train_loss` remains in a
narrow band (about 0.648–0.679, compared with `ln(2) ≈ 0.693` for a random classifier) with no clear downward trend;
`val_loss` decreases slowly from 0.678 to 0.659. Over the same 25 epochs, **`val_auc` increases monotonically from
0.698 to 0.731**. Training therefore has an effect (consistent with the baseline → trained AUC gains of Section 2),
but the gradient signal is weak enough that it mainly reshapes the ranking (AUC) without moving the decision surface
enough to correct the bias at the 0.5 threshold (recall and precision barely change). This is more consistent with a
**weak but non-zero gradient or a feature-starved model** than with a barren-plateau circuit.

### 3.5 Feature availability after classical pruning

Pruning thresholds in `src/utils/hyperparams.py` (lines 43–46): `prune_input_th=1e-2`, `prune_node_th=4e-2`,
`prune_edge_th=6e-2`, `prune_max_fanin=2`. Active inputs recorded in
`outputs/top/seed_*/results/chebyshev_coefficients.txt`:

```
seed 3:  active_inputs (raw): [0, 1]
seed 10: active_inputs (raw): [0, 1]
seed 11: active_inputs (raw): [0, 1]
seed 12: active_inputs (raw): [0, 1]
seed 13: active_inputs (raw): [0, 1]
```

In **every** seed inspected, only input columns 0 (jet mass `m`) and 1 (multiplicity `n`) survive pruning. All 20
per-particle substructure features (`dR_i`, `z_i`/`pT_i` for `i = 1..10`) are removed before the quantum stage.
Combined with the mass window (145–205 GeV, `apply_mass_cut=True`, `hyperparams.py:126`), which narrows the useful
range of `m`, the quantum circuit faces an effectively **one- or two-dimensional** decision problem. This explains
the modest AUC ceiling (0.63–0.75) and makes it easy for the model to saturate on one side of a fixed threshold.

### 3.6 Data preprocessing and normalization: `src/preprocessing/processor_top.py`

- Invariant mass `m` (lines 196–202): `log(m+1)` → `RobustScaler` → `tanh(...)`, range **(-1, 1)**.
- Multiplicity `n` (lines 207–219): `StandardScaler` → clipped to ±3σ → divided by 3, range **[-1, 1]**.
- Per-particle `dR_i` / `z_i` (lines 231–232) are stored **unscaled** in `processed_matrix[:, 2::2]` /
  `[:, 3::2]`. These are exactly the features removed in Section 3.5, so this asymmetry has no effect on the
  surviving two-feature circuit.
- Class balance: `balance_classes` / `split_into_subsets` (called from `processor_top.py`) produce a 50/50 dataset;
  every confusion matrix in Section 2 has equal row sums (e.g. 4733/4733). **Class imbalance in the data is ruled
  out** as a cause.

### 3.7 Recent hyperparameter change and script discrepancy

- The uncommitted diff of `src/utils/hyperparams.py` on the working branch at the time contained only:
  ```
  -"base_lamb_l1": 0.05,      +"base_lamb_l1": 0.01,
  -"base_lamb_entropy": 0.05, +"base_lamb_entropy": 0.01,
  ```
  This lowers the L1 and entropy regularization of the **classical** KAN's base training only. It does not affect
  the quantum hyperparameters directly, but weaker regularization changes which edges and nodes the classical model
  keeps, and therefore how strongly pruning reduces the active-input set (Section 3.5). Whether this change was a
  response to the collapse (an attempt to keep more features alive) or unrelated work remains to be confirmed.
- The then-new `scripts/run_seeds.sh` looped over seeds 10–14 and logged to
  `outputs/top/pipeline_no_split_no_masscut_<timestamp>.log`. Its name, and the name of the deleted
  `scripts/run_pipeline_no_split_no_masscut.sh` it replaced, both suggest "no split / no mass cut", but the script
  did not pass any flag disabling `apply_mass_cut` (default `True`, `hyperparams.py:126`) or the 5-way subset split
  (`n_subsets=5`, `hyperparams.py:95`). **Open issue:** the seeds behind the confusion matrices of Section 2 were,
  according to the code, trained on mass-cut, subset-split data, not on the uncut data implied by the script name.

## 4. Candidate causes (ranked by directness of evidence)

1. **Fixed 0.5 threshold on a narrow-range, biased sigmoid output.** The `[-1, 1]`-bounded `PauliZ` expectation
   value limits the sigmoid output to `[0.269, 0.731]`; any systematic positive offset pushes nearly every sample
   above 0.5. Directly supported by recall ≈ 1 and precision ≈ 0.5 in every row of Section 2, for baseline and
   trained models (Section 3.3).
2. **Warm-start bias from the classical → quantum extraction.** The `y_zero`/`subnode_bias` handling in
   `extractor.py:93-105` transfers the zero-input offset of the classical model into the initial `edge_weights`
   before any quantum training. This directly explains why the complete collapses (`[[0,N],[0,N]]`) occur at the
   **baseline** stage in seeds 3 and 10 (Sections 3.2 and 2).
3. **Classical pruning reduces the input space to two features (`m`, `n`) in every seed.** This narrows the decision
   surface and, together with the mass window, probably concentrates most of the separating power in `n`, a surface
   that easily saturates on one side of a fixed threshold (Section 3.5).
4. **Weak gradient signal during quantum fine-tuning**, but not a dead circuit: `val_auc` improves (0.698 → 0.731
   over 25 epochs, seed 10 ideal) while `train_loss`/`val_loss` remain near `ln(2)`. With only 1,000 resampled rows
   per epoch and early stopping at `patience=8`, training lacks the signal or time to correct the initial bias of
   cause 2 (Sections 3.3 and 3.4). A long entangling chain into a single readout qubit (up to 11 qubits,
   `qkan_model.py:182-192`) may contribute to weak gradients but is **not confirmed**; no gradient-norm logging
   existed to separate this from causes 2 and 3.
5. **(Secondary, process issue)** The name and comments of `scripts/run_seeds.sh` claim "no split / no mass cut",
   but its invocation leaves both active (Section 3.7), which makes the condition tested by the seed sweep unclear.

## 5. Candidate solutions

| Cause | Candidate solutions |
|---|---|
| #1 Fixed 0.5 threshold | Calibrate the decision threshold on the validation split (e.g. maximizing Youden's J or F1) instead of the hardcoded `0.5` in `quantum_kan.py:207`; or learn an explicit output scale and bias applied to the `PauliZ` expectation value before the sigmoid, so that the model can shift its operating point during training. |
| #2 Warm-start bias | Check whether `subnode_bias` is double-counted in `_evaluate_isolated_edges` (`extractor.py:60-105`) when several layers or edges reference it; alternatively, zero-center the constant (degree-0) term of the initial `edge_weights`/`output_weights` right after the warm start, before the first `evaluate_baseline()` call, so that the pre-training circuit is not biased toward one class. |
| #3 Pruning reduces to 2 features | Loosen `prune_input_th` / `prune_node_th` / `prune_edge_th` / `prune_max_fanin` (`hyperparams.py:43-46`) and recheck `active_inputs` per seed; or, if a two-feature circuit is intended (for qubit-count or runtime reasons), document it and evaluate whether `m` and `n` alone are expected to suffice for this task. |
| #4 Weak gradient signal | Add per-epoch gradient-norm logging in `fit()` to detect vanishing gradients; consider a local or multi-qubit readout instead of a single `PauliZ` on a long entangling chain; increase `n_train_samples_for_epoch`/`qkan_epochs`/`qkan_patience` if the limitation is the amount of signal per step rather than the gradient magnitude. |
| #5 Script/behaviour mismatch | Align the name and comments of `scripts/run_seeds.sh` with what it runs (mass cut and subset split at the time), or change it to disable both if that was the intended regime. |

## 6. Suggested diagnostic steps

- Plot `test_probs` histograms per seed, backend and stage to measure how tightly the scores cluster around or above
  0.5 (quantifies the severity of cause 1).
- Log the distribution of the raw `PauliZ` expectation value of the untrained (post-warm-start) circuit over a
  batch, right after `QKANModel.__init__` and before `evaluate_baseline()` applies the sigmoid. This isolates whether
  the bias is already present in the raw observable (tests cause 2 directly).
- Log per-epoch gradient norms (`edge_weights.grad`, `output_weights.grad`) during `fit()` to distinguish weak from
  vanishing gradients (cause 4).

Lightweight, read-only tests that exercise some of these checks against saved artifacts were in
`tests/test_baseline_quantum_collapse.py` (removed from `tests/`; available in git history at commit `0b7c53b`).
