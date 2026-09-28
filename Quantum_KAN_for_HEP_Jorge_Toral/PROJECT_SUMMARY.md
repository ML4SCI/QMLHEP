# Quantum Sinusoidal Kolmogorov-Arnold Networks for High Energy Physics: Project Summary

## 1. Overview and Motivation

This document summarizes the work carried out for **QKAN**, a Google Summer of Code 2026 project at ML4SCI. It follows the three project notebooks (`EDA_top.ipynb`, `Training_process.ipynb` and `Analysis_of_results.ipynb`) and concentrates on what each of them concludes and contributes.

Variational quantum circuits (VQCs) are limited in size, because classical simulation cost grows as $O(2^Q)$ with the number of qubits $Q$. Feeding a jet with dozens of features into such a circuit is therefore impractical. We use a classical **Kolmogorov-Arnold Network (KAN)** ([Liu et al. 2024](References.md#ref-liu-2024-kan)) as a **pre-processor and qubit filter**. The KAN is trained, pruned to a small and interpretable topology, and then distilled into a quantum circuit that inherits its structure. The number of surviving hidden nodes decides the number of qubits, so the classical model determines the quantum resources instead of leaving that choice to trial and error.

The workflow preprocesses the jets into balanced replicates, trains and prunes the KAN, extracts each surviving edge into a compact basis (Chebyshev or sine) as a **warm start**, fine-tunes the QKAN on the ideal simulator, evaluates it on ideal, shot-based and noisy simulators, and compares everything against a classical Random Forest.

The main task is **top-quark tagging**, using the public dataset from [Zenodo record 2603256](https://zenodo.org/records/2603256) ([Kasieczka et al. 2019](References.md#ref-kasieczka-2019)). Quark-gluon and Higgs datasets are referenced in the repository, but only top tagging is developed end to end.

---

## 2. Data Exploration and Preprocessing (`EDA_top.ipynb`)

### 2.1 What the dataset contains

The dataset consists of Monte Carlo jets simulated at 14 TeV with Pythia8 and a Delphes ATLAS detector card, without pile-up or multi-parton interactions. Jets are clustered with anti-$k_T$ at $R = 0.8$ in the $p_T$ range [550, 650] GeV. The leading 200 constituents are stored as four-momenta, zero-padded, and sorted by $p_T$. Signal jets (top quarks) are labeled 1 and QCD background 0.

The HDF5 file does not document its column order, so we inferred it from the data. The column means show a pattern every four columns: the first column of each block has a much larger mean than the other three, which stay near zero. We therefore concluded that each block is one constituent, with the **energy first and the three momentum components after it**.

### 2.2 Physical observations

From the distributions of the reconstructed quantities we drew four conclusions.

- **Invariant mass discriminates but is not sufficient.** Signal peaks around the top-quark mass (about 173 GeV), while the background is broader. The two overlap, so we select the window **145-205 GeV** for the experiments. Inside this window the trivial mass cue is largely removed and the classifiers must rely on subtler information.
- **Top jets are more populated.** The multiplicity distribution of top jets sits at higher values and is wider than that of QCD jets. This is consistent with a three-body decay.
- **Top jets are more diffuse.** The radial energy profile and the cumulative $p_T$ fraction rise more gradually for tops and more steeply for QCD, so QCD energy is more concentrated near the jet axis.
- **The $\eta$-$\phi$ scatter plots are not decisive.** We expected a visible difference in dispersion, but noise and axis scales hide it.

We also found an artifact: the $\Delta R$ distribution shows no sharp cut at 0.8, and its outliers come from the small constant added to avoid division by zero. Constituents beyond the jet radius are therefore filtered out.

### 2.3 Feature engineering

The processed representation combines **global** and **local** features:

- **Global:** the jet mass $m_{jet} = \sqrt{E^2 - p_x^2 - p_y^2 - p_z^2}$ (`m`) and the jet multiplicity (`n`), counting constituents with energy above $10^{-8}$.
- **Local:** for each retained constituent, the distance to the jet axis $\Delta R$ and the relative transverse momentum $p_{T,rel}$ (`DR_i`, `pT_i`).

To choose how many constituents to keep, we used the cumulative $p_T$ distribution and an 80% criterion, which suggested about 15 constituents. The training pipeline finally uses the **first 10 constituents**, giving **22 inputs** (2 global and 10 pairs of local features). This truncation limits model size, although each jet stores up to 200 constituents, so the model never sees most of the recorded particles.

Local features already lie in $(0, 1)$. The jet mass is transformed with $\log(m + 1)$, scaled with a robust scaler and passed through a **tanh**. The multiplicity is scaled with a robust scaler and clipped at $3\sigma$, so that it stays within $[-1, 1]$. We deliberately kept the outliers, because the centers of the two class distributions are similar and the tails carry information.

### 2.4 Balancing and replicates

The number of events in the mass window differs between classes, with more tops. We **undersample the majority class** so that both classes have equal representation. The balanced dataset is then split into **5 disjoint, class-balanced subsets**. Each seed selects one subset (`seed % 5`), so several seeds provide independent end-to-end replicates instead of a single point estimate. Preprocessing writes a canonical cache once, and every later script only selects from it.

---

## 3. Training Process and Architecture (`Training_process.ipynb`)

### 3.1 The classical KAN

The classical model is built on [pykan](https://github.com/kindxiaoming/pykan) and modified in `HEPKAN`. The modifications are motivated by the available computational resources:

- A **bug fix in `prune_input`**, which passed a module instead of a name string when rebuilding the pruned model and broke checkpoint serialization.
- A **plotting routine** that reuses one Matplotlib figure for all edges and skips pruned edges, reducing time and memory on wide networks.
- A **no-op history logger**, which avoids writing a checkpoint on every model mutation during pruning and symbolic search.

The architecture is `[22, [9, 9], 1]`. The input layer has 22 features, the hidden layer offers up to 9 addition nodes and 9 multiplication nodes, and the output has a single unit. The B-splines use grid size 5 and order $k = 3$, for **8,568 parameters** in total.

The **multiplication nodes**, introduced in [KAN 2.0](References.md#ref-liu-2024-kan2), matter because the original Kolmogorov-Arnold theorem only guarantees a representation through univariate functions and addition. Multiplication lets the network express feature interactions directly.

The main training values of each classical stage are:

| Stage | Learning rate | Max epochs | Batch size | Early-stop patience | Min delta |
|---|---|---|---|---|---|
| Base training | $5 \times 10^{-3}$ | 60 | 4096 | 7 | $10^{-2}$ |
| Retraining after pruning | $10^{-3}$ | 20 | 2048 | 6 | $5 \times 10^{-5}$ |
| Symbolic fine-tuning | $5 \times 10^{-5}$ | 15 | 2048 | 5 | $10^{-6}$ |

For the reference run (seed 10) the base model reached a test AUC of **0.792** and stopped early after about 20 of the 60 planned epochs.

### 3.2 Pruning, retraining and symbolic fitting

During the base training, the loss includes the sparsity regularization of pykan with an overall weight $\lambda = 0.01$ (L1, entropy and two spline-coefficient terms), which pushes low-contribution activations toward zero. Pruning then removes inputs below a threshold of 0.01, and hidden nodes and edges with attribution thresholds of 0.04 and 0.06. A hard **fan-in cap** keeps at most the two strongest input edges per hidden neuron. The pruned structure is retrained for up to 20 epochs with a lower overall weight ($\lambda = 0.001$) and stronger L1 and entropy terms, so it adapts to the removed connections while it stays sparse. In the reference run the validation AUC rose from 0.714 to about 0.754 during this step.

The notebook also covers **symbolic fitting**. Each surviving edge is compared against a library of functions; each candidate receives a score that balances $R^2$ and simplicity, and the best one is fixed on the edge only if its $R^2 \geq 0.85$ (otherwise the edge becomes the identity). A short fine-tuning step then adjusts the parameters of the formulas. This yields a formula-level interpretation of the model, which we consider the main interpretability advantage of KANs. It belongs to the classical branch; the quantum branch instead takes the numerical response of each edge.

The classical KAN is therefore evaluated at four stages: **base**, **retrained** (after pruning), **symbolic**, and **final** (symbolic after fine-tuning). Each stage saves its own checkpoint and plots.

### 3.3 Extraction to a quantum graph

The extractor reads the retrained checkpoint and isolates the response of each active edge by disconnecting the other inputs of its target node. Edges with a dynamic range below $10^{-3}$ are discarded. It exports a serialized graph of sum and multiplication nodes (`quantum_weights.pt`). In the reference run (seed 10), the surviving structure was small:

- **2 active input variables**, the jet mass `m` and the multiplicity `n`,
- **1 sum node and 5 multiplication nodes** in the hidden layer,
- **11 qubits**, 18 input edges, 5 `IsingZZ` transfers and 6 output edges.

Across the five seeds the graph always keeps `m` and `n`, with 10 or 11 qubits: seeds 11 and 12 keep no sum node and use 10 qubits. Qubits are counted per surviving accumulator node, **not per input**. The same input can be re-uploaded onto several wires when it feeds several hidden nodes.

### 3.4 The quantum circuit

The circuit follows five design principles:

1. **Data re-uploading** models each univariate function through repeated rotations of the input.
2. The hidden layer decides which features matter and how many qubits are used.
3. **Addition is free**: consecutive $R_Z$ rotations on the same wire accumulate their angles, so sum nodes need no two-qubit gate.
4. **Multiplication** uses an `IsingZZ` gate combined with a `CNOT`. For multiplication nodes with more than two inputs, the chain of `IsingZZ` gates is an approximation of the product.
5. The information is collapsed onto one output wire, and the prediction is a single-qubit Pauli-Z expectation.

Each input $x$ is encoded as $\theta = \arccos(x)$, with $x$ clipped to $[-0.9999, 0.9999]$, which links the circuit to the Chebyshev polynomials through $T_n(x) = \cos(n \arccos x)$. Each edge applies 4 repetitions (the Chebyshev degree) of $R_Y(w_i)$ followed by $R_Z(\theta)$, plus a final $R_Y$, giving 5 trainable weights per edge. The readout $\langle Z \rangle \in [-1, 1]$ is used directly as the logit.

The hidden-to-output stage is a variational readout and not a literal second KAN layer, because a hidden node's value lives in a qubit phase and cannot be re-uploaded without mid-circuit measurement. Only depth-2 networks are therefore supported.

The model can run on three simulators: `ideal` (`lightning.qubit`, exact expectation values), `shots` (`default.qubit` with 1024 shots) and `noisy` (Qiskit Aer density-matrix simulator with the noise model of `FakeManilaV2` and 1024 shots). Manila is a 5-qubit device, while the circuits use 10 or 11 qubits. All the training runs use the `ideal` backend.

Training uses binary cross-entropy with logits, the Adam optimizer (learning rate $5 \times 10^{-3}$, gradient norm clipped at 1.0) and a `ReduceLROnPlateau` scheduler that halves the learning rate after 6 epochs without improvement. Because simulation is expensive, each epoch trains on a **fresh random subset** of 1,000 samples and validates on a fixed set of 2,000 samples. With a batch size of 1,024, **each epoch is a single optimization step**. The training runs for up to 50 epochs and stops after 8 consecutive epochs in which neither the validation loss nor the validation AUC improves by more than $5 \times 10^{-3}$.

`scripts/train_qkan.py` evaluates the warm-started circuit on the three backends before training and again after training, which gives a before/after by ideal/shots/noisy grid.

---

## 4. Warm-Start Strategies

We compare three ways of initializing the circuit angles.

**Chebyshev polynomials.** Following the design of the [Chebyshev-KAN](References.md#ref-sidharth-2024), each isolated edge response is fitted with $y \approx \sum_{i=0}^{N} c_i T_i(x)$ on $[-1, 1]$, and the coefficients become the initial rotation angles.

*Degree selection.* The degree is fixed at $N = 4$ for every edge. An earlier version searched for the smallest degree that met an $R^2$ threshold. On smaller training pools this search almost always selected low degrees, and the circuit lost part of the information of the classical structure. Since `chebfit` refits all coefficients at each degree, the fits are not nested, and truncating at a lower degree also changes the low-order coefficients. The effect appeared as an abrupt drop in baseline AUC (from about 0.80 to 0.26-0.36), and reverting to the fixed degree restored it.

**Sine basis.** As an alternative we implemented the fixed-frequency sinusoidal basis of [SineKAN](References.md#ref-reinhardt-2024), $y \approx \sum_k A_k \sin(\text{freq}_k x + \text{phase}_k)$, with amplitudes obtained by least squares.

*Fit quality.* Over the 110 extracted edges of the five seeds, the Chebyshev basis reproduces the splines almost perfectly (mean and median $R^2$ of 0.998), while the sine basis reaches a mean $R^2$ of 0.56, a median of 0.84, and several negative values (minimum $-0.66$). The sine design matrix has **no constant column**, and every edge carries a constant offset, because the extractor gives each edge an equal share $y_{\mathrm{zero}}/n_{\mathrm{in}}$ of the constant term of its node. To approximate the offset, the least-squares solution distorts the other amplitudes, and without an intercept the fit can be worse than the mean of the response. An extended basis with a constant term is proposed as future work.

**Random initialization.** As a control, we keep the pruned topology exactly (same qubits and connections) but sample every edge and output angle from $\mathcal{N}(0, 1)$. The `IsingZZ` angles start at zero in every initialization. This isolates the value of the classical knowledge from the value of the topology alone.

---

## 5. Random Forest Baseline

To calibrate what is achievable classically, we add a **Random Forest** with 300 trees, a maximum depth of 35, at least 10 samples to split a node, at least 5 samples per leaf, a fraction of 0.35 of the features at each split, and balanced class weights. In contrast to the KAN pipeline, it receives all **22 features** without pruning. Its metrics use the same keys as the KAN trainers, so all models can be aggregated into one table.

The feature importance (MDI/Gini) ranks the jet mass and the multiplicity first by a wide margin, followed by the angular distances of the leading constituents. This **retrospectively supports the aggressive pruning** of the KAN, which independently retained the same two variables.

---

## 6. Results (`Analysis_of_results.ipynb`)

`scripts/collect_metrics.py` gathers the per-run metrics into a Parquet table, and `scripts/collect_eval_data.py` gathers the per-event predictions and training histories into a second one. All runs use the invariant mass cut and the 5 disjoint subsets: each seed (10 to 14) trains and evaluates every model on its own subset, which gives 65 rows (13 models by 5 seeds). The whole set of runs can be reproduced with `scripts/reproduce_results.sh`.

### 6.1 Model comparison

Mean $\pm$ standard deviation over the five seeds:

| Model | Test AUC | Accuracy | Bkg rejection at $\varepsilon_S = 0.5$ |
|---|---|---|---|
| Random Forest (22 features) | **0.823 $\pm$ 0.004** | 0.746 $\pm$ 0.004 | 10.04 $\pm$ 0.57 |
| Classical KAN, base | 0.789 $\pm$ 0.002 | 0.712 $\pm$ 0.004 | 7.20 $\pm$ 0.37 |
| Classical KAN, retrained | 0.756 $\pm$ 0.004 | 0.687 $\pm$ 0.005 | 5.81 $\pm$ 0.29 |
| Classical KAN, symbolic | 0.756 $\pm$ 0.004 | 0.687 $\pm$ 0.005 | 5.79 $\pm$ 0.29 |
| Classical KAN, final (symbolic + fine-tune) | 0.770 $\pm$ 0.014 | 0.696 $\pm$ 0.013 | 6.35 $\pm$ 0.75 |
| **QKAN, trained (ideal)** | **0.736 $\pm$ 0.006** | 0.563 $\pm$ 0.013 | 5.46 $\pm$ 0.14 |
| QKAN, trained (shots) | 0.732 $\pm$ 0.006 | 0.571 $\pm$ 0.015 | 5.42 $\pm$ 0.13 |
| QKAN, trained (noisy) | 0.730 $\pm$ 0.006 | 0.571 $\pm$ 0.016 | 5.40 $\pm$ 0.13 |
| QKAN warm start only, Chebyshev (ideal) | 0.698 $\pm$ 0.004 | 0.551 $\pm$ 0.015 | 4.37 $\pm$ 0.23 |
| QKAN warm start only, Sine (ideal) | 0.637 $\pm$ 0.015 | 0.531 $\pm$ 0.008 | 3.26 $\pm$ 0.15 |
| QKAN warm start only, random (ideal) | 0.488 $\pm$ 0.021 | 0.492 $\pm$ 0.014 | 1.81 $\pm$ 0.21 |

Three observations follow from this table.

First, pruning costs roughly 0.03 AUC relative to the base KAN (0.789 to 0.756). The retrained KAN already works with the same 2 inputs as the circuit, so it is the realistic ceiling for the QKAN, which sits a further 0.02 below it. The QKAN reaches an AUC of about 0.73-0.74 with only two input variables and 10 or 11 qubits.

Second, **fine-tuning matters**. Training raises the Chebyshev warm start from 0.698 to 0.736 in the ideal case, and it improves the AUC in every seed and on every backend.

Third, ideal, shot-based and noisy backends differ by no more than about 0.006 AUC. Within our noise model, the circuit is not visibly degraded.

### 6.2 Do the warm starts matter? Hypothesis tests

Since the seeds are few, we tested the differences explicitly with paired t-tests, pairing the models by seed, at $\alpha = 0.05$. The null hypothesis was that random initialization and the warm start give the same accuracy and AUC. Since we run 4 tests, we also report the Holm-corrected p-values.

| Comparison | Metric | $t$ | $p$ | Holm $p$ |
|---|---|---|---|---|
| Random vs Chebyshev | Accuracy | $-4.86$ | $0.0082$ | $0.015$ |
| Random vs Chebyshev | AUC | $-20.17$ | $3.6 \times 10^{-5}$ | $1.4 \times 10^{-4}$ |
| Random vs Sine | Accuracy | $-5.00$ | $0.0075$ | $0.015$ |
| Random vs Sine | AUC | $-17.95$ | $5.7 \times 10^{-5}$ | $1.7 \times 10^{-4}$ |

All four null hypotheses are rejected, also after the Holm correction. We nevertheless read these results cautiously: with **only five paired samples** the tests have low power, so they support the ordering of the models more than the exact size of the differences.

The same ordering appears at the 50% signal-efficiency working point:

- **Chebyshev init:** background efficiency 0.230 $\pm$ 0.012, background rejection **4.37 $\pm$ 0.23**.
- **Sine init:** background efficiency 0.307 $\pm$ 0.015, background rejection **3.26 $\pm$ 0.15**.
- **Random init:** background efficiency 0.560 $\pm$ 0.064, background rejection **1.81 $\pm$ 0.21**. It also misses the target efficiency ($0.57 \pm 0.08$), because its scores are almost constant and the threshold cannot be placed precisely.
- **Chebyshev after training:** background efficiency 0.183 $\pm$ 0.005, background rejection **5.46 $\pm$ 0.14**.

### 6.3 Background rejection curves

The ordering holds at every signal efficiency. At $\varepsilon_S = 0.3$ the Random Forest rejects about 24 background jets for each accepted one, the final classical KAN about 14 and the trained QKAN about 11. At $\varepsilon_S = 0.9$ all models converge toward low rejections (2.11 for the Random Forest and 1.59 for the trained QKAN), since keeping 90% of the signal requires accepting most of the background. The random initialization stays close to the behavior of a random classifier. A likely reason for the advantage of the Random Forest at low signal efficiency is its access to the angular distances of the leading constituents, which the pruning removes before the KAN retraining.

### 6.4 Classical pipeline stages

The symbolic fit keeps the AUC of the retrained model (0.756), so the fitted formulas reproduce the splines well. The final fine-tuning recovers part of the loss (0.770) with a larger spread between seeds, since its first epochs can be unstable (in seed 10 the first training loss reaches values of order $10^8$ before it settles). One hypothesis is that the retrained model stops before convergence, and the fine-tuning works as additional epochs on the same graph.

### 6.5 Training curves

The classical KAN converges in about 20 epochs and its validation AUC reaches a plateau before the early stopping. The QKAN behaves differently: its validation AUC grows almost linearly, by about 0.003 per epoch, and it is still increasing when training stops (between 9 and 22 epochs out of 50). Each epoch is a single optimization step, and the early-stopping threshold of $5 \times 10^{-3}$ is larger than the gain of one step. **The QKAN is therefore undertrained**, and its reported metrics are a lower bound. The validation loss stays around 0.66, close to $\ln 2$: with the logit bounded to $[-1, 1]$, even a perfect classifier cannot bring the loss below $\ln(1 + e^{-1}) \approx 0.31$. The gradients do not vanish, so a barren plateau is an unlikely explanation.

### 6.6 Output scores and calibration

**Accuracy requires caution.** The classical KAN uses almost the whole $[0, 1]$ probability range, while the QKAN probabilities stay in about $[0.48, 0.72]$ because $\langle Z \rangle$ is used as a bounded logit. With the default threshold of 0.5 almost every event is classified as signal: more than 97% of the top jets are accepted, but only 13% to 15% of the QCD jets. This explains the high recall (about 0.97) and the low accuracy (about 0.56-0.57, against about 0.70 for the classical KAN). The Random Forest and the classical KAN have nearly symmetric confusion matrices.

We hypothesize that the output stays close to the initial state $|0\rangle$, where $\langle Z \rangle = +1$, for most events. Setting the constant Chebyshev term to zero changes the AUC by less than 0.003, so it does not seem responsible. The ranking signal, measured by AUC and background rejection, survives, and a threshold chosen on the validation set or a scaling of $\langle Z \rangle$ would give an accuracy consistent with it. We report AUC and background rejection as the primary metrics for this reason.

On the shot-based and noisy backends the true negative rate of the trained QKAN is slightly higher (about 0.19 against 0.15 on `ideal`). We attribute this mainly to shot noise: since most events sit just above 0.5, a symmetric fluctuation moves more of them below the threshold than above it.

### 6.7 Computational cost

The noisy density-matrix simulation is the most expensive step, with about 1,300 to 3,300 s per test-set evaluation depending on the run, against about 135 s on the ideal simulator and 27 to 91 s with shots. For comparison, the whole classical KAN pipeline takes about 379 s and the Random Forest about 14 s. The density-matrix cost grows as $4^Q$, which is one more reason to keep the circuit small.

---

## 7. Conclusions

We draw five conclusions from the notebooks, for top tagging with the mass cut and five disjoint subsets.

1. **Classical pruning works as a qubit filter.** The KAN reduces 22 inputs to two variables (`m` and `n`) and 10 or 11 qubits, the same two variables that the Random Forest ranks first. The circuit still reaches an AUC of about 0.73. The pruning costs about 0.03 AUC in the classical model, and the retrained KAN with the same inputs sets the ceiling of the QKAN.
2. **The warm start carries real information.** The ordering Chebyshev > Sine > Random is consistent across seeds and statistically significant on five replicates, also after the Holm correction. Random initialization performs at chance level (AUC about 0.49), which shows that the topology alone is not enough. The Chebyshev basis is better because its per-edge fit is much closer to the classical response ($R^2$ of 0.998 against a mean of 0.56 for sines, whose basis lacks a constant term).
3. **Quantum training helps and is not finished.** Training raises the AUC from 0.698 to 0.736, and the validation curves are still rising when the early stopping triggers, so this value is a lower bound.
4. **Simulated noise has a small effect.** The gap between the `ideal` and `noisy` backends is at most about 0.006 AUC. This is encouraging, but it comes from a simulated noise model that covers 5 of the 10-11 qubits and **not from hardware**.
5. **The classical ceiling is set by the Random Forest.** It has access to all features and leads in every metric; the quantum model does not surpass it. The narrow output range of the circuit makes the accuracy at a 0.5 threshold misleading, so AUC and background rejection give the fair comparison. The value of the hybrid approach lies in a compact, interpretable circuit that preserves a large fraction of the classical performance.

---

## 8. Contributions

- **A reproducible, regime-aware workflow.** Data regimes are encoded in the directory layout, replicate subsets are class-balanced, every stage is idempotent by checkpoint, and results from several seeds are collected into two Parquet tables: the metrics (`scripts/collect_metrics.py`) and the per-event predictions with training histories (`scripts/collect_eval_data.py`). `scripts/reproduce_results.sh` runs the whole pipeline.
- **`HEPKAN`**, a pykan subclass that fixes a serialization bug in input pruning and reduces plotting and logging overhead.
- **A pruning rule tailored to quantum limits**, combining attribution thresholds with a hard fan-in cap.
- **A classical-to-quantum bridge.** The extractor turns a pruned KAN into a sum/multiplication graph, and the QKAN builder turns that graph into a PennyLane circuit.
- **Two interchangeable bases plus a control**, which allow the value of the warm start to be measured, on three simulation backends evaluated before and after training.
- **A diagnosed and fixed failure.** The adaptive Chebyshev degree search silently collapsed baseline AUC on smaller data pools; we found the cause and replaced it by a fixed degree.
- **A classical benchmark**, the Random Forest, evaluated with the same metrics and split.
- **An exploratory analysis** documenting the dataset layout, the $\Delta R$ artifact and the choice of 10 constituents and 22 inputs, and **a results analysis** covering hypothesis tests, rejection curves, training curves, score distributions, confusion matrices, extraction quality and computational cost.

---

## 9. Limitations and Future Work

- **Few replicates.** The statistical tests use five seeds, and each seed uses its own subset, so the seed effect and the subset effect cannot be separated.
- **Mass-cut regime only.** The results are restricted to the 145-205 GeV window, so they cannot be compared directly with the literature benchmarks on the full dataset. Evaluating the regime without the mass cut with replicates is left for future work.
- **Extreme compression.** Pruning leaves two input variables, and loosening it raises the qubit count beyond what the simulator can evaluate in reasonable time (21 qubits in a test on seed 12). Only the 10 leading constituents out of up to 200 per jet are used.
- **Undertraining.** One optimization step per epoch and a coarse early-stopping threshold stop the QKAN while it is still improving.
- **Calibration.** Quantum accuracy and precision are weak even where AUC is good. A tuned decision threshold or a scaling of $\langle Z \rangle$ should be studied.
- **Sine basis.** It needs a constant term before it can be judged fairly.
- **Simulated and partial noise.** `FakeManilaV2` approximates a real 5-qubit device, so its noise covers only part of the 10-11 qubit circuit.
- **Depth-2 networks only.** Deeper KANs would need mid-circuit measurement and re-encoding.
- **Other datasets.** Quark-gluon tagging has a preprocessing pipeline, and Higgs detection is only referenced.

The results show that a classical KAN can determine the structure of a quantum circuit, that the knowledge transferred through an accurate basis has a measurable effect, and that the resulting compact model preserves a substantial part of the classification signal. The model does not yet match the best classical baseline; this limitation is reported together with the results.

Full bibliography: [`References.md`](References.md).
