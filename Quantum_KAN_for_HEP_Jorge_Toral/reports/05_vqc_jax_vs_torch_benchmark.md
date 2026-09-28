# VQC Training-Step Benchmark: JAX vs. PyTorch Backends

**Scope:** wall-clock time of one QKAN training step (forward, loss, gradient, Adam update) on a 6-qubit circuit,
comparing the production PennyLane `lightning.qubit` + Torch path with `default.qubit` + JAX on CPU and GPU.
**Status:** benchmark only; no production code was modified. Line numbers refer to the code at the time of writing.

## Summary

At the circuit sizes produced by the QKAN pruning pipeline, `default.qubit` + JAX on CPU is about **129× faster**
per training step than the production `lightning.qubit` + Torch path. The NVIDIA Quadro P4000 GPU gives almost no
additional benefit (1.20× over the production path, about 107× slower than JAX on CPU).

## Method

Script: `tests/benchmark_vqc_jax_vs_torch.py` (standalone benchmark, not a pytest test; removed from `tests/`; available in git history at commit `0b7c53b`). No file under `src/` was modified; `QKANModel`
(`src/architectures/qkan_model.py`) is used **unmodified** as the Torch/lightning baseline.

**Synthetic graph.** No extracted QKAN graph (`quantum_weights.pt`) existed under `outputs/` at the time of the
benchmark. A synthetic graph dictionary was therefore used, shaped to match the qubit-count range and sum/mult mix of
real extracted graphs reported in `reports/02_chebyshev_degree_ceiling_comparison.md` (4–7 qubits, with either
all-sum or all-mult hidden layers).

**Circuit under test.** 6 qubits, Chebyshev degree 4 (`chebyshev_max_degree` in `src/utils/hyperparams.py`), one
sum-type hidden node (2 input edges accumulating on one wire), one mult-type node of arity 2 (a single
`IsingZZ` + `CNOT`) and one mult-type node of arity 3 (chained `IsingZZ` + `CNOT`). This exercises both branches of
the stage-2 variational readout (plain `RY` and `IsingZZ` + `CNOT`). The circuit has 7 input edges, 3 `IsingZZ`
transfers, 3 output edges and 53 trainable scalars (`edge_weights` (7,5) + `zz_weights` (3,) + `output_weights`
(3,5)), matching the scale of real extracted graphs. The connectivity was built with a pure-Python port of the
topology loop of `QKANModel.__init__` (`qkan_model.py:63-112`), used identically by both variants, so the wiring of
the JAX circuit is the same as the one `QKANModel` builds from the same graph dictionary.

**Implementations of the same circuit.**
- **`torch` (production path):** `QKANModel(graph_path=..., backend_mode="ideal")`, device
  `qml.device("lightning.qubit", wires=6)`, `interface="torch"`. Trained with
  `torch.optim.Adam(lr=5e-3, betas=(0.9,0.999), eps=1e-8)` and `BCEWithLogitsLoss`, matching the optimizer
  configuration of `QuantumKANTrainer.fit` (`src/utils/hyperparams.py`).
- **`jax` variant:** a `qml.QNode` on `qml.device("default.qubit", wires=6)` with `interface="jax"`, a line-for-line
  port of `_circuit`/`_qkan_edge` (`qkan_model.py:158-194`) using `jnp` instead of `torch`, with the same batched
  parameter-broadcasting call pattern (no `vmap`). Since `optax` is not installed, a short hand-written Adam step
  with the same hyperparameters was used instead of adding a dependency. The full training step (forward, loss,
  gradient, Adam update) is wrapped in `jax.jit`, and `jax.block_until_ready(...)` is called on the loss inside the
  timed region, since JAX dispatch is asynchronous and would otherwise measure dispatch latency instead of compute.
  `diff_method` was left at the PennyLane default for both variants.

**Data.** One fixed synthetic batch (`numpy.random.default_rng`, seeded): `X` of shape `(32, 7)`, uniform in
`[-1, 1]` (valid `arccos` domain), and `y` of shape `(32,)` in `{0, 1}`, reused identically for every iteration and
variant, so data loading does not affect the timing.

**Timing protocol.** 3 warm-up iterations (iteration 0 timed separately as `first_call_seconds` to expose JIT
compilation or tape-construction overhead; iterations 1–2 discarded), followed by 8 timed steady-state iterations
(`time.perf_counter()`). JAX fixes its backend (`cpu`/`cuda`) at import time through `JAX_PLATFORMS` and cannot switch
within a process, so `torch`, `jax_cpu` and `jax_gpu` each ran in a separate subprocess with
`CUDA_VISIBLE_DEVICES`/`JAX_PLATFORMS` set accordingly. The `jax_gpu` process forces `JAX_PLATFORMS=cuda`, so an
unusable GPU raises an error instead of silently falling back to CPU; the GPU run executed on `cuda:0`.

**Hardware.** 10 CPUs, 8 GB RAM, NVIDIA Quadro P4000 (8 GB VRAM, Pascal, compute capability 6.1, driver 580.178.04,
no system `nvcc`/CUDA toolkit).

**Package versions.** `pennylane==0.45.1`, `pennylane-lightning==0.45.0`, `torch==2.4.1+cu121` (CPU-only for this
benchmark: `lightning.qubit` does not use CUDA, and `CUDA_VISIBLE_DEVICES=""` was set), `jax==0.10.2`/`jaxlib==0.10.2`
with `jax-cuda12-plugin`/`jax-cuda12-pjrt` (pip-bundled CUDA runtime, no system toolkit required).

**Dependency handling.** `jax`, `jaxlib` and the CUDA 12 plugin were installed temporarily with
`uv pip install "jax[cuda12]"` into `.venv`, without modifying `pyproject.toml` or `uv.lock`. After the run the
environment was restored with `uv pip sync <pre-install-freeze>`. This reverted the 9 packages added by JAX (`jax`,
`jaxlib`, `jax-cuda12-pjrt`, `jax-cuda12-plugin`, `ml-dtypes`, `opt-einsum`, `nvidia-cuda-cccl-cu12`,
`nvidia-cuda-nvcc-cu12`, `nvidia-nvshmem-cu12`) and also a transitive side effect that removing only the new
packages would have missed: the JAX installation had changed `nvidia-cudnn-cu12` from `9.1.0.70` (the version
required by the `+cu121` Torch wheel) to `9.26.0.51`. The `uv pip freeze` output before and after the full
install → run → revert cycle is byte-for-byte identical, and `git status --porcelain pyproject.toml uv.lock` was
empty throughout.

## Results

| Variant | Backend | First call / compile (s) | Mean steady-state step (s, n=8) | Steps/s | Speed-up vs. `torch`/`lightning.qubit` |
|---|---|---:|---:|---:|---:|
| `torch` (production) | `lightning.qubit`, CPU | 1.03 | 0.3043 | 3.29 | 1.0× (baseline) |
| `jax` | `default.qubit`, CPU | 3.99 | 0.002367 | 422.5 | **128.6×** |
| `jax` | `default.qubit`, GPU (Quadro P4000, `cuda:0`) | 13.78 | 0.2536 | 3.94 | 1.20× |

The raw JSON output with all 8 per-step timings per variant is printed to standard output when
`tests/benchmark_vqc_jax_vs_torch.py` (see Method) is run; it is not stored under `outputs/`.

## Discussion

1. **JAX + `default.qubit` on CPU is much faster than the production `lightning.qubit` + Torch path at the current
   QKAN circuit size (about 129×)**, although `lightning.qubit` is the C++-accelerated PennyLane simulator and
   `default.qubit` is the pure-Python reference device. The probable explanation is dispatch and tape-construction
   overhead rather than simulation speed: `lightning.qubit` + Torch builds and dispatches a new PennyLane tape
   through the Torch autodiff bridge on every call, whereas `jax.jit` traces the *entire* training step once
   (compilation cost visible as `first_call_seconds ≈ 4 s`) and then executes a single compiled program per step. At
   about 70 gates on 6 qubits, the per-call Python and tape overhead dominates the wall-clock time, and JIT
   compilation removes it.
2. **The Quadro P4000 run executed on CUDA (`jax_devices: ["cuda:0"]`, no fallback) but gave almost no benefit over
   the CPU baseline (1.20×) and was about 107× slower than JAX on CPU for the same circuit.** A 6-qubit state vector
   (64 complex amplitudes) is far too small to amortize GPU kernel-launch and host–device synchronization overhead:
   each of the roughly 70 sequential one- and two-qubit gates is a separate small kernel launch, which costs more
   than the complete CPU computation. The one-time compilation cost (13.8 s vs. 4.0 s for JAX on CPU) shows that
   CUDA-specific XLA compilation is also slower.
3. **The result is specific to this circuit size.** It applies to the circuits currently produced by the QKAN pruning
   pipeline (4–7 qubits, per `reports/02_chebyshev_degree_ceiling_comparison.md`). GPU state-vector simulation is
   known to become advantageous only when the state vector is large enough (roughly ≥ 15–20 qubits, i.e. ≥ 32K–1M
   amplitudes) for per-gate compute to exceed kernel-launch and synchronization overhead, a regime that the pruned
   graphs of this pipeline do not reach.
4. **`final_loss` was consistent across the three variants** (≈ 0.6894, close to `ln(2)`, the expected BCE loss at
   initialization for a near-zero logit). This is a sanity check that the JAX port and `QKANModel` compute the same
   circuit; it is not an accuracy result (this was a fixed 8-step timing run on random data and weights).

## Conclusions and recommendations

**JAX + `default.qubit` (CPU) is recommended as the target for a future rewrite of the QKAN training loop.** A
speed-up of about 129× per step at the circuit sizes produced by this pipeline would turn training at the scale of
`qkan_epochs * n_train_samples_for_epoch / qkan_batch_size` (currently 50 epochs over 1,000-sample subsets,
`src/utils/hyperparams.py`) from a simulation-bound loop into one that could run full-dataset epochs instead of
subsampling, provided the one-time JIT compilation cost of about 4 s per shape is acceptable. This cost is paid once
per circuit topology, not per epoch or batch, as long as the static structure and array shapes of the traced
function remain fixed. The change is not a drop-in replacement: `QKANModel` hardcodes `interface="torch"`
(`qkan_model.py:133`), and the rest of the pipeline (`QuantumKANTrainer`, `ClassicKANTrainer`, metrics and plotting)
is Torch-native. Adoption requires either a JAX-native `QKANModel` variant or a Torch–JAX interoperability layer,
which is a separate implementation task.

**The Quadro P4000 GPU is not recommended for the current circuit sizes.** It does not meaningfully outperform the
CPU-only production path (1.20×) and is far slower than JAX on CPU (107×). If future pruning or extraction changes
produce substantially larger circuits (double-digit qubit counts), this conclusion should be re-tested, since the
GPU benefit for state-vector simulation grows with circuit size.

No file under `src/` was modified; `QKANModel` was used as committed. The temporary `jax`/`jaxlib`/CUDA-plugin
installation was fully reverted after the benchmark: `uv pip freeze` is identical before and after the
install → run → revert cycle, and `git status --porcelain pyproject.toml uv.lock` was empty throughout.
