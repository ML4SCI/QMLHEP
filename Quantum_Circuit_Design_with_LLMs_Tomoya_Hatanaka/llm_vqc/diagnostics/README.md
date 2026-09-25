# Circuit diagnostics (`llm_vqc.diagnostics`)

**Phase:** 3 of the LAQS-Bench roadmap (`LLM-VQC_MASTER_PLAN.md`
Section 9). See `DECISIONS.md` for the Phase 3 completion record and every
consequential design decision.

## Purpose

Three scientific measurements of a `CircuitIR`, computed **identically for
every future LAQS-Bench search arm** and with **no access to task data**:

- **expressibility** — Sim et al. 2019 KL-divergence-vs-Haar;
- **entangling capability** — average Meyer-Wallach measure;
- **gradient variance** — a barren-plateau / trainability probe.

These are measurements, not optimization targets. They must not be fed to
one search arm as extra information or budget unless that use is an
explicitly declared experimental condition in the master plan.

## Shared sampling policy (all three diagnostics)

Every diagnostic treats the circuit as a **variational ansatz over its
trainable weights**: it samples the trainable weights uniformly on
`[0, 2*pi)` while holding the data-encoding inputs at a **fixed diagnostic
input** — zeros for angle encoding (which makes the encoding gates the
identity, recovering exactly Sim et al.'s pure-ansatz picture
`U_variational(theta)|0...0>`), or a fixed seeded L2-normalized vector for
amplitude encoding. This is:

- **faithful to Sim et al.** (their sampled parameters are the trainable
  rotation angles; their circuits have no separate data encoding);
- **uniform** across angle- and amplitude-encoded circuits;
- **consistent** with the gradient diagnostic, which must differentiate
  with respect to the trainable weights (never data).

The alternative — also sampling the encoding-input angles — is recorded in
`DECISIONS.md` (D13) with the reason it was not chosen. A consequence
worth stating: a circuit with **no trainable weights** produces a single
fixed state under this policy, so it is correctly the least expressible
(the idle circuit hits the analytic upper bound).

**Seeds.** Randomness is a pure function of `(run_seed, structural_hash,
diagnostic_name)` (mirroring the Phase 2 `train_seed_for_circuit`
contract), split into independent `SeedSequence` streams for parameter
sampling, the diagnostic input, backend sampling (reserved for finite
shots; statevector simulation uses none), and bootstrap resampling. Each
of `n_replicates` replicates gets its own independent full set of streams.

## Expressibility (`expressibility.py`)

**Definition (Sim et al. Eq. 17):**
`Expr = D_KL( P_hat_PQC(F) || P_Haar(F) )`, where `F = |<psi_theta |
psi_phi>|^2` is the fidelity between the states of two independently
sampled weight vectors, `P_hat_PQC` is the histogram of sampled
fidelities, and `P_Haar(F) = (N-1)(1-F)^(N-2)` with `N = 2^n_qubits`.

- **Lower Expr = more expressible.** The least-expressible circuit (fixed
  output state) puts all fidelity mass at `F=1` and attains the exact
  upper bound `(N-1) ln(n_bins)`.
- **Estimator:** sample `n_pairs` fidelities; histogram into `n_bins`
  equal-width bins over the **fixed** range `[0, 1]` (never adapted per
  circuit — adaptive bins could bias the comparison). The Haar reference
  is computed **analytically per bin** — for bin `[a, b]`, the exact mass
  is `(1-a)^(N-1) - (1-b)^(N-1)`, always strictly positive.
- **KL zero-bin policy:** empty empirical bins (`P_i = 0`) contribute
  exactly `0` (the `0*log(0)=0` convention); no epsilon smoothing. Safe
  because every Haar bin probability is positive.
- **Robustness diagnostic:** the same fidelities are re-histogrammed at
  `robustness_n_bins` and its KL reported separately, labeled a
  bin-sensitivity check — it never replaces the primary value.
- **Uncertainty:** repeated-seed dispersion — mean ± std of the per-
  replicate Expr values (Sim et al.'s "error bars = std over 5
  independent computations").
- **Known estimator bias:** KL/entropy estimators are biased at finite
  sample size, most at low counts (Sim et al. Appendix B). Small-sample
  smoke/test configs over-report Expr; only research-scale counts (5000
  pairs) are quantitatively meaningful.
- **Metrics reported:** `expressibility_kl` (primary),
  `expressibility_kl_robustness_bins`, `mean_fidelity`, `upper_bound_kl`;
  uncertainty `expressibility_kl_std`.

## Entangling capability (`entanglement.py`)

**Definition (Sim et al. Eq. 21):** `Ent = mean over sampled theta of
Q(|psi_theta>)`, where Q is the Meyer-Wallach measure in the Brennen
single-qubit-purity form:

`Q(|psi>) = (2/n) * sum_{k=1}^{n} ( 1 - Tr[rho_k^2] )`

(the average single-qubit linear entropy). Range `[0, 1]`; `Q = 0` iff a
product state; invariant under single-qubit local unitaries and global
phase.

- **Interpretation caveat (do not call this "entanglement" unqualified):**
  Q is deliberately *undiscerning* about entanglement structure — a
  4-qubit GHZ state and a tensor product of two Bell pairs both give
  `Q = 1`. It measures average single-qubit mixedness, not every notion of
  multipartite entanglement.
- **One-qubit circuits:** `Q = 0` by construction (a single qubit cannot
  be entangled). The IR schema forbids `n_qubits < 2`, so this branch is
  defensive; the `meyer_wallach()` function itself handles `n=1`.
- **Uncertainty:** within-replicate standard error of the mean Q
  (`meyer_wallach_mean_standard_error`) plus between-replicate dispersion
  (`meyer_wallach_std_between_replicates`).
- **Metric reported:** `meyer_wallach_ent`.

## Gradient variance (`gradients.py`)

**Definition (McClean et al. 2018 barren-plateau convention):** the
variance, across `n_inits` random weight initializations, of the gradient
of a **fixed** cost observable with respect to the trainable weights, at
the fixed diagnostic input.

- **Fixed cost observable:** `<Z_0>` (Pauli-Z on qubit 0) — the same
  observable for every circuit, never chosen per-circuit.
- **Differentiation:** PennyLane `backprop` on `default.qubit`, exact.
  Verified against central finite differences in the test suite.
- **Statistics (raw and log10 both stored):** `tracked_param_variance`
  (variance of the first weight's gradient — McClean single-parameter
  convention), `aggregate_variance` (mean over weights of each weight's
  across-inits variance), `mean_abs_gradient` with a bootstrap CI.
- **Zero-parameter circuits:** reported as `NO_PARAMETERS` (a distinct,
  non-failure outcome; variances `None`).
- **Non-finite gradients:** each NaN/Inf init is counted and **excluded**
  from the statistics; if *every* init is non-finite the diagnostic is
  reported as failed.
- **Scientific caution:** a barren plateau is a claim about how gradient
  variance scales with **system size**. A single circuit's gradient
  variance, or a small smoke run, does **not** establish a barren plateau
  — see the master plan's research-integrity constraints.

## Result schema, cache, resume

`DiagnosticResult` (`results.py`) is typed, serializable, and carries **no
task test metric** (no such field exists). One result = one diagnostic on
one circuit under one config. `metrics`/`uncertainty` hold the
diagnostic-specific numbers; surrounding fields hold provenance, resource
usage, and failure metadata.

`DiagnosticStore` (`store.py`) is a SQLite store whose **cache identity**
is the tuple `(structural_hash, diagnostic_name, diagnostic_version,
config_hash, run_seed)` — a metric name alone is *not* identity, so two
diagnostics that differ in version or config are separate rows. It
supports atomic writes, failure persistence, resume-without-recompute, and
`IncompatibleDiagnosticResumeError` when a run's reproducibility-critical
config changes. `structural_hash` is the Phase 1 **syntactic** hash;
physically-equivalent-but-syntactically-different circuits are separate
entries (Phase 3 defines no physical-equivalence analysis).

## Resource / budget accounting

`DiagnosticResourceUsage` records parameter samples, states generated,
fidelity and gradient evaluations, backend executions, failed/completed
samples, wall-clock, and cache hits. Which of these consume *scientific
budget* in later experiments is a master-plan decision; Phase 3 records
the counts but does **not** change the master plan's proposal-evaluation
budget. Diagnostics must not be "free" for one arm and budgeted for
another.

## Separation from task evaluation

The diagnostics package never imports `llm_vqc.evaluation.final_test`
(verified by `tests/test_diagnostics_harness.py`) and takes no task/test
argument. Diagnostics never access task test labels, call the final-test
interface, modify splits, tune preprocessing, or make
architecture-selection decisions.

## Sample-count guidance

| Scale | n_pairs (expr) | n_samples (ent) | n_inits (grad) | n_replicates | Use |
|---|---|---|---|---|---|
| unit test | 50-100 | 50-100 | 10-20 | 1 | correctness only |
| smoke | 300 | 300 | 40 | 2 | pipeline exercise |
| **research** | **5000** | **5000** | **200** | **5** | Sim et al. scale; the only quantitatively interpretable setting |

`n_bins = 75` at all scales (Sim et al.). Smoke/unit outputs must never be
reported as scientific findings.

## Computational scaling

Each fidelity or Q sample is one `2^n`-amplitude statevector simulation;
expressibility is `O(n_pairs * n_replicates)` state preparations,
entanglement `O(n_samples * n_replicates)`, gradients `O(n_inits)`
backward passes. All exact (statevector), CPU-only, no finite-shot noise.
The smoke config (4 fixtures x 3 diagnostics) runs in ~10 s.

## Cross-backend / numerical validation

State preparation, fidelity, Meyer-Wallach, and the gradient cost
expectation are cross-checked against an independent Qiskit statevector
(`statevector_qiskit`, reconciling the little-endian/big-endian index
convention) up to global phase; gradients are cross-checked against
central finite differences. Tolerances (`~1e-9` for statevector
agreement, `~1e-7` for finite-difference gradients) reflect float64
backend behavior, not loose pass-the-test margins.

## Example

```python
from llm_vqc.diagnostics import compute_diagnostic, DiagnosticConfig

config = DiagnosticConfig()  # research-scale defaults (5000/5000/200, 75 bins)
result = compute_diagnostic(
    circuit_ir_or_dict, "expressibility", config, run_seed=0, proposal_id="demo"
)
print(result.metrics["expressibility_kl"], "+/-", result.uncertainty["expressibility_kl_std"])
```

## What Phase 3 does not do

- No search algorithm, no LLM integration (Phase 4/5).
- No claim that expressibility or entangling capability predicts task
  performance (requires evidence not gathered here).
- No barren-plateau claim from a single size or smoke run.
- No physical-equivalence-aware circuit deduplication (syntactic only).
- No finite-shot / hardware-noise diagnostics (statevector-exact only;
  the backend-sampling seed stream is reserved for a future extension).
