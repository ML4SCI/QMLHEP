"""Gradient-variance / trainability diagnostic (McClean et al. 2018
"barren plateaus" convention).

For a fixed cost observable C, we sample `n_inits` trainable-weight
vectors uniformly on [0, 2*pi), compute the gradient of <C> with respect
to every trainable weight at each initialization, and report how the
gradient magnitude is distributed. A vanishing gradient variance that
shrinks with system size is the signature of a barren plateau (untrainable
circuit).

Fixed choices (identical across ALL circuits and future search arms — the
whole point of a fair diagnostic):

- **Cost observable:** <Z_0>, the Pauli-Z expectation on qubit 0. This is
  the canonical single-qubit-observable barren-plateau probe (McClean et
  al. 2018). It is the same observable for every circuit, never selected
  per-circuit (which could make a chosen circuit look more trainable).
- **Input:** the same fixed diagnostic input as the other diagnostics
  (zeros for angle encoding; a fixed seeded normalized vector for
  amplitude encoding) — never task test data.
- **Differentiation:** PennyLane `backprop` on `default.qubit`, exact
  (not finite-shot, not finite-difference). Verified against central
  finite differences on small fixtures in the test suite.
- **Parameter initialization:** uniform [0, 2*pi).
- **dtype:** float64.

Reported statistics (both raw and log10 stored, per the task brief):

- **tracked-parameter variance:** Var over inits of the gradient of the
  *first* trainable weight — the McClean convention of tracking a single
  parameter's gradient. (Which single parameter is arbitrary but fixed to
  index 0 for determinism; documented.)
- **aggregate variance:** the mean, over all trainable weights, of each
  weight's across-inits gradient variance — a size-robust summary.
- **mean absolute gradient** and its bootstrap CI.

Edge cases:

- **Zero trainable parameters:** there is no gradient to take. This is
  reported as a distinct, non-failure outcome (`n_parameters == 0`,
  variances `None`) — a parameterless circuit is trivially "flat" but it
  is not an error.
- **Non-finite gradients (NaN / Inf):** each such init is counted and
  EXCLUDED from the variance/mean statistics (they must never silently
  enter summary numbers). If *every* init is non-finite the diagnostic is
  reported as failed.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from llm_vqc.diagnostics.config import GradientVarianceConfig
from llm_vqc.diagnostics.sampling import diagnostic_input, sample_weight_vectors
from llm_vqc.diagnostics.seeds import DiagnosticSeeds
from llm_vqc.ir.compiler_pennylane import apply_program_ops
from llm_vqc.ir.expand import CircuitProgram, build_program
from llm_vqc.ir.schema import CircuitIR


def _cost_qnode(ir: CircuitIR, program: CircuitProgram) -> qml.QNode:
    dev = qml.device("default.qubit", wires=ir.n_qubits)

    def cost_circuit(inputs, weights):
        apply_program_ops(program, inputs, weights)
        return qml.expval(qml.PauliZ(0))  # fixed observable for every circuit

    return qml.QNode(cost_circuit, dev, diff_method="backprop")


class GradientVarianceOutcome:
    def __init__(
        self,
        n_parameters: int,
        tracked_param_variance: float | None,
        tracked_param_log10_variance: float | None,
        aggregate_variance: float | None,
        aggregate_log10_variance: float | None,
        mean_abs_gradient: float | None,
        mean_abs_gradient_ci_low: float | None,
        mean_abs_gradient_ci_high: float | None,
        n_inits_completed: int,
        n_inits_non_finite: int,
        failed: bool,
        error_message: str | None,
    ) -> None:
        self.n_parameters = n_parameters
        self.tracked_param_variance = tracked_param_variance
        self.tracked_param_log10_variance = tracked_param_log10_variance
        self.aggregate_variance = aggregate_variance
        self.aggregate_log10_variance = aggregate_log10_variance
        self.mean_abs_gradient = mean_abs_gradient
        self.mean_abs_gradient_ci_low = mean_abs_gradient_ci_low
        self.mean_abs_gradient_ci_high = mean_abs_gradient_ci_high
        self.n_inits_completed = n_inits_completed
        self.n_inits_non_finite = n_inits_non_finite
        self.failed = failed
        self.error_message = error_message


def _gradient_at(qnode: qml.QNode, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(inputs, dtype=torch.float64)
    cost = qnode(x, w)
    cost.backward()
    return w.grad.detach().numpy().copy()


def _bootstrap_ci(
    values: np.ndarray, n_bootstrap: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    if n_bootstrap == 0 or values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means[b] = np.mean(sample)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def compute_gradient_variance(
    ir: CircuitIR,
    config: GradientVarianceConfig,
    seeds: DiagnosticSeeds,
    program: CircuitProgram | None = None,
) -> GradientVarianceOutcome:
    program = program if program is not None else build_program(ir)

    if program.num_parameters == 0:
        # No trainable parameters: nothing to differentiate. Not a failure.
        return GradientVarianceOutcome(
            n_parameters=0,
            tracked_param_variance=None,
            tracked_param_log10_variance=None,
            aggregate_variance=None,
            aggregate_log10_variance=None,
            mean_abs_gradient=None,
            mean_abs_gradient_ci_low=None,
            mean_abs_gradient_ci_high=None,
            n_inits_completed=0,
            n_inits_non_finite=0,
            failed=False,
            error_message=None,
        )

    qnode = _cost_qnode(ir, program)
    inputs = diagnostic_input(program, seeds.diagnostic_input)
    init_vectors = sample_weight_vectors(
        program, config.n_inits, config.param_low, config.param_high, seeds.parameter_sampling
    )

    gradient_rows: list[np.ndarray] = []
    n_non_finite = 0
    for weights in init_vectors:
        grad = _gradient_at(qnode, inputs, weights)
        if not np.all(np.isfinite(grad)):
            n_non_finite += 1
            continue  # excluded from all statistics
        gradient_rows.append(grad)

    if not gradient_rows:
        return GradientVarianceOutcome(
            n_parameters=program.num_parameters,
            tracked_param_variance=None,
            tracked_param_log10_variance=None,
            aggregate_variance=None,
            aggregate_log10_variance=None,
            mean_abs_gradient=None,
            mean_abs_gradient_ci_low=None,
            mean_abs_gradient_ci_high=None,
            n_inits_completed=0,
            n_inits_non_finite=n_non_finite,
            failed=True,
            error_message=f"all {config.n_inits} gradient evaluations were non-finite",
        )

    gradients = np.vstack(gradient_rows)  # shape (n_completed, num_parameters)

    tracked_variance = float(np.var(gradients[:, 0]))
    per_param_variance = np.var(gradients, axis=0)
    aggregate_variance = float(np.mean(per_param_variance))
    abs_gradients = np.abs(gradients).reshape(-1)
    mean_abs = float(np.mean(abs_gradients))
    ci_low, ci_high = _bootstrap_ci(abs_gradients, config.n_bootstrap, seeds.bootstrap)

    def _safe_log10(x: float) -> float:
        return float(np.log10(x)) if x > 0 else float("-inf")

    return GradientVarianceOutcome(
        n_parameters=program.num_parameters,
        tracked_param_variance=tracked_variance,
        tracked_param_log10_variance=_safe_log10(tracked_variance),
        aggregate_variance=aggregate_variance,
        aggregate_log10_variance=_safe_log10(aggregate_variance),
        mean_abs_gradient=mean_abs,
        mean_abs_gradient_ci_low=ci_low,
        mean_abs_gradient_ci_high=ci_high,
        n_inits_completed=int(gradients.shape[0]),
        n_inits_non_finite=n_non_finite,
        failed=False,
        error_message=None,
    )
