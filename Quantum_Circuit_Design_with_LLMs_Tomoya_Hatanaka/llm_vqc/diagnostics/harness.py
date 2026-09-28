"""The diagnostic harness: the single entry point for measuring a circuit.

`compute_diagnostic(ir, diagnostic_name, config, run_seed, ...)` validates
the circuit, dispatches to the selected diagnostic, records resource usage
and provenance, wraps the numbers into a `DiagnosticResult`, and (if a
store is provided) caches it under the full diagnostic identity.

This module is separate from `llm_vqc.evaluation.harness` and, like it,
never imports `llm_vqc.evaluation.final_test` — diagnostics have no access
to task test data or the final-test path (verified by a test).
"""

from __future__ import annotations

import time
from typing import Literal

from llm_vqc.diagnostics.config import (
    ENTANGLEMENT_VERSION,
    EXPRESSIBILITY_VERSION,
    GRADIENT_VARIANCE_VERSION,
    DiagnosticConfig,
)
from llm_vqc.diagnostics.entanglement import compute_entangling_capability
from llm_vqc.diagnostics.expressibility import compute_expressibility
from llm_vqc.diagnostics.gradients import compute_gradient_variance
from llm_vqc.diagnostics.results import (
    DiagnosticFailureCategory,
    DiagnosticResourceUsage,
    DiagnosticResult,
)
from llm_vqc.diagnostics.seeds import (
    DiagnosticSeeds,
    diagnostic_base_seed,
    replicate_seeds,
)
from llm_vqc.diagnostics.store import DiagnosticStore, config_hash
from llm_vqc.ir.canonicalize import canonical_json
from llm_vqc.ir.canonicalize import structural_hash as compute_structural_hash
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.ir.validators import validate_proposal

DiagnosticName = Literal["expressibility", "entanglement", "gradient_variance"]

_VERSIONS: dict[str, str] = {
    "expressibility": EXPRESSIBILITY_VERSION,
    "entanglement": ENTANGLEMENT_VERSION,
    "gradient_variance": GRADIENT_VARIANCE_VERSION,
}


def _sub_config_dict(config: DiagnosticConfig, diagnostic_name: str) -> dict:
    """The reproducibility-critical config for one diagnostic (its own
    sub-config plus the shared backend settings)."""
    shared = {
        "backend": config.backend,
        "diff_method": config.diff_method,
        "dtype": config.dtype,
    }
    sub = getattr(config, diagnostic_name).model_dump()
    return {**shared, **sub}


def compute_diagnostic(
    raw_proposal: dict | CircuitIR,
    diagnostic_name: DiagnosticName,
    config: DiagnosticConfig,
    run_seed: int,
    proposal_id: str,
    store: DiagnosticStore | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> DiagnosticResult:
    """Measure one diagnostic on one circuit. Never raises on a
    computation failure — failures are reported in the returned
    `DiagnosticResult`, so a batch of measurements does not abort on one
    bad circuit."""
    version = _VERSIONS[diagnostic_name]
    cfg_dict = _sub_config_dict(config, diagnostic_name)
    cfg_hash = config_hash(cfg_dict)

    validation = validate_proposal(raw_proposal)
    if not validation.valid:
        return DiagnosticResult(
            structural_hash=None,
            circuit_canonical_json=None,
            diagnostic_name=diagnostic_name,
            diagnostic_version=version,
            config_hash=cfg_hash,
            config=cfg_dict,
            failure_category=DiagnosticFailureCategory.INVALID_CIRCUIT,
            error_message="; ".join(f"{i.code}@{i.path}" for i in validation.issues),
            run_seed=run_seed,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )

    ir = validation.ir
    ir_hash = compute_structural_hash(ir)
    base_seed = diagnostic_base_seed(run_seed, ir_hash, diagnostic_name)

    if store is not None:
        cached = store.get_cached(ir_hash, diagnostic_name, version, cfg_hash, run_seed)
        if cached is not None:
            cached = cached.model_copy(deep=True)
            cached.resource_usage.cache_hit = True
            return cached

    start = time.perf_counter()
    program = build_program(ir)

    dispatch = {
        "expressibility": _run_expressibility,
        "entanglement": _run_entanglement,
        "gradient_variance": _run_gradient_variance,
    }
    try:
        runner = dispatch[diagnostic_name]
        result = runner(ir, config, base_seed, program, cfg_dict, cfg_hash, version)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        result = DiagnosticResult(
            structural_hash=ir_hash,
            circuit_canonical_json=canonical_json(ir),
            diagnostic_name=diagnostic_name,
            diagnostic_version=version,
            config_hash=cfg_hash,
            config=cfg_dict,
            failure_category=DiagnosticFailureCategory.COMPUTATION_ERROR,
            error_message=f"{type(exc).__name__}: {exc}",
            run_seed=run_seed,
            base_seed=base_seed,
        )

    # Fill common provenance / accounting fields.
    result.structural_hash = ir_hash
    result.circuit_canonical_json = canonical_json(ir)
    result.run_seed = run_seed
    result.base_seed = base_seed
    result.backend = config.backend
    result.diff_method = config.diff_method
    result.git_sha = git_sha
    result.git_dirty = git_dirty
    result.resource_usage.wall_clock_seconds = time.perf_counter() - start

    if store is not None:
        store.put_cached(result, run_seed)

    return result


def _run_expressibility(ir, config, base_seed, program, cfg_dict, cfg_hash, version):
    ec = config.expressibility
    reps = replicate_seeds(base_seed, ec.n_replicates)
    outcome = compute_expressibility(ir, ec, reps, program)
    usage = DiagnosticResourceUsage(
        parameter_samples=2 * ec.n_pairs * ec.n_replicates,
        states_generated=2 * ec.n_pairs * ec.n_replicates,
        fidelity_evaluations=ec.n_pairs * ec.n_replicates,
        backend_executions=2 * ec.n_pairs * ec.n_replicates,
        completed_samples=outcome.n_pairs_completed,
    )
    return DiagnosticResult(
        structural_hash=None,
        circuit_canonical_json=None,
        diagnostic_name="expressibility",
        diagnostic_version=version,
        config_hash=cfg_hash,
        config=cfg_dict,
        samples_requested=ec.n_pairs * ec.n_replicates,
        samples_completed=outcome.n_pairs_completed,
        metrics={
            "expressibility_kl": outcome.expr_mean,
            "expressibility_kl_robustness_bins": outcome.robustness_expr_mean,
            "mean_fidelity": outcome.mean_fidelity,
            "upper_bound_kl": outcome.upper_bound,
        },
        uncertainty={
            "expressibility_kl_std": outcome.expr_std,
            "expressibility_kl_robustness_std": outcome.robustness_expr_std,
        },
        resource_usage=usage,
    )


def _run_entanglement(ir, config, base_seed, program, cfg_dict, cfg_hash, version):
    enc = config.entanglement
    reps = replicate_seeds(base_seed, enc.n_replicates)
    outcome = compute_entangling_capability(ir, enc, reps, program)
    usage = DiagnosticResourceUsage(
        parameter_samples=enc.n_samples * enc.n_replicates,
        states_generated=enc.n_samples * enc.n_replicates,
        backend_executions=enc.n_samples * enc.n_replicates,
        completed_samples=outcome.n_samples_completed,
    )
    return DiagnosticResult(
        structural_hash=None,
        circuit_canonical_json=None,
        diagnostic_name="entanglement",
        diagnostic_version=version,
        config_hash=cfg_hash,
        config=cfg_dict,
        samples_requested=enc.n_samples * enc.n_replicates,
        samples_completed=outcome.n_samples_completed,
        metrics={"meyer_wallach_ent": outcome.ent_mean},
        uncertainty={
            "meyer_wallach_std_between_replicates": outcome.ent_std_between_replicates,
            "meyer_wallach_mean_standard_error": outcome.mean_standard_error,
        },
        resource_usage=usage,
    )


def _run_gradient_variance(ir, config, base_seed, program, cfg_dict, cfg_hash, version):
    gc = config.gradient_variance
    seeds = DiagnosticSeeds.from_base_seed(base_seed)
    outcome = compute_gradient_variance(ir, gc, seeds, program)

    if outcome.n_parameters == 0:
        failure = DiagnosticFailureCategory.NO_PARAMETERS
    elif outcome.failed:
        failure = DiagnosticFailureCategory.ALL_GRADIENTS_NON_FINITE
    else:
        failure = DiagnosticFailureCategory.NONE

    warnings: list[str] = []
    if outcome.n_inits_non_finite > 0:
        warnings.append(
            f"{outcome.n_inits_non_finite} of {gc.n_inits} gradient evaluations were "
            "non-finite and were excluded from the statistics"
        )

    usage = DiagnosticResourceUsage(
        parameter_samples=gc.n_inits,
        gradient_evaluations=gc.n_inits,
        backend_executions=gc.n_inits,
        failed_samples=outcome.n_inits_non_finite,
        completed_samples=outcome.n_inits_completed,
    )
    return DiagnosticResult(
        structural_hash=None,
        circuit_canonical_json=None,
        diagnostic_name="gradient_variance",
        diagnostic_version=version,
        config_hash=cfg_hash,
        config=cfg_dict,
        samples_requested=gc.n_inits,
        samples_completed=outcome.n_inits_completed,
        metrics={
            "n_parameters": float(outcome.n_parameters),
            "tracked_param_variance": outcome.tracked_param_variance,
            "tracked_param_log10_variance": outcome.tracked_param_log10_variance,
            "aggregate_variance": outcome.aggregate_variance,
            "aggregate_log10_variance": outcome.aggregate_log10_variance,
            "mean_abs_gradient": outcome.mean_abs_gradient,
        },
        uncertainty={
            "mean_abs_gradient_ci_low": outcome.mean_abs_gradient_ci_low,
            "mean_abs_gradient_ci_high": outcome.mean_abs_gradient_ci_high,
        },
        numerical_warnings=warnings,
        failure_category=failure,
        error_message=outcome.error_message,
        resource_usage=usage,
    )
