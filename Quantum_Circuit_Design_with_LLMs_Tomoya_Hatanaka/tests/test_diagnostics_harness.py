"""Diagnostic harness tests: end-to-end dispatch, result schema, resource
accounting, invalid-circuit handling, CPU-only, and the structural
guarantee that diagnostics never touch task test data / final-test."""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from llm_vqc.diagnostics import compute_diagnostic
from llm_vqc.diagnostics.config import (
    DiagnosticConfig,
    EntanglementConfig,
    ExpressibilityConfig,
    GradientVarianceConfig,
)
from llm_vqc.diagnostics.results import DiagnosticFailureCategory, DiagnosticResult
from llm_vqc.diagnostics.store import DiagnosticStore

VALID = {
    "n_qubits": 3,
    "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
    "layers": [
        {"type": "rot", "gates": ["RX", "RY", "RZ"], "wires": "all"},
        {"type": "entangle", "pattern": "ring", "gate": "CNOT"},
    ],
    "measurements": {"observable": "Z", "wires": "all"},
}

INVALID = {  # missing encoding.gate for angle type
    "n_qubits": 3,
    "encoding": {"type": "angle"},
    "layers": [],
    "measurements": {"observable": "Z"},
}


def _small_config():
    return DiagnosticConfig(
        expressibility=ExpressibilityConfig(n_pairs=100, n_replicates=1),
        entanglement=EntanglementConfig(n_samples=100, n_replicates=1),
        gradient_variance=GradientVarianceConfig(n_inits=20, n_bootstrap=50),
    )


# --- Structural quarantine (the core guarantee) -------------------------


def test_diagnostics_package_never_imports_final_test():
    """No module in llm_vqc/diagnostics may import the final-test path."""
    import pathlib

    pkg = pathlib.Path("llm_vqc/diagnostics")
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text())
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets.append(node.module or "")
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names)
        assert not any("final_test" in t for t in targets), (path.name, targets)


def test_importing_diagnostics_does_not_pull_in_final_test():
    code = (
        "import llm_vqc.diagnostics.harness; import sys; "
        "print('llm_vqc.evaluation.final_test' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def test_diagnostic_result_has_no_test_metric_field():
    field_names = set(DiagnosticResult.model_fields.keys())
    assert not any("test" in name.lower() for name in field_names), field_names


def test_compute_diagnostic_signature_has_no_task_or_test_parameter():
    import inspect

    params = set(inspect.signature(compute_diagnostic).parameters.keys())
    assert not any("test" in p.lower() for p in params), params
    assert not any("task" in p.lower() for p in params), params


# --- Functional behavior ------------------------------------------------


@pytest.mark.parametrize("name", ["expressibility", "entanglement", "gradient_variance"])
def test_each_diagnostic_runs_end_to_end(name):
    result = compute_diagnostic(VALID, name, _small_config(), run_seed=0, proposal_id="p1")
    assert result.failure_category == DiagnosticFailureCategory.NONE
    assert result.structural_hash is not None
    assert result.diagnostic_name == name
    assert result.diagnostic_version
    assert result.config_hash
    assert result.samples_completed > 0
    assert result.resource_usage.wall_clock_seconds >= 0.0


def test_invalid_circuit_is_reported_without_crashing():
    result = compute_diagnostic(
        INVALID, "expressibility", _small_config(), run_seed=0, proposal_id="p"
    )
    assert result.failure_category == DiagnosticFailureCategory.INVALID_CIRCUIT
    assert result.structural_hash is None
    assert result.error_message


def test_diagnostic_result_serialization_is_stable():
    result = compute_diagnostic(VALID, "entanglement", _small_config(), run_seed=0, proposal_id="p")
    text1 = result.model_dump_json()
    text2 = result.model_dump_json()
    assert text1 == text2
    assert DiagnosticResult.model_validate_json(text1) == result


def test_results_are_deterministic_across_calls():
    a = compute_diagnostic(VALID, "expressibility", _small_config(), run_seed=0, proposal_id="p")
    b = compute_diagnostic(VALID, "expressibility", _small_config(), run_seed=0, proposal_id="q")
    assert a.metrics["expressibility_kl"] == b.metrics["expressibility_kl"]


def test_resource_accounting_records_sample_counts():
    cfg = _small_config()
    result = compute_diagnostic(VALID, "expressibility", cfg, run_seed=0, proposal_id="p")
    usage = result.resource_usage
    # 100 pairs * 1 replicate -> 100 fidelity evals, 200 states
    assert usage.fidelity_evaluations == 100
    assert usage.states_generated == 200
    assert usage.completed_samples == 100


def test_gradient_variance_on_parameterless_circuit_reports_no_parameters():
    parameterless = {
        "n_qubits": 2,
        "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
        "layers": [
            {"type": "rot", "gates": ["H"], "wires": "all"},
            {"type": "entangle", "pattern": "pairs", "gate": "CNOT", "pairs": [[0, 1]]},
        ],
        "measurements": {"observable": "Z", "wires": "all"},
    }
    result = compute_diagnostic(
        parameterless, "gradient_variance", _small_config(), run_seed=0, proposal_id="p"
    )
    assert result.failure_category == DiagnosticFailureCategory.NO_PARAMETERS
    assert result.metrics["n_parameters"] == 0.0


def test_store_cache_hit_avoids_recomputation(tmp_path):
    store = DiagnosticStore.open_or_create(
        tmp_path / "d.sqlite", run_id="r1", config_json="{}",
        config_reproducibility_fields={"c": 1}, git_sha=None, created_at="t0",
    )
    cfg = _small_config()
    r1 = compute_diagnostic(VALID, "expressibility", cfg, run_seed=0, proposal_id="p", store=store)
    r2 = compute_diagnostic(VALID, "expressibility", cfg, run_seed=0, proposal_id="q", store=store)
    assert r1.resource_usage.cache_hit is False
    assert r2.resource_usage.cache_hit is True
    assert r1.metrics["expressibility_kl"] == r2.metrics["expressibility_kl"]
    assert store.count() == 1
    store.close()


def test_cpu_only_execution():
    """Diagnostics run on default.qubit (CPU); the result records the
    backend and completes without any GPU requirement."""
    result = compute_diagnostic(
        VALID, "gradient_variance", _small_config(), run_seed=0, proposal_id="p"
    )
    assert result.backend == "default.qubit"
    assert result.diff_method == "backprop"
    assert result.failure_category == DiagnosticFailureCategory.NONE
