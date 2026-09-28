"""Adversarial / robustness tests for the diagnostics layer."""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.diagnostics.config import (
    DiagnosticConfig,
    EntanglementConfig,
    ExpressibilityConfig,
    GradientVarianceConfig,
)
from llm_vqc.diagnostics.entanglement import meyer_wallach
from llm_vqc.diagnostics.expressibility import kl_divergence_to_haar
from llm_vqc.diagnostics.harness import compute_diagnostic
from llm_vqc.diagnostics.results import DiagnosticFailureCategory
from llm_vqc.diagnostics.sampling import fidelity


def _cfg():
    return DiagnosticConfig(
        expressibility=ExpressibilityConfig(n_pairs=50, n_replicates=1),
        entanglement=EntanglementConfig(n_samples=50, n_replicates=1),
        gradient_variance=GradientVarianceConfig(n_inits=10, n_bootstrap=0),
    )


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"n_qubits": "three"},
        {
            "n_qubits": 3,
            "encoding": {"type": "angle"},
            "layers": [],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY", "wires": [99]},
            "layers": [],
            "measurements": {"observable": "Z", "wires": "all"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
            "layers": [{"type": "entangle", "pattern": "star", "gate": "CNOT"}],  # missing center
            "measurements": {"observable": "Z", "wires": "all"},
        },
    ],
)
@pytest.mark.parametrize("name", ["expressibility", "entanglement", "gradient_variance"])
def test_malformed_circuit_never_crashes_and_is_reported_invalid(raw, name):
    result = compute_diagnostic(raw, name, _cfg(), run_seed=0, proposal_id="p")
    assert result.failure_category == DiagnosticFailureCategory.INVALID_CIRCUIT
    assert result.structural_hash is None


def test_meyer_wallach_on_near_product_state_is_near_zero():
    eps = 1e-7
    state = np.array([np.sqrt(1 - eps**2), 0, 0, eps])  # almost |00>
    q = meyer_wallach(state, 2)
    assert 0.0 <= q < 1e-10


def test_meyer_wallach_clips_tiny_numerical_excursions_into_unit_interval():
    """Floating-point noise must never push Q outside [0, 1]."""
    # A genuinely maximally-entangled state may compute a purity of
    # 0.5 +/- 1e-16; the result must still be clamped to <= 1.
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    q = meyer_wallach(bell, 2)
    assert 0.0 <= q <= 1.0


def test_fidelity_of_identical_states_is_exactly_one():
    state = np.array([0.6, 0.8j, 0.0, 0.0])
    assert fidelity(state, state) == pytest.approx(1.0, abs=1e-12)


def test_fidelity_of_orthogonal_states_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert fidelity(a, b) == pytest.approx(0.0, abs=1e-12)


def test_kl_with_fidelities_all_in_the_final_near_one_bin():
    """Fidelities clustered at F ~ 1 (least expressive) give a large but
    finite KL — the Haar reference bin there has tiny but positive mass."""
    fidelities = np.full(300, 0.999)
    kl = kl_divergence_to_haar(fidelities, n_qubits=3, n_bins=75)
    assert np.isfinite(kl)
    assert kl > 0


def test_kl_handles_fidelities_exactly_at_bin_edges():
    fidelities = np.array([0.0, 0.5, 1.0] * 100)
    kl = kl_divergence_to_haar(fidelities, n_qubits=2, n_bins=75)
    assert np.isfinite(kl)
