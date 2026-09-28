"""Meyer-Wallach diagnostic tests: analytic states, qubit-ordering,
global-phase invariance, circuit-level entangling capability."""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.diagnostics.config import EntanglementConfig
from llm_vqc.diagnostics.entanglement import compute_entangling_capability, meyer_wallach
from llm_vqc.diagnostics.seeds import diagnostic_base_seed, replicate_seeds
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)

_MEAS = MeasurementSpec(observable="Z", wires="all")
_ENC = EncodingSpec(type="angle", gate="RY", wires="all")


def _bell():
    return np.array([1, 0, 0, 1]) / np.sqrt(2)


def test_meyer_wallach_bell_state_is_one():
    assert meyer_wallach(_bell(), 2) == pytest.approx(1.0, abs=1e-9)


def test_meyer_wallach_product_state_is_zero():
    assert meyer_wallach(np.array([0, 1, 0, 0]), 2) == pytest.approx(0.0, abs=1e-12)


def test_meyer_wallach_arbitrary_separable_product_is_zero():
    q0 = np.array([np.cos(0.6), np.sin(0.6)])
    q1 = np.array([np.cos(0.3), np.sin(0.3)])
    assert meyer_wallach(np.kron(q0, q1), 2) == pytest.approx(0.0, abs=1e-9)


def test_meyer_wallach_ghz_is_one():
    ghz = np.zeros(8)
    ghz[0] = ghz[7] = 1 / np.sqrt(2)
    assert meyer_wallach(ghz, 3) == pytest.approx(1.0, abs=1e-9)


def test_meyer_wallach_is_undiscerning_bell_pairs_equal_ghz():
    """Sim et al.: a tensor product of two Bell pairs and the 4-qubit GHZ
    both give Q = 1 (documented limitation of MW)."""
    two_bell = np.kron(_bell(), _bell())
    ghz4 = np.zeros(16)
    ghz4[0] = ghz4[15] = 1 / np.sqrt(2)
    assert meyer_wallach(two_bell, 4) == pytest.approx(1.0, abs=1e-9)
    assert meyer_wallach(ghz4, 4) == pytest.approx(1.0, abs=1e-9)


def test_meyer_wallach_partially_entangled_is_between_zero_and_one():
    pe = np.array([np.cos(0.4), 0, 0, np.sin(0.4)])
    q = meyer_wallach(pe, 2)
    assert 0.0 < q < 1.0


def test_meyer_wallach_is_global_phase_invariant():
    state = _bell()
    phased = np.exp(1j * 0.9) * state
    assert meyer_wallach(phased, 2) == pytest.approx(meyer_wallach(state, 2), abs=1e-12)


def test_meyer_wallach_qubit_ordering_asymmetric_state():
    """|01> vs |10> are both product (Q=0), but an asymmetric entangled
    state exercises the per-qubit reduced-state ordering. Entangling only
    qubits 0,1 of a 3-qubit register (qubit 2 idle) must leave qubit 2
    pure -> its linear entropy contributes 0."""
    # (|00>+|11>)/sqrt2 on qubits 0,1; qubit 2 = |0>
    bell01 = _bell()
    state = np.kron(bell01, np.array([1.0, 0.0]))  # qubit2 last (least significant)
    # qubits 0 and 1 maximally mixed (entropy 1/2 each), qubit 2 pure (0)
    # Q = (2/3)(0.5 + 0.5 + 0) = 2/3
    assert meyer_wallach(state, 3) == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_one_qubit_measure_returns_zero():
    """A single qubit cannot be entangled; Q=0 by construction."""
    assert meyer_wallach(np.array([np.cos(0.7), np.sin(0.7)]), 1) == 0.0


def _ent(ir, n_samples=300, n_rep=2):
    cfg = EntanglementConfig(n_samples=n_samples, n_replicates=n_rep)
    reps = replicate_seeds(diagnostic_base_seed(0, structural_hash(ir), "entanglement"), n_rep)
    return compute_entangling_capability(ir, cfg, reps)


def test_no_entangling_gates_gives_zero_capability():
    ir = CircuitIR(
        n_qubits=3, encoding=_ENC,
        layers=[RotationLayer(gates=["RX", "RY"], wires="all")], measurements=_MEAS,
    )
    outcome = _ent(ir)
    assert outcome.ent_mean == pytest.approx(0.0, abs=1e-9)


def test_entangling_circuit_gives_positive_capability_with_uncertainty():
    ir = CircuitIR(
        n_qubits=3, encoding=_ENC,
        layers=[
            RotationLayer(gates=["RX", "RY"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
            RotationLayer(gates=["RX"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=_MEAS,
    )
    outcome = _ent(ir)
    assert 0.0 < outcome.ent_mean <= 1.0
    assert outcome.mean_standard_error >= 0.0
    assert outcome.ent_std_between_replicates >= 0.0


def test_entangling_capability_is_deterministic_given_seed():
    ir = CircuitIR(
        n_qubits=3, encoding=_ENC,
        layers=[
            RotationLayer(gates=["RX"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=_MEAS,
    )
    assert _ent(ir).ent_mean == _ent(ir).ent_mean
