"""Tests for the capacity-controlled experiment (T1_capacity_controlled_v1).

Proves the scientific invariants the experiment depends on: fixed qubits /
encoding / measurement, *exactly* 12 quantum params and 105 total trainable
params for every candidate, equal classical capacity, deterministic explicit
init, transparent rejection (no silent repair), budget equality, resume
equivalence, and structural test-quarantine.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.experiments.capacity_controlled import baselines as B
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.arms import (
    ControlledEvolutionaryArm,
    ControlledGreedyArm,
    ControlledRandomArm,
)
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

RNG = np.random.default_rng(12345)


def _total_params(ir: CircuitIR) -> int:
    m = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
    return sum(p.numel() for p in m.parameters())


# --- Invariants of every generated candidate --------------------------------

def test_random_genomes_are_all_valid_and_exact_capacity():
    for _ in range(500):
        ir = S.genome_to_ir(S.random_genome(RNG))
        assert S.validate_controlled(ir) == []
        assert _total_params(ir) == S.TOTAL_TRAINABLE_PARAMS == 105


def test_equal_classical_and_quantum_capacity_across_candidates():
    embeds, heads, quants = set(), set(), set()
    for _ in range(200):
        ir = S.genome_to_ir(S.random_genome(RNG))
        m = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
        named = dict(m.named_parameters())
        embeds.add(sum(p.numel() for n, p in named.items() if n.startswith("embed.")))
        heads.add(sum(p.numel() for n, p in named.items() if n.startswith("head.")))
        quants.add(sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")))
        assert m.embed.in_features == 21 and m.embed.out_features == 4
        assert m.head.in_features == 4 and m.head.out_features == 1
    assert embeds == {88}
    assert heads == {5}
    assert quants == {12}


def test_neighbors_and_crossover_preserve_invariants():
    g = S.deterministic_seed_genome()
    assert S.validate_controlled(S.genome_to_ir(g)) == []
    for _ in range(50):
        for nb in S.neighbors(g, RNG):
            assert S.validate_controlled(S.genome_to_ir(nb)) == []
        g = S.sample_neighbor(g, RNG)
    for _ in range(200):
        child = S.crossover(S.random_genome(RNG), S.random_genome(RNG), RNG)
        ir = S.genome_to_ir(child)
        assert S.validate_controlled(ir) == []
        assert _total_params(ir) == 105


# --- The validator rejects each forbidden deviation -------------------------

def _valid_ir() -> CircuitIR:
    return S.genome_to_ir(B.fixed_hea_genome())


def _codes(ir: CircuitIR) -> set[str]:
    return {i.code for i in S.validate_controlled(ir)}


def test_rejects_amplitude_encoding():
    ir = _valid_ir().model_copy(update={"encoding": EncodingSpec(type="amplitude", wires="all")})
    assert "controlled.encoding_type" in _codes(ir)


def test_rejects_wrong_qubit_count():
    g = B.fixed_hea_genome()
    ir = S.genome_to_ir(g).model_copy(update={"n_qubits": 5})
    assert "controlled.n_qubits" in _codes(ir)


def test_rejects_different_encoding_wires():
    ir = _valid_ir().model_copy(
        update={"encoding": EncodingSpec(type="angle", gate="RY", wires=[0, 1])})
    assert "controlled.encoding_wires" in _codes(ir) or "controlled.encoding_width" in _codes(ir)


def test_rejects_different_encoding_gate():
    ir = _valid_ir().model_copy(
        update={"encoding": EncodingSpec(type="angle", gate="RX", wires="all")})
    assert "controlled.encoding_gate" in _codes(ir)


def test_rejects_different_measurement_observable():
    ir = _valid_ir().model_copy(update={"measurements": MeasurementSpec(observable="X", wires="all")})
    assert "controlled.measurement_obs" in _codes(ir)


def test_rejects_different_measurement_wires():
    ir = _valid_ir().model_copy(update={"measurements": MeasurementSpec(observable="Z", wires=[0, 1])})
    assert "controlled.measurement_wires" in _codes(ir) or "controlled.measurement_width" in _codes(ir)


def test_rejects_too_few_quantum_params():
    ir = CircuitIR(
        n_qubits=4, encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RY"], wires="all")],  # only 4 params
        measurements=MeasurementSpec(observable="Z", wires="all"))
    assert "controlled.param_count" in _codes(ir)


def test_rejects_too_many_quantum_params():
    ir = CircuitIR(
        n_qubits=4, encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RY"], wires="all") for _ in range(4)],  # 16 params
        measurements=MeasurementSpec(observable="Z", wires="all"))
    assert "controlled.param_count" in _codes(ir)


def test_rejects_excessive_depth_gates_two_qubit():
    # 3 CRZ rings (12 params, ok) + many free CNOT rings -> blows structural limits
    layers = [EntangleLayer(pattern="ring", gate="CRZ", wires="all") for _ in range(3)]
    layers += [EntangleLayer(pattern="ring", gate="CNOT", wires="all") for _ in range(10)]
    ir = CircuitIR(
        n_qubits=4, encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=layers, measurements=MeasurementSpec(observable="Z", wires="all"))
    codes = _codes(ir)
    assert "controlled.too_many_blocks" in codes
    assert codes & {"controlled.depth", "controlled.gate_count", "controlled.two_qubit"}


def test_valid_controlled_circuit_passes():
    assert S.validate_controlled(_valid_ir()) == []


# --- Explicit initialization ------------------------------------------------

def test_explicit_init_is_deterministic_and_in_range():
    ir = _valid_ir()

    def snap(seed):
        torch.manual_seed(seed)
        m = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
        apply_explicit_init(m, seed)
        return m

    m1, m2, m3 = snap(3), snap(3), snap(99)
    v1 = torch.cat([p.detach().flatten() for p in m1.parameters()])
    v2 = torch.cat([p.detach().flatten() for p in m2.parameters()])
    v3 = torch.cat([p.detach().flatten() for p in m3.parameters()])
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, v3)
    q = m1.q_layer.weights.detach()
    assert q.min() >= -np.pi and q.max() <= np.pi


# --- Baselines --------------------------------------------------------------

def test_baseline_param_counts():
    assert B.count_params(B.BottleneckModel()) == 93
    assert B.count_params(B.MLPBaseline()) == 107  # mismatch +2 vs 105, recorded
    for g in (B.fixed_hea_genome(), B.fixed_random_genome()):
        ir = S.genome_to_ir(g)
        assert S.validate_controlled(ir) == []
        assert _total_params(ir) == 105


# --- Search integration: budget, capacity, quarantine, resume ---------------

def _run(arm, store, run_id, budget, seed):
    task = T1GaussianPeakTask()
    tv = task.build(seed=0)
    cfg = TrainingConfig(epochs=1)
    runner = SearchRunner(
        arm, "T1", tv, cfg, budget_limit=budget, run_seed=seed,
        result_store=store, run_id=run_id,
        extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
    return runner.run()


def test_search_consumes_exact_budget_and_uniform_capacity(tmp_path):
    store = ResultStore(tmp_path / "s.sqlite")
    res = _run(ControlledRandomArm(lower_is_better=True), store, "cr_s0", budget=5, seed=0)
    assert res.ledger_summary["consumed_budget"] == 5
    # every evaluated circuit has identical controlled capacity
    totals = set()
    for e in store.all_evaluations("T1"):
        ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
        assert S.validate_controlled(ir) == []
        totals.add(_total_params(ir))
    assert totals == {105}
    store.close()


def test_all_three_controlled_arms_run(tmp_path):
    for arm in (ControlledRandomArm(lower_is_better=True),
                ControlledGreedyArm(lower_is_better=True, k=3),
                ControlledEvolutionaryArm(lower_is_better=True)):
        store = ResultStore(tmp_path / f"{arm.name}.sqlite")
        res = _run(arm, store, f"{arm.name}_s0", budget=5, seed=0)
        assert res.ledger_summary["consumed_budget"] == 5
        assert res.selected_structural_hash is not None
        store.close()


def test_resume_equivalence(tmp_path):
    # direct B=5
    store_a = ResultStore(tmp_path / "a.sqlite")
    res_a = _run(ControlledRandomArm(lower_is_better=True), store_a, "run", budget=5, seed=1)
    store_a.close()
    # B=3 then resume to B=5 in a fresh store
    store_b = ResultStore(tmp_path / "b.sqlite")
    _run(ControlledRandomArm(lower_is_better=True), store_b, "run", budget=3, seed=1)
    store_b.close()
    store_b2 = ResultStore(tmp_path / "b.sqlite")
    res_b = _run(ControlledRandomArm(lower_is_better=True), store_b2, "run", budget=5, seed=1)
    store_b2.close()
    assert res_a.selected_structural_hash == res_b.selected_structural_hash
    assert res_a.ledger_summary == res_b.ledger_summary


def test_no_test_feedback_and_quarantine():
    import ast

    from llm_vqc.search.feedback import SearchFeedback
    from llm_vqc.tasks.base import TrainValData
    assert not any("test" in f for f in SearchFeedback.model_fields)
    assert not hasattr(TrainValData, "test")
    # The search runner must not IMPORT the quarantined final-test module
    # (docstrings may mention it; imports must not reference it).
    tree = ast.parse(Path(inspect.getfile(SearchRunner)).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("final_test" in m for m in imported)
    import llm_vqc.search.runner as runner_mod
    assert not hasattr(runner_mod, "evaluate_on_test")


def test_controlled_invalid_recorded_transparently(tmp_path):
    # an arm that emits an out-of-space (amplitude) proposal is recorded INVALID,
    # not silently repaired, and does not consume budget.
    from llm_vqc.evaluation.harness import evaluate_candidate
    from llm_vqc.ir.budget import BudgetLedger, ProposalOutcome
    task = T1GaussianPeakTask()
    tv = task.build(seed=0)
    ledger = BudgetLedger()
    bad = CircuitIR(
        n_qubits=4, encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=[RotationLayer(gates=["RY"], wires="all") for _ in range(3)],
        measurements=MeasurementSpec(observable="Z", wires="all")).model_dump()
    result = evaluate_candidate(
        bad, "T1", 0, tv, TrainingConfig(epochs=1), "p0", ledger,
        extra_validator=S.validate_controlled)
    assert result.validation_outcome.value == "invalid"
    assert any(i.code == "controlled.encoding_type" for i in result.validation_issues)
    assert ledger.summary()["consumed_budget"] == 0  # INVALID is budget-free
