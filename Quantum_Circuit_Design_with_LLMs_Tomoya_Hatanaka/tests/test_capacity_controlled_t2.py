"""T2-v1 fairness / capacity / quarantine / ablation-invariant tests."""

from __future__ import annotations

import numpy as np
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.seeds import TrainingSeeds, train_seed_for_circuit
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled import t2_ablations as AB
from llm_vqc.experiments.capacity_controlled import t2_baselines as CB
from llm_vqc.experiments.capacity_controlled import t2_task as T2
from llm_vqc.experiments.capacity_controlled.arms import ControlledRandomArm
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR, EncodingSpec
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t2_digits import T2DigitsTask, _load_raw_binary_digits


# --- task: stratified, disjoint, deterministic, no preprocessing leakage ----

def test_t2_splits_stratified_disjoint_deterministic():
    task = T2DigitsTask()
    for s in (0, 1, 2, 3, 4):
        tv = task.build(seed=s); test = task.build_test(seed=s)
        tr, va, te = set(tv.train.sample_ids), set(tv.val.sample_ids), set(test.sample_ids)
        assert not (tr & va) and not (tr & te) and not (va & te)  # disjoint
        for tg in (tv.train.targets, tv.val.targets, test.targets):
            assert 0.4 < tg.mean() < 0.6  # stratified/balanced
    # deterministic
    a = T2DigitsTask().build(seed=1).train.sample_ids
    b = T2DigitsTask().build(seed=1).train.sample_ids
    assert a == b


def test_t2_preprocessing_fitted_on_train_only():
    # test features must be produced by the TRAIN-fitted preprocessing (no leakage)
    task = T2DigitsTask()
    tv = task.build(seed=0); test = task.build_test(seed=0)
    pixels, labels, orig = _load_raw_binary_digits()
    # map test sample ids back to raw pixels and transform via the train-fit preprocessing
    id_to_row = {f"T2-test-orig{orig[i]}": i for i in range(len(orig))}
    rows = [id_to_row[sid] for sid in test.sample_ids]
    reconstructed = tv.preprocessing.transform(pixels[rows])
    assert np.allclose(reconstructed, test.features)  # identical => train-fit prep, no leak


# --- capacity + grammar ----------------------------------------------------

def test_t2_capacity_is_61_for_all_controlled_candidates():
    rng = np.random.default_rng(7)
    for _ in range(200):
        ir = S.genome_to_ir(S.random_genome(rng))
        assert S.validate_controlled(ir) == []
        m = HybridQNNModel(ir, raw_feature_dim=T2.RAW_FEATURE_DIM, head_out_dim=1)
        named = dict(m.named_parameters())
        assert sum(p.numel() for n, p in named.items() if n.startswith("embed.")) == 44
        assert sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")) == 12
        assert sum(p.numel() for n, p in named.items() if n.startswith("head.")) == 5
        assert sum(p.numel() for p in m.parameters()) == T2.TOTAL_TRAINABLE_PARAMS == 61


def test_t2_amplitude_rejected():
    ir = S.genome_to_ir(AB.predeclared_architectures()[0]).model_copy(
        update={"encoding": EncodingSpec(type="amplitude", wires="all")})
    assert any(i.code == "controlled.encoding_type" for i in S.validate_controlled(ir))


def test_t2_selection_metric_is_logloss():
    tv = T2.build_t2(0)
    assert tv.spec.metric_name == "logloss"
    assert tv.spec.lower_is_better is True


# --- search: exact budget, capacity, quarantine, resume --------------------

def _run(arm, store, run_id, budget, seed, tv):
    runner = SearchRunner(arm, "T2", tv, TrainingConfig(epochs=2), budget_limit=budget,
                          run_seed=seed, result_store=store, run_id=run_id,
                          extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
    return runner.run()


def test_t2_search_budget_capacity_and_selection(tmp_path):
    tv = T2.build_t2(0)
    store = ResultStore(tmp_path / "s.sqlite")
    res = _run(ControlledRandomArm(lower_is_better=True), store, "r0", 5, 0, tv)
    assert res.ledger_summary["consumed_budget"] == 5
    for e in store.all_evaluations("T2"):
        ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
        assert S.validate_controlled(ir) == []
        assert T2.total_trainable_params(ir) == 61
    store.close()


def test_t2_resume_deterministic(tmp_path):
    tv = T2.build_t2(0)
    sa = ResultStore(tmp_path / "a.sqlite")
    ra = _run(ControlledRandomArm(lower_is_better=True), sa, "run", 5, 1, tv); sa.close()
    sb = ResultStore(tmp_path / "b.sqlite")
    _run(ControlledRandomArm(lower_is_better=True), sb, "run", 3, 1, tv); sb.close()
    sb2 = ResultStore(tmp_path / "b.sqlite")
    rb = _run(ControlledRandomArm(lower_is_better=True), sb2, "run", 5, 1, tv); sb2.close()
    assert ra.selected_structural_hash == rb.selected_structural_hash
    assert ra.ledger_summary == rb.ledger_summary


def test_t2_no_test_in_search_feedback():
    from llm_vqc.search.feedback import SearchFeedback
    from llm_vqc.tasks.base import TrainValData
    assert not any("test" in f for f in SearchFeedback.model_fields)
    assert not hasattr(TrainValData, "test")


# --- paired ablations invariants -------------------------------------------

def test_paired_frozen_trainable_identical_initial_quantum_weights():
    tv = T2.build_t2(0)
    ir = S.genome_to_ir(AB.predeclared_architectures()[3])
    ts = train_seed_for_circuit(0, structural_hash(ir))
    tr = train_model(ir, tv, TrainingConfig(epochs=2), ts, init_policy=apply_explicit_init, freeze_quantum=False)
    fr = train_model(ir, tv, TrainingConfig(epochs=2), ts, init_policy=apply_explicit_init, freeze_quantum=True)
    # frozen quantum weights == the shared initial quantum weights
    torch.manual_seed(0)
    m0 = HybridQNNModel(ir, raw_feature_dim=10, head_out_dim=1)
    apply_explicit_init(m0, TrainingSeeds.from_train_seed(ts).param_init)
    q_init = m0.q_layer.weights.detach()
    assert torch.allclose(q_init, torch.tensor(fr.trained_circuit_weights))  # frozen unchanged
    assert not torch.allclose(q_init, torch.tensor(tr.trained_circuit_weights))  # trainable moved


def test_product_counterpart_has_no_entangling_gates_and_preserves_params():
    for e, p in AB.entangled_pairs(20):
        ire, irp = S.genome_to_ir(e), S.genome_to_ir(p)
        assert circuit_cost_summary(ire).two_qubit_gate_count >= 1
        assert circuit_cost_summary(irp).two_qubit_gate_count == 0
        assert S.validate_controlled(irp) == []
        assert T2.total_trainable_params(ire) == T2.total_trainable_params(irp) == 61


def test_predeclared_architectures_stable():
    a = [structural_hash(S.genome_to_ir(g)) for g in AB.predeclared_architectures()]
    b = [structural_hash(S.genome_to_ir(g)) for g in AB.predeclared_architectures()]
    assert a == b and len(a) == 20


# --- classical baselines: validation-only, deterministic -------------------

def test_c6_classical_search_uses_validation_only():
    tv = T2.build_t2(0); test = T2.build_t2_test(0)
    r1 = CB.c6_fair_nas(tv, test, seed=2, budget=8)
    r2 = CB.c6_fair_nas(tv, test, seed=2, budget=8)
    assert r1["selected_config"] == r2["selected_config"]  # reproducible, test-independent
    best_val = min(c["val_bae"] for c in r1["candidates"])
    assert abs(r1["val_bae"] - best_val) < 1e-12  # selected = argmin validation
    assert r1["n_candidates"] == 8


def test_c0_trivial_is_chance_level():
    tv = T2.build_t2(0); test = T2.build_t2_test(0)
    r = CB.c0_trivial(tv, test)
    assert abs(r["balanced_accuracy_error"] - 0.5) < 1e-9
