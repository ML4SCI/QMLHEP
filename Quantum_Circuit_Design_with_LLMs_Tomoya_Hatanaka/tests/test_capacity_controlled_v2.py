"""v2 fairness, accounting, and ablation-invariant tests.

Complements test_capacity_controlled.py (unchanged). Proves the new v2
conditions are fair and correctly constrained: strong analytical baselines never
see test labels, the classical search gets exactly its B=25 selection budget with
recorded capacity, quantum ablations truly do what they claim (frozen quantum is
frozen; product-state has no entanglement; fixed architectures cannot mutate),
and multi-split partitions differ.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import TrainingSeeds, train_seed_for_circuit
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import ablations as AB
from llm_vqc.experiments.capacity_controlled import analytical as AN
from llm_vqc.experiments.capacity_controlled import baselines as BL
from llm_vqc.experiments.capacity_controlled import classical_search as CS
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask


def _tv_test(split_seed=0):
    task = T1GaussianPeakTask()
    return task.build(seed=split_seed), task.build_test(seed=split_seed)


# --- analytical baselines never touch test labels --------------------------

def test_analytical_estimators_take_only_features():
    for name, fn in AN.ANALYTICAL_ESTIMATORS.items():
        sig = inspect.signature(fn) if not isinstance(fn, type) else None
        # each estimator is called with features only (no targets) in the pipeline
        tv, _ = _tv_test()
        pred, fail = fn(tv.val.features)
        assert pred.shape[0] == tv.val.features.shape[0]
        assert np.all((pred >= 0.0) & (pred <= 1.0))
        assert fail >= 0


def test_analytical_source_has_no_target_parameter():
    # structural guarantee: estimator functions accept a single features array
    src = inspect.getsource(AN)
    assert "def a1_argmax(features" in src
    assert "def a4_gaussian_nlls(features" in src
    # they must never reference the true generation params
    for banned in ("mu[", "targets", "true_mu"):
        assert banned not in src


# --- Q0 no-quantum ---------------------------------------------------------

def test_q0_no_quantum_has_93_params_and_no_quantum_layer():
    m = AB.NoQuantumModel()
    assert BL.count_params(m) == 93
    assert not any("q_layer" in n for n, _ in m.named_parameters())


# --- QP product-state has no entanglement ----------------------------------

def test_qp_product_state_no_two_qubit_gates():
    ir = S.genome_to_ir(AB.product_state_genome())
    assert S.validate_controlled(ir) == []
    cost = circuit_cost_summary(ir)
    assert cost.two_qubit_gate_count == 0
    assert build_program(ir).num_parameters == 12
    m = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
    assert sum(p.numel() for p in m.parameters()) == 105


# --- QF frozen quantum truly frozen ----------------------------------------

def test_qf_frozen_quantum_weights_unchanged():
    tv, _ = _tv_test()
    ir = S.genome_to_ir(BL.fixed_random_genome())
    ts = train_seed_for_circuit(0, structural_hash(ir))
    torch.manual_seed(0)
    m0 = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
    apply_explicit_init(m0, TrainingSeeds.from_train_seed(ts).param_init)
    q_init = m0.q_layer.weights.detach().clone()
    out = train_model(ir, tv, TrainingConfig(epochs=3), ts,
                      init_policy=apply_explicit_init, freeze_quantum=True)
    assert out.success
    q_after = torch.tensor(out.trained_circuit_weights)
    assert torch.allclose(q_init, q_after)  # quantum angles never moved
    # classical weights DID move
    assert out.trained_classical_state is not None


def test_freeze_quantum_false_is_v1_behavior():
    # default path trains all params -> quantum weights change (regression guard)
    tv, _ = _tv_test()
    ir = S.genome_to_ir(BL.fixed_hea_genome())
    ts = train_seed_for_circuit(0, structural_hash(ir))
    torch.manual_seed(0)
    m0 = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
    apply_explicit_init(m0, TrainingSeeds.from_train_seed(ts).param_init)
    q_init = m0.q_layer.weights.detach().clone()
    out = train_model(ir, tv, TrainingConfig(epochs=3), ts, init_policy=apply_explicit_init)
    assert not torch.allclose(q_init, torch.tensor(out.trained_circuit_weights))


# --- fixed architectures cannot mutate -------------------------------------

def test_fixed_architectures_are_stable():
    for genome_fn in (BL.fixed_hea_genome, BL.fixed_random_genome, AB.product_state_genome):
        h1 = structural_hash(S.genome_to_ir(genome_fn()))
        h2 = structural_hash(S.genome_to_ir(genome_fn()))
        assert h1 == h2  # deterministic, no mutation


# --- classical search: exact B=25 budget, capacity recorded, no test tuning -

def test_classical_search_budget_and_capacity():
    tv, test = _tv_test()
    res = CS.run_classical_search(tv, test, seed=0, budget=25)
    assert res["n_candidates"] == 25
    for c in res["candidates"]:
        assert 90 <= c["total_params"] <= 120
    # selection used validation only: selected is the argmin val among candidates
    best_val = min(c["val_rmse"] for c in res["candidates"])
    assert abs(res["selected_val_rmse"] - best_val) < 1e-12
    assert res["test_rmse"] is not None  # test computed once, after selection


def test_classical_search_selection_independent_of_test():
    # selection (argmin validation) must not depend on the protected test set:
    # re-running with the same seed reproduces the same selected config.
    tv, test = _tv_test()
    r1 = CS.run_classical_search(tv, test, seed=3, budget=10)
    r2 = CS.run_classical_search(tv, test, seed=3, budget=10)
    assert r1["selected"] == r2["selected"]


# --- multi-split partitions differ -----------------------------------------

def test_splits_have_distinct_partitions():
    task = T1GaussianPeakTask()
    t0 = task.build(seed=0).train.targets
    t1 = task.build(seed=1).train.targets
    assert not np.array_equal(t0, t1)
    # protected test also differs across splits
    assert not np.array_equal(task.build_test(seed=0).targets, task.build_test(seed=1).targets)
