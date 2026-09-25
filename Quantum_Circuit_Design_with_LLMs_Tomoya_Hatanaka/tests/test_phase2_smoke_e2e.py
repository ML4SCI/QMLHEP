"""End-to-end Phase 2 smoke tests: T1 and T2 through the full
IR -> compiler -> training -> validation -> (quarantined) final-test
pipeline, CPU-only. Kept lightweight per instructions -- small epoch
counts, no publication-scale training.
"""

from __future__ import annotations

from llm_vqc.evaluation.final_test import evaluate_on_test
from llm_vqc.evaluation.harness import evaluate_candidate
from llm_vqc.evaluation.seeds import train_seed_for_circuit
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.ir.budget import BudgetLedger
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.validators import validate_proposal
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask
from llm_vqc.tasks.t2_digits import T2DigitsTask

SMALL_CIRCUIT = {
    "n_qubits": 4,
    "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
    "layers": [
        {"type": "rot", "gates": ["RY", "RZ"], "wires": "all"},
        {"type": "entangle", "pattern": "ring", "gate": "CNOT"},
    ],
    "measurements": {"observable": "Z", "wires": "all"},
}


def test_cpu_only_execution():
    ir = validate_proposal(SMALL_CIRCUIT).ir
    task = T1GaussianPeakTask()
    data = task.build(seed=0)
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    output = train_model(ir, data, TrainingConfig(epochs=2, device="cpu"), train_seed)
    assert output.success
    # Every returned weight value must be a plain Python float (already
    # detached to CPU by _state_dict_to_lists), never a CUDA tensor.
    for value in output.trained_circuit_weights:
        assert isinstance(value, float)


def test_t1_end_to_end_smoke_through_full_pipeline():
    task = T1GaussianPeakTask()
    train_val = task.build(seed=0)
    test_split = task.build_test(seed=0)  # quarantined, obtained separately
    ledger = BudgetLedger()
    config = TrainingConfig(epochs=10)

    search_result = evaluate_candidate(
        SMALL_CIRCUIT, "T1", run_seed=0, train_val=train_val, training_config=config,
        proposal_id="smoke-1", ledger=ledger,
    )
    assert search_result.training_outcome.value == "success"
    # Knipfer-magnitude RMSE ballpark (~0.02-0.06), per Phase 2 acceptance
    # criterion; not a strict scientific claim, just a sanity range.
    assert 0.0 < search_result.val_metric_value < 0.5

    ir = validate_proposal(SMALL_CIRCUIT).ir
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    training_output = train_model(ir, train_val, config, train_seed)
    final = evaluate_on_test(
        ir, training_output.trained_classical_state, test_split, task.spec, train_seed
    )
    assert final.n_test_samples == 2000
    assert 0.0 < final.test_metric_value < 0.5


def test_t2_end_to_end_smoke_through_full_pipeline():
    task = T2DigitsTask()
    train_val = task.build(seed=0)
    test_split = task.build_test(seed=0)
    ledger = BudgetLedger()
    config = TrainingConfig(epochs=10)

    search_result = evaluate_candidate(
        SMALL_CIRCUIT, "T2", run_seed=0, train_val=train_val, training_config=config,
        proposal_id="smoke-1", ledger=ledger,
    )
    assert search_result.training_outcome.value == "success"
    assert 0.0 <= search_result.val_metric_value <= 1.0  # AUC range

    ir = validate_proposal(SMALL_CIRCUIT).ir
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    training_output = train_model(ir, train_val, config, train_seed)
    final = evaluate_on_test(
        ir, training_output.trained_classical_state, test_split, task.spec, train_seed
    )
    assert final.test_metric_name == "auc"
    assert 0.0 <= final.test_metric_value <= 1.0


def test_search_visible_val_metric_differs_from_the_independently_computed_test_metric():
    """End-to-end sanity check that validation and test are genuinely
    different partitions scored independently: the val metric (computed
    during training on the val split) and the test metric (computed once,
    separately, via evaluate_on_test on the disjoint test split) need not
    match numerically, because they are different data. If the harness
    were accidentally leaking test data into validation, these would tend
    to become suspiciously identical across repeated runs; this is a
    coarse sanity signal on top of the structural guarantees in
    test_evaluation_harness.py and test_evaluation_results.py."""
    task = T1GaussianPeakTask()
    train_val = task.build(seed=0)
    test_split = task.build_test(seed=0)
    ledger = BudgetLedger()
    config = TrainingConfig(epochs=10)

    search_result = evaluate_candidate(
        SMALL_CIRCUIT, "T1", run_seed=0, train_val=train_val, training_config=config,
        proposal_id="smoke-1", ledger=ledger,
    )

    ir = validate_proposal(SMALL_CIRCUIT).ir
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    training_output = train_model(ir, train_val, config, train_seed)
    final = evaluate_on_test(
        ir, training_output.trained_classical_state, test_split, task.spec, train_seed
    )

    assert search_result.val_metric_value != final.test_metric_value
