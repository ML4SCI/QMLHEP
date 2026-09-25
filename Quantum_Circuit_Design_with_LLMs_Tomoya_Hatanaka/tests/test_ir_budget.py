"""Tests for the Phase 1 budget-accounting data structures."""

from __future__ import annotations

from llm_vqc.ir.budget import BudgetLedger, ProposalOutcome, ProposalRecord
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec, RotationLayer


def _ir(n_qubits=3) -> CircuitIR:
    return CircuitIR(
        n_qubits=n_qubits,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


def test_empty_ledger_reports_zero_everything():
    ledger = BudgetLedger()
    assert ledger.summary() == {
        "num_proposed": 0,
        "num_valid": 0,
        "num_invalid": 0,
        "num_duplicate": 0,
        "num_failed": 0,
        "num_unique": 0,
        "consumed_budget": 0,
    }


def test_ledger_counts_valid_invalid_duplicate_and_failed_separately():
    ledger = BudgetLedger()
    ir_a = _ir(3)
    ir_b = _ir(4)
    h_a = structural_hash(ir_a)
    h_b = structural_hash(ir_b)

    ledger.record_proposal(
        ProposalRecord(
            outcome=ProposalOutcome.VALID, structural_hash=h_a, cost=circuit_cost_summary(ir_a)
        )
    )
    ledger.record_proposal(
        ProposalRecord(
            outcome=ProposalOutcome.VALID, structural_hash=h_b, cost=circuit_cost_summary(ir_b)
        )
    )
    ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=h_a))
    ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.INVALID))
    ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=h_b))

    assert ledger.num_proposed == 5
    assert ledger.num_valid == 2
    assert ledger.num_invalid == 1
    assert ledger.num_duplicate == 1
    assert ledger.num_failed == 1
    assert ledger.num_unique == 2  # h_a and h_b are the only distinct hashes seen
    # consumed_budget = everything except INVALID: 2 valid + 1 duplicate + 1 failed
    assert ledger.consumed_budget == 4
    assert ledger.remaining_budget(budget_limit=4) == 0
    assert ledger.is_exhausted(budget_limit=4)
    assert not ledger.is_exhausted(budget_limit=5)


def test_ledger_is_a_pure_recorder_not_an_implicit_cache():
    """Recording the same structural_hash twice as VALID does not
    auto-reclassify the second one — the caller decides the outcome."""
    ledger = BudgetLedger()
    ir_a = _ir(3)
    h = structural_hash(ir_a)
    ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=h))
    ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=h))
    assert ledger.num_valid == 2  # not deduplicated automatically
    assert ledger.num_unique == 1  # but the unique-hash count reflects only 1 distinct circuit


def test_invalid_records_do_not_require_a_structural_hash():
    ledger = BudgetLedger()
    ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.INVALID, issues=[]))
    assert ledger.num_invalid == 1
    assert ledger.num_unique == 0


def test_only_invalid_outcome_is_free_of_budget():
    assert not ProposalRecord(outcome=ProposalOutcome.INVALID).consumes_budget
    assert ProposalRecord(outcome=ProposalOutcome.VALID).consumes_budget
    assert ProposalRecord(outcome=ProposalOutcome.DUPLICATE).consumes_budget
    assert ProposalRecord(outcome=ProposalOutcome.FAILED).consumes_budget
