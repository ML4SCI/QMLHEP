"""Tests for durable `BudgetLedger` reconstruction (Stage 2 of the search-
and-pilot work cycle; fulfills governance decision G2-C).

These are the scenarios the work brief called out as mandatory:
clean reconstruction, reconstruction after interruption, invalid
proposals, duplicates, failed evaluations, incompatible resumes, and
exact budget exhaustion after restart.
"""

from __future__ import annotations

import pytest

from llm_vqc.evaluation.store import IncompatibleResumeError, ResultStore
from llm_vqc.ir.budget import BudgetLedger, ProposalOutcome, ProposalRecord
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec, RotationLayer

REPRO_FIELDS = {"task": "T1", "arm": "random", "budget": 10, "seed": 0}


def _ir(n_qubits=3) -> CircuitIR:
    return CircuitIR(
        n_qubits=n_qubits,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


def _store(tmp_path, run_id="run1"):
    return ResultStore.open_or_create(
        tmp_path / "results.sqlite",
        run_id=run_id,
        config_json="{}",
        config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None,
        created_at="t0",
    )


def test_clean_reconstruction_matches_original_ledger(tmp_path):
    store = _store(tmp_path)
    ir_a, ir_b = _ir(3), _ir(4)
    h_a, h_b = structural_hash(ir_a), structural_hash(ir_b)

    original = BudgetLedger()
    original.record_and_persist(
        store, "run1", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=h_a)
    )
    original.record_and_persist(
        store, "run1", ProposalRecord(outcome=ProposalOutcome.INVALID)
    )
    original.record_and_persist(
        store, "run1", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=h_b)
    )

    reconstructed = BudgetLedger.from_store(store, "run1")
    assert reconstructed.summary() == original.summary()
    assert reconstructed.num_unique == 2
    store.close()


def test_reconstruction_after_interruption_continues_gaplessly(tmp_path):
    """Simulate a crash: close and reopen the store mid-run, build a fresh
    ledger from it, and keep recording — the combined history must be
    identical to one unbroken run."""
    store = _store(tmp_path)
    ir_a = _ir(3)
    h_a = structural_hash(ir_a)

    ledger = BudgetLedger()
    ledger.record_and_persist(
        store, "run1", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=h_a)
    )
    ledger.record_and_persist(store, "run1", ProposalRecord(outcome=ProposalOutcome.INVALID))
    store.close()  # simulated crash

    # Resume: fresh process, fresh store handle, fresh ledger from disk.
    resumed_store = _store(tmp_path)
    resumed_ledger = BudgetLedger.from_store(resumed_store, "run1")
    assert resumed_ledger.num_proposed == 2
    assert resumed_ledger.consumed_budget == 1  # only the VALID one

    resumed_ledger.record_and_persist(
        resumed_store,
        "run1",
        ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=h_a),
    )
    assert resumed_ledger.num_proposed == 3
    assert resumed_ledger.consumed_budget == 2

    final = BudgetLedger.from_store(resumed_store, "run1")
    assert final.summary() == resumed_ledger.summary()
    resumed_store.close()


def test_invalid_proposals_preserved_and_free_of_budget_after_reconstruction(tmp_path):
    store = _store(tmp_path)
    ledger = BudgetLedger()
    for _ in range(3):
        ledger.record_and_persist(store, "run1", ProposalRecord(outcome=ProposalOutcome.INVALID))
    ledger.record_and_persist(
        store, "run1", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash="h1")
    )

    reconstructed = BudgetLedger.from_store(store, "run1")
    assert reconstructed.num_invalid == 3
    assert reconstructed.consumed_budget == 1  # invalid proposals never consume budget
    store.close()


def test_duplicate_accounting_preserved_after_reconstruction(tmp_path):
    store = _store(tmp_path)
    ir_a = _ir(3)
    h_a = structural_hash(ir_a)
    ledger = BudgetLedger()
    ledger.record_and_persist(
        store, "run1", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=h_a)
    )
    for _ in range(4):
        ledger.record_and_persist(
            store, "run1", ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=h_a)
        )

    reconstructed = BudgetLedger.from_store(store, "run1")
    assert reconstructed.num_duplicate == 4
    assert reconstructed.num_unique == 1  # still just one distinct circuit
    assert reconstructed.consumed_budget == 5  # 1 valid + 4 duplicates, all consume budget
    store.close()


def test_failed_evaluations_preserved_and_consume_budget_after_reconstruction(tmp_path):
    store = _store(tmp_path)
    ir_a = _ir(3)
    h_a = structural_hash(ir_a)
    cost = circuit_cost_summary(ir_a)
    ledger = BudgetLedger()
    ledger.record_and_persist(
        store,
        "run1",
        ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=h_a, cost=cost),
    )

    reconstructed = BudgetLedger.from_store(store, "run1")
    assert reconstructed.num_failed == 1
    assert reconstructed.consumed_budget == 1  # failed evaluations still consumed budget
    assert reconstructed.num_unique == 1  # the structural hash is still counted as seen
    store.close()


def test_incompatible_resume_is_rejected(tmp_path):
    db = tmp_path / "results.sqlite"
    store = ResultStore.open_or_create(
        db,
        run_id="run1",
        config_json="{}",
        config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None,
        created_at="t0",
    )
    store.close()

    different_fields = {**REPRO_FIELDS, "budget": 999}  # budget changed: not resumable
    with pytest.raises(IncompatibleResumeError):
        ResultStore.open_or_create(
            db,
            run_id="run1",
            config_json="{}",
            config_reproducibility_fields=different_fields,
            git_sha=None,
            created_at="t1",
        )


def test_exact_budget_exhaustion_after_restart(tmp_path):
    """A search run at budget_limit=5 that crashes after 3 consumed units
    must, on restart, allow exactly 2 more consuming proposals -- never
    more, never fewer."""
    budget_limit = 5
    store = _store(tmp_path)
    ledger = BudgetLedger()
    for _ in range(3):
        ledger.record_and_persist(
            store, "run1", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=None)
        )
    assert not ledger.is_exhausted(budget_limit)
    assert ledger.remaining_budget(budget_limit) == 2
    store.close()

    resumed_store = _store(tmp_path)
    resumed = BudgetLedger.from_store(resumed_store, "run1")
    assert resumed.consumed_budget == 3
    assert resumed.remaining_budget(budget_limit) == 2

    # Consume exactly the remaining budget.
    for _ in range(resumed.remaining_budget(budget_limit)):
        resumed.record_and_persist(
            resumed_store,
            "run1",
            ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=None),
        )

    assert resumed.is_exhausted(budget_limit)
    assert resumed.consumed_budget == budget_limit
    assert resumed.remaining_budget(budget_limit) == 0

    # A second restart must see exactly the same exhausted state -- no
    # extra budget appears from having crossed a restart boundary.
    final = BudgetLedger.from_store(resumed_store, "run1")
    assert final.consumed_budget == budget_limit
    assert final.is_exhausted(budget_limit)
    resumed_store.close()


def test_different_run_ids_do_not_share_budget(tmp_path):
    """Two arms/seeds sharing one SQLite file must not see each other's
    proposal events -- run_id is the isolation boundary."""
    store = _store(tmp_path, run_id="run_a")
    ledger_a = BudgetLedger()
    ledger_a.record_and_persist(
        store, "run_a", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash="h1")
    )

    # A second run_id, same underlying file.
    ledger_b = BudgetLedger()
    for _ in range(2):
        ledger_b.record_and_persist(
            store, "run_b", ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash="h2")
        )

    reconstructed_a = BudgetLedger.from_store(store, "run_a")
    reconstructed_b = BudgetLedger.from_store(store, "run_b")
    assert reconstructed_a.num_proposed == 1
    assert reconstructed_b.num_proposed == 2
    store.close()


def test_append_proposal_event_is_append_only_not_overwrite(tmp_path):
    """Reusing a proposal_index must fail loudly, not silently overwrite --
    protects the append-only invariant `from_store` relies on."""
    import sqlite3

    store = _store(tmp_path)
    store.append_proposal_event(
        "run1", 0, ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash="h1")
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.append_proposal_event(
            "run1", 0, ProposalRecord(outcome=ProposalOutcome.INVALID)
        )
    store.close()
