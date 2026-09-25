"""The search-facing evaluation harness.

This is the ONLY function any future search arm (random, evolutionary,
greedy, LLM-guided) calls to turn a circuit proposal into a score.

**Test quarantine is structural, not conventional.** `evaluate_candidate`
takes a `TrainValData` (train + val only — the type itself has no `test`
attribute, see `llm_vqc.tasks.base`) and returns an `EvaluationResult`
(no test-metric field exists on that class, see
`llm_vqc.evaluation.results`). There is no parameter, flag, or code path
in this function that can make test data or a test metric reach a search
algorithm — reaching test data would require importing
`llm_vqc.evaluation.final_test`, which this module does not do.

Every call updates the passed-in `BudgetLedger` exactly once, covering the
outcome categories the master plan's protocol needs (Section 6.3):
invalid proposals are recorded but do not consume training budget;
duplicate proposals (same structural hash already seen) consume budget
by returning the cached result; proposals that pass validation but whose
compilation or training subsequently fails are recorded as FAILED (not
VALID) so the ledger can report invalid/duplicate/failed rates as
distinct statistics — all three still consume budget, only INVALID does
not (Section 6.3's explicit rule), making the master plan's own
"exploration collapse" measurement possible.

If `proposal_event_store` and `run_id` are both given, every ledger
update is also durably persisted (`BudgetLedger.record_and_persist`) so
that a crashed and resumed search run reconstructs an identical ledger
via `BudgetLedger.from_store()` — see `llm_vqc.ir.budget` for the
crash-safety argument. This is optional and backward compatible: omit
both to get the original Phase 1/2 in-memory-only behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.evaluation.seeds import train_seed_for_circuit
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.ir.budget import (
    BudgetLedger,
    ProposalEventStore,
    ProposalOutcome,
    ProposalRecord,
)
from llm_vqc.ir.canonicalize import canonical_json
from llm_vqc.ir.canonicalize import structural_hash as compute_structural_hash
from llm_vqc.ir.compiler_pennylane import CompilerError
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.ir.validators import ValidationIssue, validate_proposal
from llm_vqc.tasks.base import TrainValData


class CandidateCache(Protocol):
    """Storage-backend-agnostic cache interface. `llm_vqc.evaluation.store.
    ResultStore` satisfies this structurally (matching method names,
    Protocol-style — no inheritance needed); harness.py only depends on
    this narrow protocol, not on SQLite."""

    def get_cached(
        self, task_name: str, structural_hash: str, train_seed: int
    ) -> EvaluationResult | None: ...

    def put_cached(
        self,
        task_name: str,
        structural_hash: str,
        train_seed: int,
        result: EvaluationResult,
        trained_weights: dict | None,
    ) -> None: ...


def evaluate_candidate(
    raw_proposal: dict | CircuitIR,
    task_name: str,
    run_seed: int,
    train_val: TrainValData,
    training_config: TrainingConfig,
    proposal_id: str,
    ledger: BudgetLedger,
    cache: CandidateCache | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
    proposal_event_store: ProposalEventStore | None = None,
    run_id: str | None = None,
    extra_validator: Callable[[CircuitIR], list[ValidationIssue]] | None = None,
    init_policy: Callable[[HybridQNNModel, int], None] | None = None,
) -> EvaluationResult:
    """Validate, (maybe) compile, (maybe) train, and score one proposal.

    `train_val` must be the task's train+val partition (never test data —
    the type does not permit it). Returns an `EvaluationResult` reporting
    only validation-set metrics.

    `extra_validator` (optional, backward compatible): an additional
    experiment-specific invariant check run *after* the standard
    `validate_proposal` succeeds. If it returns any issues, the proposal is
    recorded as INVALID (transparently, with those issues, budget-free — the
    same contract as a schema-invalid proposal) and never trained. Nothing is
    silently repaired. `init_policy` is forwarded to `train_model`.
    """

    def _record(record: ProposalRecord) -> None:
        if proposal_event_store is not None and run_id is not None:
            ledger.record_and_persist(proposal_event_store, run_id, record)
        else:
            ledger.record_proposal(record)

    validation = validate_proposal(raw_proposal)

    if not validation.valid:
        result = EvaluationResult(
            proposal_id=proposal_id,
            task_name=task_name,
            run_seed=run_seed,
            structural_hash=None,
            circuit_canonical_json=None,
            validation_outcome=ValidationOutcome.INVALID,
            validation_issues=validation.issues,
            compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.INVALID_PROPOSAL,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        _record(
            ProposalRecord(
                outcome=ProposalOutcome.INVALID, structural_hash=None, issues=validation.issues
            )
        )
        return result

    ir = validation.ir

    if extra_validator is not None:
        extra_issues = extra_validator(ir)
        if extra_issues:
            result = EvaluationResult(
                proposal_id=proposal_id,
                task_name=task_name,
                run_seed=run_seed,
                structural_hash=None,
                circuit_canonical_json=None,
                validation_outcome=ValidationOutcome.INVALID,
                validation_issues=extra_issues,
                compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
                training_outcome=TrainingOutcome.NOT_ATTEMPTED,
                failure_category=FailureCategory.INVALID_PROPOSAL,
                git_sha=git_sha,
                git_dirty=git_dirty,
            )
            _record(
                ProposalRecord(
                    outcome=ProposalOutcome.INVALID,
                    structural_hash=None,
                    issues=extra_issues,
                )
            )
            return result

    ir_hash = compute_structural_hash(ir)
    ir_json = canonical_json(ir)
    train_seed = train_seed_for_circuit(run_seed, ir_hash)

    if cache is not None:
        cached = cache.get_cached(task_name, ir_hash, train_seed)
        if cached is not None:
            duplicate_result = cached.model_copy(
                update={
                    "proposal_id": proposal_id,
                    "is_duplicate": True,
                    "cache_hit": True,
                }
            )
            _record(ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=ir_hash))
            return duplicate_result

    try:
        cost = circuit_cost_summary(ir)
    except CompilerError as exc:
        result = EvaluationResult(
            proposal_id=proposal_id,
            task_name=task_name,
            run_seed=run_seed,
            structural_hash=ir_hash,
            circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.FAILED,
            compilation_error=str(exc),
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.COMPILATION_ERROR,
            train_seed=train_seed,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash, cost=None))
        return result

    try:
        HybridQNNModel(
            ir,
            raw_feature_dim=train_val.spec.raw_feature_dim,
            head_out_dim=train_val.spec.classical_head_out_dim,
        )
    except (CompilerError, RuntimeError, ValueError) as exc:
        result = EvaluationResult(
            proposal_id=proposal_id,
            task_name=task_name,
            run_seed=run_seed,
            structural_hash=ir_hash,
            circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.FAILED,
            compilation_error=str(exc),
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            circuit_cost=cost,
            failure_category=FailureCategory.COMPILATION_ERROR,
            train_seed=train_seed,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash, cost=cost))
        return result

    training_output = train_model(ir, train_val, training_config, train_seed, init_policy=init_policy)

    if not training_output.success:
        result = EvaluationResult(
            proposal_id=proposal_id,
            task_name=task_name,
            run_seed=run_seed,
            structural_hash=ir_hash,
            circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.SUCCESS,
            training_outcome=TrainingOutcome.FAILED,
            training_error=training_output.error_message,
            train_loss_history=training_output.train_loss_history,
            val_metric_history=training_output.val_metric_history,
            epochs_completed=training_output.epochs_completed,
            wall_clock_seconds=training_output.wall_clock_seconds,
            circuit_cost=cost,
            failure_category=FailureCategory.TRAINING_DIVERGED
            if "Diverged" in (training_output.error_message or "")
            else FailureCategory.TRAINING_ERROR,
            train_seed=train_seed,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash, cost=cost))
        return result

    result = EvaluationResult(
        proposal_id=proposal_id,
        task_name=task_name,
        run_seed=run_seed,
        structural_hash=ir_hash,
        circuit_canonical_json=ir_json,
        validation_outcome=ValidationOutcome.VALID,
        compilation_outcome=CompilationOutcome.SUCCESS,
        training_outcome=TrainingOutcome.SUCCESS,
        val_metric_name=train_val.spec.metric_name,
        val_metric_value=training_output.final_val_metric,
        val_metric_history=training_output.val_metric_history,
        train_loss_history=training_output.train_loss_history,
        epochs_completed=training_output.epochs_completed,
        wall_clock_seconds=training_output.wall_clock_seconds,
        circuit_cost=cost,
        failure_category=FailureCategory.NONE,
        train_seed=train_seed,
        git_sha=git_sha,
        git_dirty=git_dirty,
        trained_weights_ref=ir_hash if cache is not None else None,
    )
    _record(ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=ir_hash, cost=cost))

    if cache is not None:
        cache.put_cached(
            task_name,
            ir_hash,
            train_seed,
            result,
            trained_weights={
                "circuit_weights": training_output.trained_circuit_weights,
                "classical_state": training_output.trained_classical_state,
            },
        )

    return result
