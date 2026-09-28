"""`SearchFeedback`: the ONLY information any search arm ever receives back
after one candidate-evaluation proposal.

Per LLM-VQC_MASTER_PLAN.md Section 6.3 ("Feedback to searchers uses
validation only. Test metrics are computed once and quarantined until
analysis"), this is structurally test-free the same way `EvaluationResult`
and `TrainValData` are: there is no field on this class that could ever
hold a test metric, a test label, another arm's results, or a privileged
hand-authored circuit. `from_evaluation_result` is a pure, narrow
projection off `EvaluationResult` — it cannot invent fields `EvaluationResult`
does not have, so widening this class to leak test data would require a
visible change to `EvaluationResult` first (which final_test.py alone
produces, and which harness.py never imports).
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.ir.budget import ProposalOutcome
from llm_vqc.ir.metrics import CircuitCostSummary
from llm_vqc.ir.validators import ValidationIssue


def infer_proposal_outcome(result: EvaluationResult) -> ProposalOutcome:
    """Derive the `ProposalOutcome` category from an `EvaluationResult`.

    Mirrors exactly the classification `llm_vqc.evaluation.harness.
    evaluate_candidate` uses when it records to the `BudgetLedger` --
    single source of truth for "what kind of outcome was this", so a
    search arm's feedback and the ledger's accounting can never disagree.
    """
    if result.validation_outcome == ValidationOutcome.INVALID:
        return ProposalOutcome.INVALID
    if result.is_duplicate:
        return ProposalOutcome.DUPLICATE
    if (
        result.compilation_outcome == CompilationOutcome.FAILED
        or result.training_outcome == TrainingOutcome.FAILED
    ):
        return ProposalOutcome.FAILED
    return ProposalOutcome.VALID


class SearchFeedback(BaseModel):
    """Validation-only feedback for one proposal, handed to `SearchArm.update_state`.

    `diagnostics` is `None` unless an experimental condition explicitly
    declares diagnostics as a feedback channel (master plan ablation A1);
    the shared runner is responsible for only populating it under that
    declared condition, never by default.
    """

    proposal_id: str
    outcome: ProposalOutcome
    structural_hash: str | None
    is_duplicate: bool
    validation_issues: list[ValidationIssue]
    compilation_outcome: CompilationOutcome
    training_outcome: TrainingOutcome
    val_metric_name: str | None
    val_metric_value: float | None
    circuit_cost: CircuitCostSummary | None
    diagnostics: dict[str, dict[str, float | None]] | None = None

    @classmethod
    def from_evaluation_result(
        cls,
        result: EvaluationResult,
        diagnostics: dict[str, dict[str, float | None]] | None = None,
    ) -> SearchFeedback:
        return cls(
            proposal_id=result.proposal_id,
            outcome=infer_proposal_outcome(result),
            structural_hash=result.structural_hash,
            is_duplicate=result.is_duplicate,
            validation_issues=result.validation_issues,
            compilation_outcome=result.compilation_outcome,
            training_outcome=result.training_outcome,
            val_metric_name=result.val_metric_name,
            val_metric_value=result.val_metric_value,
            circuit_cost=result.circuit_cost,
            diagnostics=diagnostics,
        )
