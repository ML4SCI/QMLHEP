"""`SearchRunner`: the ONE execution loop every search arm runs through.

This is the "shared search framework" the master plan requires (Section
5.1/6.4): every arm -- `random`, `evolutionary`, `greedy`, and the LLM
arms -- is driven by this exact loop, using the identical `CircuitIR`
validator, compiler, task evaluator, training protocol, candidate budget,
duplicate policy, failure accounting, and result store. An arm cannot
bypass IR validation (every proposal goes through
`llm_vqc.evaluation.harness.evaluate_candidate`, which validates
internally) and cannot reach task-test data (this module never imports
`llm_vqc.evaluation.final_test`; verified in
`tests/test_search_framework.py`).

**Termination is exact.** The loop's only continue condition is
`not ledger.is_exhausted(budget_limit)`, checked before every proposal;
`BudgetLedger.from_store()` on resume guarantees a crashed-and-restarted
run cannot receive extra budget (governance decision G2-C).

**Checkpointing is unconditional.** Arm state is persisted (via
`ResultStore.save_run_state`) immediately after initialization and after
every single evaluated proposal -- Section 11's "budget ledger
checkpointed after every evaluation" extended to arm state, since an arm's
search-algorithm state is exactly as unrecoverable as its budget count
would be if lost.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_vqc.evaluation.harness import evaluate_candidate
from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import derive_child_seed, train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.ir.budget import BudgetLedger
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.ir.validators import ValidationIssue
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.feedback import SearchFeedback
from llm_vqc.search.results import SearchRunResult
from llm_vqc.tasks.base import TrainValData

#: Generous ceiling on total proposals (valid + invalid + duplicate +
#: failed) regardless of consumed budget, so a pathologically broken arm
#: that always emits invalid IR (which never consumes budget) cannot loop
#: forever. Chosen large enough to never bind a correctly-behaving arm at
#: any budget this project uses (see master plan Section 6.3, B <= 60).
_MAX_PROPOSALS_PER_BUDGET_UNIT = 25


class SearchRunnerError(Exception):
    """Raised when a run cannot proceed safely (e.g. an arm appears to
    never produce a budget-consuming proposal)."""


class SearchRunner:
    def __init__(
        self,
        arm: SearchArm,
        task_name: str,
        train_val: TrainValData,
        training_config: TrainingConfig,
        budget_limit: int,
        run_seed: int,
        result_store: ResultStore,
        run_id: str,
        git_sha: str | None = None,
        git_dirty: bool | None = None,
        extra_validator: Callable[[CircuitIR], list[ValidationIssue]] | None = None,
        init_policy: Callable[[HybridQNNModel, int], None] | None = None,
    ) -> None:
        self.arm = arm
        self.task_name = task_name
        self.train_val = train_val
        self.training_config = training_config
        self.budget_limit = budget_limit
        self.run_seed = run_seed
        self.result_store = result_store
        self.run_id = run_id
        self.git_sha = git_sha
        self.git_dirty = git_dirty
        self.extra_validator = extra_validator
        self.init_policy = init_policy

    def _load_or_initialize_state(self):
        raw = self.result_store.load_run_state(self.run_id)
        if raw is not None:
            return self.arm.deserialize_state(raw)
        arm_seed = derive_child_seed(self.run_seed, "search", self.arm.name)
        state = self.arm.initialize(arm_seed)
        self.result_store.save_run_state(self.run_id, self.arm.serialize_state(state))
        return state

    def run(self) -> SearchRunResult:
        ledger = BudgetLedger.from_store(self.result_store, self.run_id)
        state = self._load_or_initialize_state()

        max_proposals = self.budget_limit * _MAX_PROPOSALS_PER_BUDGET_UNIT
        proposal_index = ledger.num_proposed

        while not ledger.is_exhausted(self.budget_limit):
            if proposal_index >= max_proposals:
                raise SearchRunnerError(
                    f"arm {self.arm.name!r} made {proposal_index} proposals without "
                    f"consuming the budget of {self.budget_limit} -- it appears to never "
                    "produce a valid, duplicate, or failed (budget-consuming) proposal. "
                    "Aborting rather than looping forever."
                )
            proposal_id = f"{self.run_id}:{proposal_index}"
            raw_proposal = self.arm.propose(state)
            result = evaluate_candidate(
                raw_proposal,
                self.task_name,
                self.run_seed,
                self.train_val,
                self.training_config,
                proposal_id,
                ledger,
                cache=self.result_store,
                git_sha=self.git_sha,
                git_dirty=self.git_dirty,
                proposal_event_store=self.result_store,
                run_id=self.run_id,
                extra_validator=self.extra_validator,
                init_policy=self.init_policy,
            )
            feedback = SearchFeedback.from_evaluation_result(result)
            state = self.arm.update_state(state, raw_proposal, feedback)
            self.result_store.save_run_state(self.run_id, self.arm.serialize_state(state))
            proposal_index += 1

        return self._finalize(state, ledger)

    def _finalize(self, state, ledger: BudgetLedger) -> SearchRunResult:
        best_hash = self.arm.select_final(state)
        train_seed: int | None = None
        metric_name: str | None = None
        metric_value: float | None = None

        if best_hash is not None:
            train_seed = train_seed_for_circuit(self.run_seed, best_hash)
            cached = self.result_store.get_cached(self.task_name, best_hash, train_seed)
            if cached is not None:
                metric_name = cached.val_metric_name
                metric_value = cached.val_metric_value

        return SearchRunResult(
            run_id=self.run_id,
            arm_name=self.arm.name,
            task_name=self.task_name,
            run_seed=self.run_seed,
            budget_limit=self.budget_limit,
            ledger_summary=ledger.summary(),
            selected_structural_hash=best_hash,
            selected_train_seed=train_seed,
            selected_val_metric_name=metric_name,
            selected_val_metric_value=metric_value,
        )
