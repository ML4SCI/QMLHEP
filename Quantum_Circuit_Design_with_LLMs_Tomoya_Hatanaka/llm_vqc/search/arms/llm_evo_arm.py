"""The `llm_evo` search arm (master plan Section 6.2): "LLM as mutation
operator: prompt contains top-k archive (IR + scores + diversity note),
LLM proposes offspring; FunSearch-style."

The archive is bounded (`archive_size`, default 5) and always sorted by
validation metric -- only successfully-scored proposals ever enter it, so
a run of invalid/failed proposals never displaces a known-good archive
entry with nothing. Same malformed-output handling and approved-feedback-
fields-only policy as `llm_iter_arm` (see that module's docstring).
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.driver import LLMDriver
from llm_vqc.llm.prompts import build_evo_user_prompt, build_system_prompt
from llm_vqc.llm.provider import LLMProvider
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback

_MALFORMED_PLACEHOLDER_KEY = "_llm_malformed"


class ArchiveEntry(BaseModel):
    ir: CircuitIR
    structural_hash: str
    val_metric: float


class LLMEvoState(BaseModel):
    seed: int
    n_proposed: int = 0
    archive: list[ArchiveEntry] = []
    best_hash: str | None = None
    best_metric: float | None = None


def _summarize_archive_entry(entry: ArchiveEntry) -> str:
    n_layers = len(entry.ir.layers)
    return (
        f"circuit (hash {entry.structural_hash[:8]}): n_qubits={entry.ir.n_qubits}, "
        f"n_layers={n_layers}, metric={entry.val_metric:.4f}"
    )


class LLMEvoArm(SearchArm[LLMEvoState]):
    name = "llm_evo"

    def __init__(
        self,
        lower_is_better: bool,
        task_description: str,
        provider: LLMProvider,
        result_store: ResultStore,
        run_id: str,
        temperature: float = 0.7,
        max_repair_attempts: int = 2,
        budget: LLMApiBudget | None = None,
        cost_estimate_per_call_usd: float = 0.0,
        archive_size: int = 5,
        user_prompt_suffix: str = "",
    ) -> None:
        self.lower_is_better = lower_is_better
        self.system_prompt = build_system_prompt(task_description)
        self.driver = LLMDriver(provider, max_repair_attempts, budget, cost_estimate_per_call_usd)
        self.result_store = result_store
        self.run_id = run_id
        self.temperature = temperature
        self.archive_size = archive_size
        self.user_prompt_suffix = user_prompt_suffix

    def initialize(self, seed: int) -> LLMEvoState:
        return LLMEvoState(seed=seed)

    def propose(self, state: LLMEvoState) -> dict:
        archive_lines = [_summarize_archive_entry(e) for e in state.archive]
        user_prompt = build_evo_user_prompt(archive_lines) + self.user_prompt_suffix

        proposal_id = f"{self.run_id}:{state.n_proposed}"
        outcome = self.driver.propose(
            proposal_id, self.system_prompt, user_prompt, self.temperature
        )
        for record in outcome.call_records:
            self.result_store.append_llm_call(self.run_id, record)

        if outcome.proposal is not None:
            return outcome.proposal
        return {_MALFORMED_PLACEHOLDER_KEY: True, "retry_count": outcome.retry_count}

    def update_state(
        self, state: LLMEvoState, proposal: dict, feedback: SearchFeedback
    ) -> LLMEvoState:
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value

        archive = list(state.archive)
        if feedback.val_metric_value is not None and feedback.structural_hash is not None:
            archive.append(
                ArchiveEntry(
                    ir=CircuitIR.model_validate(proposal),
                    structural_hash=feedback.structural_hash,
                    val_metric=feedback.val_metric_value,
                )
            )
            archive.sort(key=lambda e: e.val_metric if self.lower_is_better else -e.val_metric)
            archive = archive[: self.archive_size]

        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "archive": archive,
                "best_hash": best_hash,
                "best_metric": best_metric,
            }
        )

    def select_final(self, state: LLMEvoState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> LLMEvoState:
        return LLMEvoState.model_validate_json(raw_json)
