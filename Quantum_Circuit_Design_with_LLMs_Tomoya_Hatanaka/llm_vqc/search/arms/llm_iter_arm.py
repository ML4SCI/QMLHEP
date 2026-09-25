"""The `llm_iter` search arm (master plan Section 6.2): "Knipfer-style
conversational agent (full history in context) emitting IR."

Also implements the open-loop control (ablation A5, "LLM proposes B
circuits blind, no scores returned") via the `open_loop` constructor flag
-- same arm, same prompt schema description, the only difference being
whether prior-proposal feedback is included in the user prompt.

**Only approved feedback fields ever reach a prompt.** `_summarize_feedback`
reads only `SearchFeedback`'s validation-only fields (outcome, validation
issue count, cost stats, val metric, duplicate flag) -- there is no test
metric to accidentally include (impossible: `SearchFeedback` has no such
field) and no other arm's results are ever visible to this one.

**Malformed LLM output becomes an ordinary INVALID proposal.** If
`LLMDriver` exhausts its repair attempts without a parseable, schema-valid
proposal, this arm hands the shared harness a deliberately-invalid
placeholder dict -- which the harness's ordinary validator rejects as
`INVALID`, exactly like any other arm's malformed proposal (budget-free
for everyone, governance decision G4-B; see `llm_vqc.llm.driver`'s
docstring for the full reasoning).
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.driver import LLMDriver
from llm_vqc.llm.prompts import (
    build_iter_user_prompt,
    build_open_loop_user_prompt,
    build_system_prompt,
)
from llm_vqc.llm.provider import LLMProvider
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback

_MALFORMED_PLACEHOLDER_KEY = "_llm_malformed"


class LLMIterState(BaseModel):
    seed: int
    n_proposed: int = 0
    history_lines: list[str] = []
    best_hash: str | None = None
    best_metric: float | None = None


def _summarize_feedback(proposal_index: int, feedback: SearchFeedback) -> str:
    if feedback.outcome.value == "invalid":
        return f"Proposal {proposal_index}: INVALID ({len(feedback.validation_issues)} issue(s))."
    cost_str = ""
    if feedback.circuit_cost is not None:
        cost_str = (
            f", depth={feedback.circuit_cost.depth}, "
            f"gates={feedback.circuit_cost.gate_count}, "
            f"params={feedback.circuit_cost.parameter_count}"
        )
    metric_str = (
        f"{feedback.val_metric_name}={feedback.val_metric_value:.4f}"
        if feedback.val_metric_value is not None
        else "no metric (training/compilation failed)"
    )
    duplicate_str = " [duplicate of an earlier proposal]" if feedback.is_duplicate else ""
    return f"Proposal {proposal_index}: {metric_str}{cost_str}{duplicate_str}."


class LLMIterArm(SearchArm[LLMIterState]):
    name = "llm_iter"

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
        open_loop: bool = False,
        history_window: int | None = None,
        user_prompt_suffix: str = "",
    ) -> None:
        self.lower_is_better = lower_is_better
        self.system_prompt = build_system_prompt(task_description)
        self.driver = LLMDriver(provider, max_repair_attempts, budget, cost_estimate_per_call_usd)
        self.result_store = result_store
        self.run_id = run_id
        self.temperature = temperature
        self.open_loop = open_loop
        # Compact-feedback mode (token-limited providers): instead of the
        # full raw history, the prompt carries one deterministic
        # best-so-far line plus only the last `history_window` feedback
        # lines. None = full history (the original behavior, used for the
        # OpenAI runs). Fixed per experiment condition, identical across
        # seeds; only master-plan-approved fields ever appear either way.
        self.history_window = history_window
        # Fixed format-reminder appended to every user prompt (recorded
        # verbatim in provenance; never carries result information).
        self.user_prompt_suffix = user_prompt_suffix

    def initialize(self, seed: int) -> LLMIterState:
        return LLMIterState(seed=seed)

    def propose(self, state: LLMIterState) -> dict:
        if self.open_loop:
            user_prompt = build_open_loop_user_prompt(state.n_proposed)
        else:
            lines = state.history_lines
            if self.history_window is not None and lines:
                summary = (
                    f"Best so far: {state.best_metric:.4f}"
                    if state.best_metric is not None
                    else "Best so far: none yet"
                )
                lines = [summary, *lines[-self.history_window:]]
            user_prompt = build_iter_user_prompt(lines)
        user_prompt += self.user_prompt_suffix

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
        self, state: LLMIterState, proposal: dict, feedback: SearchFeedback
    ) -> LLMIterState:
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value

        history_lines = state.history_lines
        if not self.open_loop:
            history_lines = [*history_lines, _summarize_feedback(state.n_proposed, feedback)]

        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "history_lines": history_lines,
                "best_hash": best_hash,
                "best_metric": best_metric,
            }
        )

    def select_final(self, state: LLMIterState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> LLMIterState:
        return LLMIterState.model_validate_json(raw_json)
