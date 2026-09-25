"""Typed result of one completed (or checkpointed) search-arm run."""

from __future__ import annotations

from pydantic import BaseModel


class SearchRunResult(BaseModel):
    """What `SearchRunner.run()` returns when a run's budget is exhausted.

    Carries no test metric -- selecting the final candidate happens on
    validation data only (`selected_val_metric_value`, computed by the
    same `TaskSpec.metric_name` every candidate in this run used). Running
    the quarantined final-test evaluation on `selected_structural_hash` is
    a separate, explicit step the caller takes after this object is
    returned; `llm_vqc.search` never imports `llm_vqc.evaluation.final_test`.
    """

    run_id: str
    arm_name: str
    task_name: str
    run_seed: int
    budget_limit: int
    ledger_summary: dict[str, int]
    selected_structural_hash: str | None
    selected_train_seed: int | None
    selected_val_metric_name: str | None
    selected_val_metric_value: float | None
