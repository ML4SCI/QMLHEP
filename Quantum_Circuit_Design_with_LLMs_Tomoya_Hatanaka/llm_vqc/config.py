"""Experiment configuration schema.

This is deliberately minimal for Phase 0 (see LLM-VQC_MASTER_PLAN.md
Section 9): a base config with the fields every future experiment run
needs (name, seed) plus an open ``params`` bag for phase-specific fields
(task, arm, budget, model, etc.) that will be added in Phase 1+ without
requiring changes here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Base configuration for a single experiment run."""

    name: str
    seed: int
    params: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RunConfig.model_validate(raw)
