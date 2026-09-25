"""The shared search framework (Phase 4/5 of the master plan).

Every search arm (`random`, `evolutionary`, `greedy`, `llm_iter`,
`llm_evo`) is a `SearchArm` subclass driven by the identical
`SearchRunner` loop -- see `llm_vqc.search.runner` for the fairness
argument. `llm_vqc.search` never imports `llm_vqc.evaluation.final_test`.
"""

from __future__ import annotations

from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback, infer_proposal_outcome
from llm_vqc.search.results import SearchRunResult
from llm_vqc.search.runner import SearchRunner, SearchRunnerError

__all__ = [
    "SearchArm",
    "SearchFeedback",
    "SearchRunResult",
    "SearchRunner",
    "SearchRunnerError",
    "infer_proposal_outcome",
    "is_strictly_better",
]
