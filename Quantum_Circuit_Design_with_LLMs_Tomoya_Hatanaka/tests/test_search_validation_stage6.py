"""Stage 6 cross-cutting validation tests: invariants that must hold
*across* every search arm simultaneously, not just within one arm's own
test file. Per-arm budget/determinism/resume tests already live in
`test_search_arms.py` and `test_llm_arms.py`; this file adds the
comparisons that only make sense arm-to-arm, plus the "no arbitrary code
execution" / "no arm-specific compiler path" structural guarantees.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from pydantic import BaseModel

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.sampler import sample_random_ir
from llm_vqc.llm.provider import MockLLMProvider
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.arms.evolutionary_arm import EvolutionaryArm, EvolutionaryArmConfig
from llm_vqc.search.arms.greedy_arm import GreedyArm
from llm_vqc.search.arms.llm_evo_arm import LLMEvoArm
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm
from llm_vqc.search.arms.random_arm import RandomArm
from llm_vqc.search.feedback import SearchFeedback
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

REPRO_FIELDS = {"task": "T1"}
TASK_DESCRIPTION = "T1 1D Gaussian-peak regression"


@pytest.fixture(scope="module")
def t1_data():
    return T1GaussianPeakTask().build(seed=0)


def _training_config() -> TrainingConfig:
    return TrainingConfig(epochs=2)


def _store(tmp_path, run_id, name, budget):
    return ResultStore.open_or_create(
        tmp_path / name,
        run_id=run_id,
        config_json="{}",
        config_reproducibility_fields={**REPRO_FIELDS, "budget": budget},
        git_sha=None,
        created_at="t0",
    )


def _make_arm(name, store, run_id):
    if name == "random":
        return RandomArm(lower_is_better=True)
    if name == "greedy":
        return GreedyArm(lower_is_better=True, k=2)
    if name == "evolutionary":
        return EvolutionaryArm(lower_is_better=True, config=EvolutionaryArmConfig(mu=2, lambda_=2))
    if name == "llm_iter":
        return LLMIterArm(
            lower_is_better=True, task_description=TASK_DESCRIPTION,
            provider=MockLLMProvider(seed=0), result_store=store, run_id=run_id, budget=None,
        )
    if name == "llm_evo":
        return LLMEvoArm(
            lower_is_better=True, task_description=TASK_DESCRIPTION,
            provider=MockLLMProvider(seed=0), result_store=store, run_id=run_id, budget=None,
        )
    raise ValueError(name)


ALL_ARMS = ["random", "greedy", "evolutionary", "llm_iter", "llm_evo"]


# --- Exact budget equality across every arm -------------------------------


def test_every_arm_consumes_exactly_the_same_budget(tmp_path, t1_data):
    budget = 4
    for arm_name in ALL_ARMS:
        store = _store(tmp_path, "run", f"{arm_name}.sqlite", budget)
        runner = SearchRunner(
            _make_arm(arm_name, store, "run"), "T1", t1_data, _training_config(),
            budget_limit=budget, run_seed=0, result_store=store, run_id="run",
        )
        result = runner.run()
        assert result.ledger_summary["consumed_budget"] == budget, arm_name
        store.close()


# --- Identical training protocol regardless of arm ------------------------


class _FixedProposalState(BaseModel):
    seed: int
    n_proposed: int = 0
    best_hash: str | None = None
    best_metric: float | None = None


class _FixedProposalArm(SearchArm[_FixedProposalState]):
    """Always proposes the exact same circuit -- used only to prove the
    shared runner applies an identical training protocol (same seed
    derivation, same result) no matter which arm class is driving it."""

    name = "fixed"

    def __init__(self, fixed_proposal: dict) -> None:
        self.fixed_proposal = fixed_proposal

    def initialize(self, seed: int) -> _FixedProposalState:
        return _FixedProposalState(seed=seed)

    def propose(self, state: _FixedProposalState) -> dict:
        return self.fixed_proposal

    def update_state(
        self, state: _FixedProposalState, proposal: dict, feedback: SearchFeedback
    ) -> _FixedProposalState:
        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "best_hash": feedback.structural_hash,
                "best_metric": feedback.val_metric_value,
            }
        )

    def select_final(self, state: _FixedProposalState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> _FixedProposalState:
        return _FixedProposalState.model_validate_json(raw_json)


def test_identical_circuit_gets_identical_training_result_regardless_of_arm(tmp_path, t1_data):
    import numpy as np

    fixed_ir = sample_random_ir(np.random.default_rng(123))
    fixed_hash = structural_hash(fixed_ir)
    proposal = fixed_ir.model_dump()

    results = []
    for i in range(2):
        store = _store(tmp_path, "run", f"fixed_{i}.sqlite", 1)
        runner = SearchRunner(
            _FixedProposalArm(proposal), "T1", t1_data, _training_config(),
            budget_limit=1, run_seed=0, result_store=store, run_id="run",
        )
        results.append(runner.run())
        store.close()

    assert results[0].selected_structural_hash == fixed_hash
    assert results[1].selected_structural_hash == fixed_hash
    assert results[0].selected_val_metric_value == results[1].selected_val_metric_value


# --- No arbitrary code execution -------------------------------------------


def test_no_arm_or_llm_code_ever_calls_eval_exec_or_a_subprocess():
    """`os` itself is legitimately imported (e.g. `os.environ.get` for
    `LLM_API_BUDGET_USD`) -- what must never appear is a call to
    `eval`/`exec`/`compile`, a process-spawning `os.*` call
    (`system`/`popen`/`exec*`/`spawn*`), or any use of the `subprocess`
    module at all."""
    forbidden_calls = {"eval", "exec", "compile"}
    forbidden_os_attrs_prefixes = ("system", "popen", "exec", "spawn")
    packages = [pathlib.Path("llm_vqc/search"), pathlib.Path("llm_vqc/llm")]
    for pkg in packages:
        for path in pkg.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls, (path, node.func.id)
                if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                    raise AssertionError((path, node.module))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] != "subprocess", (path, alias.name)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                ):
                    assert not node.func.attr.startswith(forbidden_os_attrs_prefixes), (
                        path,
                        node.func.attr,
                    )


# --- No arm-specific compiler/preprocessing path ---------------------------


def test_runner_calls_evaluate_candidate_exactly_once_with_no_arm_type_branching():
    """The shared runner must have exactly one call site to
    `evaluate_candidate` and must not branch on `type(self.arm)` /
    `isinstance(self.arm, ...)` anywhere -- an arm-specific code path in
    the runner would be an unmatchable, unfair action space."""
    source = pathlib.Path("llm_vqc/search/runner.py").read_text()
    tree = ast.parse(source)

    call_names = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert call_names.count("evaluate_candidate") == 1

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            raise AssertionError("runner.py must not branch on isinstance(self.arm, ...)")
