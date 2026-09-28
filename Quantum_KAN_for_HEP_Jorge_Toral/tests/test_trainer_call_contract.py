# Static (AST) checks that the training scripts call ClassicKANTrainer the way
# its current API expects: one trainer instance per script, prune_and_save_kan
# called only with keyword arguments that exist in its signature, and the
# run seed / split seed wired to the right places. A renamed or removed
# keyword argument otherwise only surfaces as a TypeError deep into a
# multi-hour pipeline run.
import ast
import inspect
from pathlib import Path

from src.architectures.classic_kan import ClassicKANTrainer

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
SCRIPT_PATHS = [
    SCRIPTS_DIR / "train_kan.py",
]
TRAINING_SCRIPT_PATHS = [
    SCRIPTS_DIR / "train_kan.py",
    SCRIPTS_DIR / "train_rf.py",
]

INSTANCE_METHOD_NAMES = {
    name
    for name, _ in inspect.getmembers(ClassicKANTrainer, predicate=inspect.isfunction)
    if not name.startswith("_")
}


def _trainer_var_names(tree):
    """Local names assigned from `ClassicKANTrainer(...)` or `<module>.ClassicKANTrainer(...)`."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        is_ctor_call = (
            isinstance(func, ast.Name) and func.id == "ClassicKANTrainer"
        ) or (
            isinstance(func, ast.Attribute) and func.attr == "ClassicKANTrainer"
        )
        if is_ctor_call:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _prune_call_kwargs(tree, trainer_vars):
    """Keyword-argument names used at each `<trainer>.prune_and_save_kan(...)`
    call site."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "prune_and_save_kan"
            and isinstance(func.value, ast.Name)
            and func.value.id in trainer_vars
        ):
            calls.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return calls


def test_instance_method_set_is_nonempty():
    # Sanity check on the reflection itself, so a refactor that renames/removes
    # every stage method fails loudly here instead of silently elsewhere.
    assert {"train_kan_model", "prune_and_save_kan", "retrain_pruned_kan",
            "simplify_and_save", "finetune_symbolic_model"} <= INSTANCE_METHOD_NAMES


def test_scripts_instantiate_trainer_exactly_once():
    for path in SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        trainer_vars = _trainer_var_names(tree)
        assert len(trainer_vars) == 1, (
            f"{path.name} should construct exactly one ClassicKANTrainer instance, "
            f"found assignments to: {trainer_vars or 'none'}"
        )


def test_prune_and_save_kan_call_sites_use_real_kwargs():
    real_params = set(inspect.signature(ClassicKANTrainer.prune_and_save_kan).parameters) - {"self"}
    for path in SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        call_sites = _prune_call_kwargs(tree, _trainer_var_names(tree))
        assert call_sites, f"{path.name} never calls prune_and_save_kan on its trainer"
        for used_kwargs in call_sites:
            unknown = used_kwargs - real_params
            assert not unknown, (
                f"{path.name} calls prune_and_save_kan with unknown kwargs {unknown} "
                f"(valid: {sorted(real_params)})"
            )


def _calls_named(tree, name):
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == name) or (
            isinstance(func, ast.Name) and func.id == name
        ):
            calls.append(node)
    return calls


def test_training_runs_use_run_seed_while_preprocessors_use_split_seed():
    for path in TRAINING_SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "set_seed"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "workspace"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Attribute)
            and call.args[0].attr == "seed"
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
        ), f"{path.name} should set the runtime RNG from args.seed"

        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load_and_preprocess_data"
            and any(
                kw.arg == "seed" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "seed"
                for kw in call.keywords
            )
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
        ), f"{path.name} should select the data subset using args.seed"

    for path in (SCRIPTS_DIR.parent / "src" / "preprocessing" / "processor_top.py",
                 SCRIPTS_DIR.parent / "src" / "preprocessing" / "processor_qg.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        set_seed_calls = _calls_named(tree, "set_seed")
        assert any(
            len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "subset_split_seed"
            for call in set_seed_calls
        ), f"{path.name} should seed the canonical split with subset_split_seed"
        assert any(
            len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "seed"
            for call in set_seed_calls
        ), f"{path.name} should restore the run-specific seed when selecting a subset"
