# Covers workspace.resolve_variant / get_config(full_dataset=...) / iter_run_dirs: the
# single place that encodes a run's data regime (mass cut, n_subsets, subset) in
# directory names. Pure path logic -- no data, models or training involved.
import itertools
import os

import pytest

from src.utils import workspace

# get_config keys that are intentionally shared by every regime.
SHARED_KEYS = {"root", "raw_data_dir", "aggregate_dir", "metrics_table_path", "pipeline_logs_dir",
               "sine_comparison_summary_path"}


def _path_values(config):
    root = str(config["root"])
    return {k: v for k, v in config.items()
            if k not in SHARED_KEYS and isinstance(v, str) and v.startswith(root)}


def test_resolve_variant_labels():
    v = workspace.resolve_variant("top", apply_mass_cut=True, n_subsets=5, seed=13)
    assert (v["cut"], v["data_label"], v["run_label"], v["subset_id"]) == ("mass_cut", "n5", "n5_subset3", 3)
    assert v["variant"] == "mass_cut/n5_subset3"

    v = workspace.resolve_variant("top", apply_mass_cut=False, n_subsets=1, seed=42)
    assert (v["cut"], v["data_label"], v["run_label"], v["subset_id"]) == ("no_mass_cut", "full", "full", 0)

    # quark-gluon has no mass cut -> no cut level in the layout.
    v = workspace.resolve_variant("quark-gluon", apply_mass_cut=True, n_subsets=5, seed=7)
    assert v["cut"] is None
    assert v["variant"] == "n5_subset2"


def test_full_dataset_forces_no_mass_cut_and_single_subset():
    cfg = workspace.get_config("top", 42, full_dataset=True)
    assert cfg["apply_mass_cut"] is False
    assert cfg["n_subsets"] == 1
    assert cfg["full_dataset"] is True
    assert cfg["variant"] == "no_mass_cut/full"
    assert cfg["run_dir"].endswith(os.path.join("outputs", "top", "no_mass_cut", "full", "seed_42"))
    assert cfg["canonical_cache_file"].endswith(
        os.path.join("data", "processed", "top", "no_mass_cut", "full", "preprocessed_subsets.pt"))

    # full_dataset wins over explicit overrides
    cfg = workspace.get_config("top", 42, full_dataset=True, apply_mass_cut=True, n_subsets=5)
    assert cfg["apply_mass_cut"] is False and cfg["n_subsets"] == 1


def test_default_regime_is_not_full_and_names_the_subset():
    cfg = workspace.get_config("top", 10)
    assert cfg["full_dataset"] is False
    assert cfg["n_subsets"] > 1
    assert cfg["run_dir"].endswith(
        os.path.join("outputs", "top", "mass_cut", f"n{cfg['n_subsets']}_subset{10 % cfg['n_subsets']}", "seed_10"))
    # the canonical cache is seed-independent: it names the partition, not the subset
    assert cfg["canonical_cache_file"].endswith(
        os.path.join("data", "processed", "top", "mass_cut", f"n{cfg['n_subsets']}", "preprocessed_subsets.pt"))
    assert workspace.get_config("top", 11)["canonical_cache_file"] == cfg["canonical_cache_file"]


def test_quantum_weights_live_under_the_run_dir():
    for kwargs in ({}, {"full_dataset": True}):
        cfg = workspace.get_config("top", 42, **kwargs)
        assert cfg["polynomial_weights_dir"].startswith(cfg["run_dir"])
        assert cfg["qkan_ideal_path"].startswith(cfg["run_dir"])


def test_regimes_never_share_a_path():
    full = _path_values(workspace.get_config("top", 42, full_dataset=True))
    part = _path_values(workspace.get_config("top", 42))
    nocut = _path_values(workspace.get_config("top", 42, apply_mass_cut=False))
    for a, b in itertools.combinations((full, part, nocut), 2):
        clashes = {k for k in a.keys() & b.keys() if a[k] == b[k]}
        assert not clashes, f"regimes share paths: {sorted(clashes)}"


@pytest.mark.parametrize("baseline,random_init,backend", [
    (False, False, "ideal"), (False, False, "noisy"), (False, False, "shots"),
    (True, False, "ideal"), (True, False, "noisy"), (True, False, "shots"),
    (False, True, "ideal"),
])
def test_config_defines_every_key_quantum_evaluate_requests(baseline, random_init, backend):
    """QuantumKANTrainer.evaluate builds `{metric}_qkan{suffix}_{backend}` keys; a missing
    key used to surface only at the very end of a run (KeyError for random-init npy keys)."""
    cfg = workspace.get_config("top", 10)
    suffix = ("_baseline" if baseline else "") + ("_random" if random_init else "")
    for key in (f"roc_qkan{suffix}_{backend}", f"pr_qkan{suffix}_{backend}",
                f"cm_qkan{suffix}_{backend}", f"cm_qkan{suffix}_{backend}_normalized",
                f"metrics_qkan{suffix}_{backend}",
                f"qkan_eval_data_true{suffix}_{backend}", f"qkan_eval_data_probs{suffix}_{backend}",
                f"qkan_eval_data_binary{suffix}_{backend}"):
        assert key in cfg, key

    if not baseline:  # training-side keys used by QuantumKANTrainer._setup_backend_paths
        prefix = "random_" if random_init else ""
        assert f"qkan_{prefix}{backend}_path" in cfg
        assert f"history_{prefix}{backend}_loss" in cfg


def test_iter_run_dirs_parses_variant_layout_and_flags_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "get_project_root", lambda: tmp_path)
    for kwargs, seed in (({}, 12), ({"full_dataset": True}, 42), ({"apply_mass_cut": False}, 4)):
        os.makedirs(workspace.get_config("top", seed, **kwargs)["run_dir"])
    os.makedirs(tmp_path / "outputs" / "top" / "seed_3")                    # pre-variant layout
    os.makedirs(tmp_path / "outputs" / "top" / "legacy" / "seed_5")          # explicitly relocated
    os.makedirs(tmp_path / "outputs" / "top" / "aggregate")                  # not a run

    runs = {r["seed"]: r for r in workspace.iter_run_dirs("top")}
    n = workspace.get_config("top", 0)["n_subsets"]

    assert set(runs) == {12, 42, 4, 3, 5}
    assert runs[12]["variant"] == f"mass_cut/n{n}_subset{12 % n}" and runs[12]["apply_mass_cut"] is True
    assert runs[42]["variant"] == "no_mass_cut/full" and runs[42]["n_subsets"] == 1
    assert runs[4]["variant"] == f"no_mass_cut/n{n}_subset{4 % n}" and runs[4]["apply_mass_cut"] is False
    assert runs[3]["legacy"] and runs[3]["variant"] == "legacy"
    assert runs[5]["legacy"]
    assert not runs[12]["legacy"]
