# Covers src/utils/reporting.py: metrics collection into a long-format table.
# _flatten_metrics_file is tested directly against a real temp JSON file;
# compute_run_statistics is tested against a small synthetic outputs/<task>/seed_*/
# tree built via workspace.get_config's own path keys, so this stays valid even if
# the underlying directory layout changes. No real pipeline run required.
import json
import os

import pandas as pd
import pytest

from src.utils import reporting, workspace


def test_flatten_metrics_file_missing_path_returns_none(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert reporting._flatten_metrics_file(str(missing), tags={"model": "x"}) is None


def test_flatten_metrics_file_merges_tags_and_serializes_nested_values(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "Test AUC": 0.91,
        "Test Accuracy": 0.88,
        "Confusion Matrix": [[10, 2], [1, 9]],
        "Backend": "ideal",
    }))

    row = reporting._flatten_metrics_file(
        str(metrics_path), tags={"task": "top", "seed": 3, "subset_id": 3, "model": "qkan_ideal"}
    )

    assert row["task"] == "top"
    assert row["seed"] == 3
    assert row["subset_id"] == 3
    assert row["model"] == "qkan_ideal"
    assert row["Test AUC"] == 0.91
    assert row["Backend"] == "ideal"
    # Nested list value must be JSON-stringified, not left as a Python list.
    assert row["Confusion Matrix"] == json.dumps([[10, 2], [1, 9]])


def test_compute_run_statistics_collects_only_existing_files_tagged_by_model(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "get_project_root", lambda: tmp_path)

    task = "top"
    for seed in (0, 1):
        config = workspace.get_config(task, seed)

        base_path = config["base_eval_metrics"]
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        with open(base_path, "w") as f:
            json.dump({"Test AUC": 0.5 + seed * 0.01}, f)

        qkan_path = config["metrics_qkan_ideal"]
        os.makedirs(os.path.dirname(qkan_path), exist_ok=True)
        with open(qkan_path, "w") as f:
            json.dump({"Test AUC": 0.6 + seed * 0.01, "Backend": "ideal"}, f)

        # Every other registry entry (retrained/symbolic/final/noisy/shots/baseline...)
        # is deliberately left missing to exercise the partial-sweep skip path.

    df = reporting.compute_run_statistics(task)

    # 2 seeds x 2 present stages = 4 rows, nothing else.
    assert len(df) == 4
    assert set(df["model"].unique()) == {"classical_base", "qkan_ideal"}
    assert set(df["seed"].unique()) == {0, 1}

    # Regime tags come from the run directory (default regime: mass cut, n_subsets
    # partitions), and subset_id = seed % n_subsets for every row.
    n_subsets = workspace.get_config(task, 0)["n_subsets"]
    for _, row in df.iterrows():
        assert row["variant"] == f"mass_cut/n{n_subsets}_subset{row['seed'] % n_subsets}"
        assert row["n_subsets"] == n_subsets
        assert row["subset_id"] == row["seed"] % n_subsets

    # Pure collection: no mean/std/aggregate columns anywhere in the output.
    assert not any(col.lower() in ("mean", "std") for col in df.columns)

    base_rows = df[df["model"] == "classical_base"].sort_values("seed")
    assert list(base_rows["Test AUC"]) == [0.5, 0.51]


def test_compute_run_statistics_tags_full_dataset_and_legacy_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "get_project_root", lambda: tmp_path)
    task = "top"

    # A full-dataset run: seed 42 would be subset 2 of 5, but must NOT be tagged that way.
    full_cfg = workspace.get_config(task, 42, full_dataset=True)
    os.makedirs(os.path.dirname(full_cfg["base_eval_metrics"]), exist_ok=True)
    with open(full_cfg["base_eval_metrics"], "w") as f:
        json.dump({"Test AUC": 0.95}, f)

    # A pre-variant-layout run sitting directly under outputs/<task>/seed_<N>/.
    legacy_metrics = tmp_path / "outputs" / task / "seed_3" / "results" / "01_base" / "base_eval_metrics.json"
    legacy_metrics.parent.mkdir(parents=True)
    legacy_metrics.write_text(json.dumps({"Test AUC": 0.9}))

    df = reporting.compute_run_statistics(task).set_index("seed")

    assert df.loc[42, "variant"] == "no_mass_cut/full"
    assert df.loc[42, "n_subsets"] == 1
    assert df.loc[42, "subset_id"] == 0
    assert not df.loc[42, "apply_mass_cut"]

    assert df.loc[3, "variant"] == "legacy"
    assert df.loc[3, "Test AUC"] == 0.9


def test_collect_eval_data_collects_arrays_and_histories(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(workspace, "get_project_root", lambda: tmp_path)
    task = "top"
    config = workspace.get_config(task, 0)

    def save(key, values):
        os.makedirs(os.path.dirname(config[key]), exist_ok=True)
        np.save(config[key], np.array(values, dtype=np.float32).reshape(-1, 1))

    # Trained QKAN stage: arrays + a history without train_auc.
    save("qkan_eval_data_true_ideal", [0, 1, 1])
    save("qkan_eval_data_probs_ideal", [0.2, 0.7, 0.9])
    save("qkan_eval_data_binary_ideal", [0, 1, 1])
    with open(config["history_ideal_loss"], "w") as f:
        json.dump({"train_loss": [0.7, 0.6], "val_loss": [0.69, 0.65], "val_auc": [0.6, 0.7]}, f)

    # Untrained RF: probs only (true/binary missing), no history.
    save("rf_eval_data_probs", [0.4, 0.5])

    df = reporting.collect_eval_data(task).set_index("model")

    assert set(df.index) == {"qkan_ideal", "random_forest"}

    qkan = df.loc["qkan_ideal"]
    assert qkan["seed"] == 0
    assert list(qkan["y_true"]) == [0.0, 1.0, 1.0]
    assert list(qkan["y_probs"]) == pytest.approx([0.2, 0.7, 0.9])
    assert list(qkan["val_auc"]) == [0.6, 0.7]
    assert qkan["train_auc"] is None
    assert qkan["n_events"] == 3
    assert qkan["n_epochs"] == 2

    rf = df.loc["random_forest"]
    assert rf["y_true"] is None and rf["y_binary"] is None
    assert rf["val_loss"] is None
    assert rf["n_events"] == 2
    assert pd.isna(rf["n_epochs"])
