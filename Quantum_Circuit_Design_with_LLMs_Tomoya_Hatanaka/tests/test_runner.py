"""Tests for the Phase 0 config -> run-dir -> manifest scaffolding."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llm_vqc.config import RunConfig, load_config
from llm_vqc.manifest import ManifestError, create_run_dir, write_manifest
from llm_vqc.runner import run_from_config


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: my_run\nseed: 42\nparams:\n  foo: bar\n")

    config = load_config(config_path)

    assert isinstance(config, RunConfig)
    assert config.name == "my_run"
    assert config.seed == 42
    assert config.params == {"foo": "bar"}


def test_load_config_requires_name_and_seed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: only_name\n")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_create_run_dir_is_unique_and_nested(tmp_path):
    run_dir_1 = create_run_dir(tmp_path, "exp")
    run_dir_2 = create_run_dir(tmp_path, "exp")

    assert run_dir_1.is_dir()
    assert run_dir_2.is_dir()
    assert run_dir_1 != run_dir_2
    assert run_dir_1.parent == run_dir_2.parent == tmp_path / "exp"


def test_write_manifest_requires_existing_dir(tmp_path):
    config = RunConfig(name="x", seed=1)
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(ManifestError):
        write_manifest(missing_dir, config)


def test_write_manifest_records_required_provenance_fields(tmp_path):
    config = RunConfig(name="prov_test", seed=7, params={"a": 1})
    run_dir = create_run_dir(tmp_path, config.name)

    manifest_path = write_manifest(run_dir, config)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    # Resolved config
    assert manifest["config"]["name"] == "prov_test"
    assert manifest["config"]["seed"] == 7
    assert manifest["config"]["params"] == {"a": 1}

    # Seed
    assert manifest["seed"] == 7

    # Git provenance: SHA is a 40-char hex string when git is available.
    assert manifest["git_sha"] is None or len(manifest["git_sha"]) == 40
    assert isinstance(manifest["git_dirty"], (bool, type(None)))

    # Environment capture
    assert isinstance(manifest["python_version"], str) and manifest["python_version"]
    assert isinstance(manifest["pip_freeze"], list)
    assert len(manifest["pip_freeze"]) > 0  # this project has dependencies installed

    # Timestamp
    assert "created_at" in manifest


def test_run_from_config_end_to_end(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: e2e_smoke\nseed: 123\n")

    run_dir = run_from_config(config_path, runs_dir=tmp_path / "runs")

    assert run_dir.is_dir()
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["config"]["name"] == "e2e_smoke"
    assert manifest["seed"] == 123
