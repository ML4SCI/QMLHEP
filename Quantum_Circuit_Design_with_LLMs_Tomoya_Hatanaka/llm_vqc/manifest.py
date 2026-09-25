"""Run-directory manifest writer.

Every experiment run must be reproducible from its run directory alone
(LLM-VQC_MASTER_PLAN.md Section 5.5 / Section 11). This module writes the
provenance every later phase's runner will rely on: the resolved config,
git commit + dirty flag, installed package versions, and the seed.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_vqc.config import RunConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent


class ManifestError(Exception):
    """Raised when a run manifest cannot be constructed."""


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _pip_freeze() -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return sorted(line for line in result.stdout.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def create_run_dir(base_dir: str | Path, name: str) -> Path:
    """Create a fresh, uniquely-named run directory under base_dir/name/.

    A short random suffix is appended to the timestamp so that two calls
    within the same wall-clock second (e.g. in a fast smoke-test loop)
    never collide.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    run_dir = Path(base_dir) / name / f"{timestamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(run_dir: str | Path, config: RunConfig) -> Path:
    """Write manifest.json (provenance) into run_dir. Returns the manifest path."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ManifestError(f"run_dir does not exist: {run_dir}")

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(),
        "seed": config.seed,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
    }

    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return manifest_path
