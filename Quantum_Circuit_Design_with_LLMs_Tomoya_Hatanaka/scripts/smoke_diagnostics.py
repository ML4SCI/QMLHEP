#!/usr/bin/env python
"""Phase 3 smoke workflow: run the diagnostics on a few representative
CircuitIR fixtures and store the structured results in a resumable
DiagnosticStore.

This is NOT a search algorithm and NOT a research sweep. It exercises the
full validate -> dispatch -> measure -> store pipeline on a handful of
hand-written fixtures at smoke scale. Outputs must not be reported as
scientific results.

Usage:
    python scripts/smoke_diagnostics.py configs/diagnostics_smoke.yaml
    # run again with the same config: resumes, skips completed measurements
    python scripts/smoke_diagnostics.py configs/diagnostics_smoke.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_vqc.config import load_config  # noqa: E402
from llm_vqc.diagnostics.config import DiagnosticConfig  # noqa: E402
from llm_vqc.diagnostics.harness import compute_diagnostic  # noqa: E402
from llm_vqc.diagnostics.store import (  # noqa: E402
    DiagnosticStore,
    IncompatibleDiagnosticResumeError,
)
from llm_vqc.ir.schema import (  # noqa: E402
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RepeatBlock,
    RotationLayer,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _fixture_circuits() -> dict[str, CircuitIR]:
    meas = MeasurementSpec(observable="Z", wires="all")
    enc = EncodingSpec(type="angle", gate="RY", wires="all")
    hea_block = [
        RotationLayer(gates=["RX", "RY", "RZ"], wires="all"),
        EntangleLayer(pattern="ring", gate="CNOT"),
    ]
    return {
        "idle_3q": CircuitIR(n_qubits=3, encoding=enc, layers=[], measurements=meas),
        "local_only_3q": CircuitIR(
            n_qubits=3, encoding=enc,
            layers=[RotationLayer(gates=["RX", "RY"], wires="all")], measurements=meas,
        ),
        "hea_depth1_3q": CircuitIR(
            n_qubits=3, encoding=enc,
            layers=[RepeatBlock(times=1, body=hea_block)], measurements=meas,
        ),
        "hea_depth3_3q": CircuitIR(
            n_qubits=3, encoding=enc,
            layers=[RepeatBlock(times=3, body=hea_block)], measurements=meas,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a diagnostics smoke YAML config")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--allow-incompatible", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    params = config.params
    diagnostic_names = params["diagnostics"]
    diag_config = DiagnosticConfig.model_validate(params["config"])

    run_dir = Path(args.runs_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "diagnostics.sqlite"

    repro_fields = {"diagnostics": diagnostic_names, "config": diag_config.model_dump()}
    try:
        store = DiagnosticStore.open_or_create(
            db_path, run_id=config.name, config_json=config.model_dump_json(),
            config_reproducibility_fields=repro_fields, git_sha=_git_sha(),
            created_at=datetime.now(timezone.utc).isoformat(),
            allow_incompatible=args.allow_incompatible,
        )
    except IncompatibleDiagnosticResumeError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        sys.exit(1)

    git_sha = _git_sha()
    evaluated = 0
    resumed = 0
    for circuit_name, ir in _fixture_circuits().items():
        for diagnostic_name in diagnostic_names:
            before = store.count()
            result = compute_diagnostic(
                ir, diagnostic_name, diag_config, run_seed=config.seed,
                proposal_id=f"{config.name}-{circuit_name}", store=store, git_sha=git_sha,
            )
            if result.resource_usage.cache_hit:
                resumed += 1
            elif store.count() > before:
                evaluated += 1
            primary = next(iter(result.metrics.items()), (None, None))
            print(
                f"[{circuit_name:16s} {diagnostic_name:17s}] "
                f"{primary[0]}={primary[1]} failure={result.failure_category.value} "
                f"cache_hit={result.resource_usage.cache_hit}"
            )

    print(f"\nDone. Evaluated this run: {evaluated}, resumed/skipped: {resumed}")
    print(f"Store total rows: {store.count()}")
    store.close()


if __name__ == "__main__":
    main()
