#!/usr/bin/env python
"""Run the capacity-controlled T1 experiment (search + baselines + protected test).

Every search arm proposes from the SAME controlled grammar
(`llm_vqc.experiments.capacity_controlled.space`): identical qubits, encoding,
measurement, classical embedding/head, and EXACTLY 12 trainable quantum params
(105 total trainable params) for every candidate. Only the internal VQC
architecture varies. The shared harness enforces the controlled invariants via
`extra_validator` and the shared explicit init policy via `init_policy`.

Fails loudly (RuntimeError) if any evaluated candidate's total trainable
parameter count is not exactly 105.

No LLM arms. No LLM API calls. No amplitude encoding. Test metrics are computed
once per completed run, after selection, and never enter search feedback.

Usage:
  python scripts/capacity_controlled/run_experiment.py --budget 5  --seeds 0            # smoke
  python scripts/capacity_controlled/run_experiment.py --budget 25 --seeds 0,1,2,3,4    # pilot
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from llm_vqc.evaluation.final_test import evaluate_on_test
from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import derive_child_seed, train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import baselines as BL
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.arms import (
    ControlledEvolutionaryArm,
    ControlledEvoConfig,
    ControlledGreedyArm,
    ControlledRandomArm,
)
from llm_vqc.experiments.capacity_controlled.init_policy import (
    apply_explicit_init,
    init_policy_metadata,
)
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

ROOT = Path(__file__).resolve().parents[2]
DATA_SPLIT_SEED = 0
TASK_NAME = "T1"
GREEDY_K = 5
EVO_CFG = ControlledEvoConfig(mu=5, lambda_=5)
ARMS = ("controlled_random", "controlled_greedy", "controlled_evolutionary")


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _make_arm(name: str):
    if name == "controlled_random":
        return ControlledRandomArm(lower_is_better=True)
    if name == "controlled_greedy":
        return ControlledGreedyArm(lower_is_better=True, k=GREEDY_K)
    if name == "controlled_evolutionary":
        return ControlledEvolutionaryArm(lower_is_better=True, config=EVO_CFG)
    raise ValueError(name)


def _assert_capacity(ir: CircuitIR) -> int:
    m = HybridQNNModel(ir, raw_feature_dim=S.RAW_FEATURE_DIM, head_out_dim=1)
    total = sum(p.numel() for p in m.parameters())
    if total != S.TOTAL_TRAINABLE_PARAMS:
        raise RuntimeError(
            f"CAPACITY VIOLATION: candidate has {total} trainable params, expected "
            f"{S.TOTAL_TRAINABLE_PARAMS}. Structural hash={structural_hash(ir)[:12]}")
    return total


def run_search(budget: int, seeds: list[int], run_dir: Path) -> dict:
    task = T1GaussianPeakTask()
    train_val = task.build(seed=DATA_SPLIT_SEED)
    test_split = task.build_test(seed=DATA_SPLIT_SEED)
    cfg = TrainingConfig()  # AdamW, lr 0.05, 20 epochs, batch 16 (master-plan defaults)
    git_sha = _git_sha()
    db_path = run_dir / "results.sqlite"

    runs = []
    for arm_name in ARMS:
        for seed in seeds:
            run_id = f"{arm_name}_s{seed}"
            store = ResultStore.open_or_create(
                db_path, run_id=run_id, config_json="{}",
                config_reproducibility_fields={
                    "space": S.SPACE_NAME, "arm": arm_name, "budget": budget,
                    "data_split_seed": DATA_SPLIT_SEED, "repetition_seed": seed,
                    "training": cfg.model_dump(), "init_policy": init_policy_metadata()["name"],
                    "quantum_param_count": S.QUANTUM_PARAM_COUNT},
                git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat())
            arm = _make_arm(arm_name)
            runner = SearchRunner(
                arm, TASK_NAME, train_val, cfg, budget_limit=budget, run_seed=seed,
                result_store=store, run_id=run_id, git_sha=git_sha,
                extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
            result = runner.run()

            # protected final test on the selected circuit (no retrain)
            test_rmse = None
            h = result.selected_structural_hash
            ts = result.selected_train_seed
            if h is not None:
                w = store.get_trained_weights(TASK_NAME, h, ts)
                c = store.get_cached(TASK_NAME, h, ts)
                if w and c and c.circuit_canonical_json:
                    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
                    _assert_capacity(ir)
                    tr = evaluate_on_test(ir, w["classical_state"], test_split,
                                          train_val.spec, ts)
                    test_rmse = tr.test_metric_value
            print(f"[{run_id}] ledger={result.ledger_summary} "
                  f"val={result.selected_val_metric_value} test={test_rmse}")
            runs.append({
                "run_id": run_id, "arm": arm_name, "seed": seed,
                "ledger_summary": result.ledger_summary,
                "selected_structural_hash": h,
                "selected_val_metric_value": result.selected_val_metric_value,
                "test_metric_value": test_rmse})
            store.close()
    return {"runs": runs, "git_sha": git_sha}


def run_baselines(seeds: list[int]) -> list[dict]:
    task = T1GaussianPeakTask()
    tv = task.build(seed=DATA_SPLIT_SEED)
    test = task.build_test(seed=DATA_SPLIT_SEED)
    cfg = TrainingConfig()
    out = []

    # C1 / C2 classical
    for name, factory in (("C1_bottleneck", BL.BottleneckModel), ("C2_param_matched_mlp", BL.MLPBaseline)):
        for seed in seeds:
            model = factory()
            ts = derive_child_seed(seed, "classical_baseline", name)
            res = BL.train_classical_baseline(model, tv, cfg, ts)
            test_model = factory()
            test_rmse = BL.evaluate_classical_on_test(test_model, res["state_dict"], test,
                                                      tv.spec.metric_name)
            out.append({"baseline": name, "kind": "classical", "seed": seed,
                        "total_params": res["total_params"],
                        "val_rmse": res["final_val_metric"], "test_rmse": test_rmse,
                        "wall_clock_seconds": res["wall_clock_seconds"]})
            print(f"[baseline {name} s{seed}] params={res['total_params']} "
                  f"val={res['final_val_metric']:.6f} test={test_rmse:.6f}")

    # Q1 / Q2 fixed VQC
    for name, genome in (("Q1_fixed_hea", BL.fixed_hea_genome()),
                         ("Q2_fixed_random", BL.fixed_random_genome())):
        ir = S.genome_to_ir(genome)
        _assert_capacity(ir)
        h = structural_hash(ir)
        for seed in seeds:
            ts = train_seed_for_circuit(seed, h)
            tro = train_model(ir, tv, cfg, ts, init_policy=apply_explicit_init)
            if not tro.success:
                out.append({"baseline": name, "kind": "fixed_vqc", "seed": seed,
                            "total_params": S.TOTAL_TRAINABLE_PARAMS, "val_rmse": None,
                            "test_rmse": None, "wall_clock_seconds": tro.wall_clock_seconds,
                            "error": tro.error_message})
                continue
            fr = evaluate_on_test(ir, tro.trained_classical_state, test, tv.spec, ts)
            out.append({"baseline": name, "kind": "fixed_vqc", "seed": seed,
                        "total_params": S.TOTAL_TRAINABLE_PARAMS,
                        "val_rmse": tro.final_val_metric, "test_rmse": fr.test_metric_value,
                        "wall_clock_seconds": tro.wall_clock_seconds,
                        "structural_hash": h[:12]})
            print(f"[baseline {name} s{seed}] val={tro.final_val_metric:.6f} "
                  f"test={fr.test_metric_value:.6f}")
    return out


def write_config(run_dir: Path, budget: int, seeds: list[int], git_sha: str | None) -> None:
    out_pkg = ROOT / "outputs" / "capacity_controlled_t1_v1"
    out_pkg.mkdir(parents=True, exist_ok=True)
    cfg = TrainingConfig()
    config = {
        "space_name": S.SPACE_NAME,
        "task": "T1_gaussian_peak",
        "data_split_seed": DATA_SPLIT_SEED,
        "budget": budget, "repetition_seeds": seeds, "arms": list(ARMS),
        "greedy_k": GREEDY_K, "evo_config": EVO_CFG.model_dump(),
        "fixed_controls": {
            "n_qubits": S.N_QUBITS, "encoding_type": "angle", "encoding_gate": S.ENCODING_GATE,
            "encoding_wires": "all", "amplitude_encoding": "forbidden",
            "measurement_observable": S.MEASUREMENT_OBSERVABLE, "measurement_wires": "all",
            "classical_embed": f"Linear(21->{S.N_QUBITS})", "classical_head": f"Linear({S.N_QUBITS}->1)",
            "quantum_param_count": S.QUANTUM_PARAM_COUNT,
            "embed_params": S.EMBED_PARAMS, "head_params": S.HEAD_PARAMS,
            "total_trainable_params": S.TOTAL_TRAINABLE_PARAMS,
        },
        "structural_limits": {"max_blocks": S.MAX_BLOCKS, "max_depth": S.MAX_DEPTH,
                              "max_total_gates": S.MAX_TOTAL_GATES,
                              "max_two_qubit_gates": S.MAX_TWO_QUBIT_GATES},
        "param_block_kinds": list(S.PARAM_BLOCK_KINDS),
        "free_block_templates": [list(t) for t in S.FREE_BLOCK_TEMPLATES],
        "training_config": cfg.model_dump(),
        "init_policy": init_policy_metadata(),
        "baselines": {"C1": "bottleneck 21->4->1 (93 params)",
                      "C2": "MLP 21->4->3->1 (107 params, +2 vs 105)",
                      "Q1": "fixed hardware-efficient RY x3 + CNOT-ring x2 (105 params)",
                      "Q2": f"fixed random architecture, seed {BL.Q2_FIXED_SEED} (105 params)"},
        "git_sha": git_sha,
        "integrity": ["no LLM arms", "no LLM API calls", "no amplitude encoding",
                      "validation-only search feedback", "protected test scored once after selection",
                      "quantum param count NOT tuned after seeing results",
                      "identical init policy for every arm"],
    }
    (out_pkg / "experiment_config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {out_pkg}/experiment_config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seeds", type=str, required=True, help="comma-separated, e.g. 0,1,2,3,4")
    ap.add_argument("--run-dir", type=str,
                    default=str(ROOT / "runs" / "capacity_controlled_t1_v1"))
    ap.add_argument("--tag", type=str, default="pilot")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"CAPACITY-CONTROLLED T1 v1 — tag={args.tag} budget={args.budget} seeds={seeds}")
    print(f"controlled: 4q angle-RY, Z-all, embed 21->4, head 4->1, "
          f"{S.QUANTUM_PARAM_COUNT} quantum params, {S.TOTAL_TRAINABLE_PARAMS} total")
    print("=" * 72)

    search = run_search(args.budget, seeds, run_dir)
    baselines = run_baselines(seeds)
    write_config(run_dir, args.budget, seeds, search["git_sha"])

    summary = {"tag": args.tag, "budget": args.budget, "seeds": seeds,
               "git_sha": search["git_sha"], "runs": search["runs"], "baselines": baselines}
    (run_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {run_dir}/{args.tag}_summary.json")
    print("Next: scripts/capacity_controlled/analyze.py")


if __name__ == "__main__":
    main()
