#!/usr/bin/env python
"""Run the v2 capacity-controlled matrix: 3 splits x 10 seeds x 3 search arms
(B=25) + quantum ablations (Q0/QF/QP/QE/QR) + classical baselines (C1/C2/C3) +
analytical estimators (A1-A4). Protected test scored once per cell after
selection. Fails loudly if any controlled VQC candidate != 105 trainable params.

Resumable: search cells persist in per-split SQLite stores (re-running a
completed cell just re-finalizes, no extra budget); ablation/baseline/analytical
cells persist to per-split aux JSON and are skipped if already present.

Usage:
  python scripts/capacity_controlled/run_experiment_v2.py --splits 0,1,2 --seeds 0,1,2,3,4,5,6,7,8,9 --budget 25 --tag pilot
  python scripts/capacity_controlled/run_experiment_v2.py --splits 0 --seeds 0 --budget 5 --tag smoke   # smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from llm_vqc.evaluation.final_test import evaluate_on_test
from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import derive_child_seed, train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import ablations as AB
from llm_vqc.experiments.capacity_controlled import analytical as AN
from llm_vqc.experiments.capacity_controlled import baselines as BL
from llm_vqc.experiments.capacity_controlled import classical_search as CS
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.arms import (
    ControlledEvolutionaryArm,
    ControlledEvoConfig,
    ControlledGreedyArm,
    ControlledRandomArm,
)
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init, init_policy_metadata
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

ROOT = Path(__file__).resolve().parents[2]
TASK = "T1"
GREEDY_K = 5
EVO_CFG = ControlledEvoConfig(mu=5, lambda_=5)
ARMS = ("controlled_random", "controlled_evolutionary", "controlled_greedy")


def _git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _make_arm(name):
    if name == "controlled_random":
        return ControlledRandomArm(lower_is_better=True)
    if name == "controlled_evolutionary":
        return ControlledEvolutionaryArm(lower_is_better=True, config=EVO_CFG)
    if name == "controlled_greedy":
        return ControlledGreedyArm(lower_is_better=True, k=GREEDY_K)
    raise ValueError(name)


def _assert_capacity(ir):
    m = HybridQNNModel(ir, raw_feature_dim=S.RAW_FEATURE_DIM, head_out_dim=1)
    total = sum(p.numel() for p in m.parameters())
    if total != S.TOTAL_TRAINABLE_PARAMS:
        raise RuntimeError(f"CAPACITY VIOLATION: {total} != {S.TOTAL_TRAINABLE_PARAMS} "
                           f"(hash {structural_hash(ir)[:12]})")
    return total


def _cost(ir):
    c = circuit_cost_summary(ir)
    return {"depth": c.depth, "gate_count": c.gate_count,
            "two_qubit_gate_count": c.two_qubit_gate_count}


def run_search_cell(store, arm_name, split, seed, budget, tv, test, cfg, git_sha):
    run_id = f"{arm_name}_split{split}_s{seed}"
    store2 = ResultStore.open_or_create(
        store, run_id=run_id, config_json="{}",
        config_reproducibility_fields={"space": S.SPACE_NAME, "arm": arm_name, "budget": budget,
                                       "data_split_seed": split, "repetition_seed": seed,
                                       "training": cfg.model_dump(),
                                       "init_policy": init_policy_metadata()["name"]},
        git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat())
    runner = SearchRunner(_make_arm(arm_name), TASK, tv, cfg, budget_limit=budget, run_seed=seed,
                          result_store=store2, run_id=run_id, git_sha=git_sha,
                          extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
    result = runner.run()
    test_rmse = None
    h, ts = result.selected_structural_hash, result.selected_train_seed
    if h is not None:
        w = store2.get_trained_weights(TASK, h, ts)
        c = store2.get_cached(TASK, h, ts)
        if w and c and c.circuit_canonical_json:
            ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
            _assert_capacity(ir)
            test_rmse = evaluate_on_test(ir, w["classical_state"], test, tv.spec, ts).test_metric_value
    store2.close()
    return {"run_id": run_id, "arm": arm_name, "split": split, "seed": seed,
            "ledger_summary": result.ledger_summary,
            "selected_structural_hash": h,
            "selected_val_metric_value": result.selected_val_metric_value,
            "test_metric_value": test_rmse}


def train_fixed_vqc(ir, tv, test, seed, freeze_quantum=False):
    _assert_capacity(ir)
    h = structural_hash(ir)
    ts = train_seed_for_circuit(seed, h)
    out = train_model(ir, tv, TrainingConfig(), ts, init_policy=apply_explicit_init,
                      freeze_quantum=freeze_quantum)
    if not out.success:
        return {"val_rmse": None, "test_rmse": None, "error": out.error_message,
                "wall_clock_seconds": out.wall_clock_seconds}
    test_rmse = evaluate_on_test(ir, out.trained_classical_state, test, tv.spec, ts).test_metric_value
    return {"val_rmse": out.final_val_metric, "test_rmse": test_rmse,
            "wall_clock_seconds": out.wall_clock_seconds, "structural_hash": h[:12], **_cost(ir)}


def run_aux_for_split(split, seeds, tv, test, aux_path: Path):
    """Ablations + classical + analytical for one split, cached to aux_path."""
    aux = json.loads(aux_path.read_text()) if aux_path.exists() else {}
    cfg = TrainingConfig()

    # analytical (no seed dependence) — computed once per split
    if "analytical" not in aux:
        analytical = []
        for name, fn in AN.ANALYTICAL_ESTIMATORS.items():
            vpred, vfail = fn(tv.val.features)
            tpred, tfail = fn(test.features)
            analytical.append({"estimator": name, "split": split,
                               "val_rmse": AN.rmse(vpred, tv.val.targets),
                               "test_rmse": AN.rmse(tpred, test.targets),
                               "val_failures": vfail, "test_failures": tfail})
        aux["analytical"] = analytical
        aux_path.write_text(json.dumps(aux, indent=2))

    # per-seed ablations + classical
    aux.setdefault("ablations", {})
    aux.setdefault("classical", {})
    qe = AB.HEA_IR if hasattr(AB, "HEA_IR") else S.genome_to_ir(BL.fixed_hea_genome())
    qr = S.genome_to_ir(BL.fixed_random_genome())
    qp = S.genome_to_ir(AB.product_state_genome())
    for seed in seeds:
        key = str(seed)
        if key not in aux["ablations"]:
            row = {}
            # Q0 no-quantum (classical shell)
            q0 = AB.NoQuantumModel()
            ts0 = derive_child_seed(seed, "q0_no_quantum", str(split))
            r0 = BL.train_classical_baseline(q0, tv, cfg, ts0)
            q0t = AB.NoQuantumModel()
            row["Q0_no_quantum"] = {"val_rmse": r0["final_val_metric"],
                                    "test_rmse": BL.evaluate_classical_on_test(
                                        q0t, r0["state_dict"], test, tv.spec.metric_name),
                                    "total_params": r0["total_params"], "trainable_params": 93,
                                    "frozen_quantum_params": 0,
                                    "wall_clock_seconds": r0["wall_clock_seconds"]}
            # QF frozen random quantum
            rf = train_fixed_vqc(qr, tv, test, seed, freeze_quantum=True)
            rf.update({"total_params": 105, "trainable_params": 93, "frozen_quantum_params": 12})
            row["QF_frozen_random"] = rf
            # QP product state
            rp = train_fixed_vqc(qp, tv, test, seed)
            rp.update({"total_params": 105, "trainable_params": 105, "frozen_quantum_params": 0})
            row["QP_product_state"] = rp
            # QE fixed entangled (Q1)
            re = train_fixed_vqc(qe, tv, test, seed)
            re.update({"total_params": 105, "trainable_params": 105, "frozen_quantum_params": 0})
            row["QE_fixed_entangled"] = re
            # QR fixed random (Q2)
            rr = train_fixed_vqc(qr, tv, test, seed)
            rr.update({"total_params": 105, "trainable_params": 105, "frozen_quantum_params": 0})
            row["QR_fixed_random"] = rr
            aux["ablations"][key] = row
            aux_path.write_text(json.dumps(aux, indent=2))
        if key not in aux["classical"]:
            c1 = BL.BottleneckModel(); ts1 = derive_child_seed(seed, "C1", str(split))
            r1 = BL.train_classical_baseline(c1, tv, cfg, ts1)
            c1t = BL.BottleneckModel()
            c2 = BL.MLPBaseline(); ts2 = derive_child_seed(seed, "C2", str(split))
            r2 = BL.train_classical_baseline(c2, tv, cfg, ts2)
            c2t = BL.MLPBaseline()
            c3 = CS.run_classical_search(tv, test, seed=seed)
            aux["classical"][key] = {
                "C1_bottleneck": {"val_rmse": r1["final_val_metric"],
                                  "test_rmse": BL.evaluate_classical_on_test(c1t, r1["state_dict"], test, tv.spec.metric_name),
                                  "total_params": r1["total_params"]},
                "C2_param_matched_mlp": {"val_rmse": r2["final_val_metric"],
                                         "test_rmse": BL.evaluate_classical_on_test(c2t, r2["state_dict"], test, tv.spec.metric_name),
                                         "total_params": r2["total_params"]},
                "C3_classical_search": {"val_rmse": c3["selected_val_rmse"], "test_rmse": c3["test_rmse"],
                                        "total_params": c3["selected"]["total_params"],
                                        "param_mismatch_vs_105": c3["param_mismatch_vs_105"],
                                        "selected": c3["selected"], "n_candidates": c3["n_candidates"]},
            }
            aux_path.write_text(json.dumps(aux, indent=2))
    return aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="0,1,2")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_t1_v2"))
    args = ap.parse_args()
    splits = [int(x) for x in args.splits.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainingConfig(); git_sha = _git_sha()

    print("=" * 72)
    print(f"V2 tag={args.tag} splits={splits} seeds={seeds} budget={args.budget} arms={ARMS}")
    print(f"controlled: 4q angle-RY Z-all, {S.TOTAL_TRAINABLE_PARAMS} total params/candidate")
    print("=" * 72)

    task = T1GaussianPeakTask()
    all_runs, all_aux = [], {}
    for split in splits:
        tv = task.build(seed=split); test = task.build_test(seed=split)
        split_dir = run_dir / f"split_{split}"; split_dir.mkdir(parents=True, exist_ok=True)
        db = split_dir / "results.sqlite"
        for arm in ARMS:
            for seed in seeds:
                row = run_search_cell(db, arm, split, seed, args.budget, tv, test, cfg, git_sha)
                all_runs.append(row)
                print(f"[{row['run_id']}] val={row['selected_val_metric_value']} test={row['test_metric_value']}")
        aux = run_aux_for_split(split, seeds, tv, test, split_dir / "aux_results.json")
        all_aux[str(split)] = aux
        print(f"[split {split}] aux done (ablations+classical+analytical)")

    summary = {"tag": args.tag, "splits": splits, "seeds": seeds, "budget": args.budget,
               "git_sha": git_sha, "arms": list(ARMS), "runs": all_runs, "aux": all_aux}
    (run_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {run_dir}/{args.tag}_summary.json")
    print("Next: scripts/capacity_controlled/analyze_v2.py")


if __name__ == "__main__":
    main()
