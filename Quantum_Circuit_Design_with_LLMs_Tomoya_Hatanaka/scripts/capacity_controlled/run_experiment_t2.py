#!/usr/bin/env python
"""Run the T2-v1 matrix: primary search (5 splits x 6 seeds x 3 arms, B=25) +
classical baselines C0-C6 + paired quantum ablations (freeze/train, product/
entangled). Selection on validation log-loss; protected-test primary metric is
balanced-accuracy error (computed once after selection). Fails loudly if any
primary VQC candidate != 61 trainable params.

Resumable: per-split SQLite stores for search; per-split aux JSON for baselines/
ablations (skipped if already present). No LLM API.

Usage:
  python scripts/capacity_controlled/run_experiment_t2.py --splits 0,1,2,3,4 --seeds 0,1,2,3,4,5 --budget 25 --tag pilot
  python scripts/capacity_controlled/run_experiment_t2.py --splits 0 --seeds 0 --budget 5 --tag smoke --abl-seeds 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import derive_child_seed, train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled import t2_ablations as AB
from llm_vqc.experiments.capacity_controlled import t2_baselines as CB
from llm_vqc.experiments.capacity_controlled import t2_task as T2
from llm_vqc.experiments.capacity_controlled.arms import (
    ControlledEvolutionaryArm,
    ControlledEvoConfig,
    ControlledGreedyArm,
    ControlledRandomArm,
)
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init, init_policy_metadata
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.search.runner import SearchRunner

ROOT = Path(__file__).resolve().parents[2]
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


def run_search_cell(db, arm_name, split, seed, budget, tv, test, cfg, git_sha):
    run_id = f"{arm_name}_split{split}_s{seed}"
    store = ResultStore.open_or_create(
        db, run_id=run_id, config_json="{}",
        config_reproducibility_fields={"space": "T2_capacity_controlled_v1", "arm": arm_name,
                                       "budget": budget, "data_split_seed": split,
                                       "repetition_seed": seed, "training": cfg.model_dump(),
                                       "init_policy": init_policy_metadata()["name"],
                                       "selection_metric": T2.SELECTION_METRIC},
        git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat())
    runner = SearchRunner(_make_arm(arm_name), T2.TASK_NAME, tv, cfg, budget_limit=budget,
                          run_seed=seed, result_store=store, run_id=run_id, git_sha=git_sha,
                          extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
    result = runner.run()
    row = {"run_id": run_id, "arm": arm_name, "split": split, "seed": seed,
           "ledger_summary": result.ledger_summary,
           "selected_structural_hash": result.selected_structural_hash,
           "selected_val_logloss": result.selected_val_metric_value, "test": None}
    h, ts = result.selected_structural_hash, result.selected_train_seed
    if h is not None:
        w = store.get_trained_weights(T2.TASK_NAME, h, ts)
        c = store.get_cached(T2.TASK_NAME, h, ts)
        if w and c and c.circuit_canonical_json:
            ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
            T2.assert_capacity(ir)
            row["test"] = T2.evaluate_all_test_metrics(ir, w["classical_state"], test)
    store.close()
    return row


def _train_and_test(ir, tv, test, train_seed, freeze_quantum=False):
    T2.assert_capacity(ir)
    out = train_model(ir, tv, TrainingConfig(), train_seed, init_policy=apply_explicit_init,
                      freeze_quantum=freeze_quantum)
    if not out.success:
        return {"error": out.error_message, "balanced_accuracy_error": None}
    md = T2.evaluate_all_test_metrics(ir, out.trained_classical_state, test)
    md["val_logloss"] = out.final_val_metric
    md["wall_clock_seconds"] = out.wall_clock_seconds
    return md


def run_aux(split, seeds, abl_seeds, tv, test, aux_path: Path):
    aux = json.loads(aux_path.read_text()) if aux_path.exists() else {}

    # classical baselines per seed
    aux.setdefault("classical", {})
    for seed in seeds:
        if str(seed) not in aux["classical"]:
            aux["classical"][str(seed)] = CB.run_all_classical(tv, test, seed=seed)
            aux_path.write_text(json.dumps(aux, indent=2))

    # ablation A: frozen vs trainable, 20 archs x abl_seeds
    if "freeze_train" not in aux:
        rows = []
        for ai, g in enumerate(AB.predeclared_architectures()):
            ir = S.genome_to_ir(g); h = structural_hash(ir)
            twoq = AB.two_qubit_count(g)
            for seed in abl_seeds:
                ts = train_seed_for_circuit(seed, h)  # identical init for both variants
                tr = _train_and_test(ir, tv, test, ts, freeze_quantum=False)
                fr = _train_and_test(ir, tv, test, ts, freeze_quantum=True)
                rows.append({"arch": ai, "split": split, "seed": seed, "two_qubit": twoq,
                             "hash": h[:10],
                             "trainable_bae": tr["balanced_accuracy_error"],
                             "frozen_bae": fr["balanced_accuracy_error"]})
        aux["freeze_train"] = rows
        aux_path.write_text(json.dumps(aux, indent=2))

    # ablation B: product vs entangled, 20 pairs x abl_seeds (paired init seed)
    if "product_entangled" not in aux:
        rows = []
        for pi, (e, p) in enumerate(AB.entangled_pairs(20)):
            ire = S.genome_to_ir(e); irp = S.genome_to_ir(p)
            mm = AB.pair_gate_mismatch(e, p)
            for seed in abl_seeds:
                pair_seed = derive_child_seed(seed, "pairB", str(pi))  # identical init for both
                re = _train_and_test(ire, tv, test, pair_seed)
                rp = _train_and_test(irp, tv, test, pair_seed)
                rows.append({"pair": pi, "split": split, "seed": seed,
                             "entangled_bae": re["balanced_accuracy_error"],
                             "product_bae": rp["balanced_accuracy_error"],
                             "entangled_two_qubit": mm["entangled_two_qubit"],
                             "gate_count_mismatch": mm["gate_count_mismatch"],
                             "param_count_preserved": mm["param_count_preserved"]})
        aux["product_entangled"] = rows
        aux_path.write_text(json.dumps(aux, indent=2))
    return aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--abl-seeds", default="0,1,2")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_t2_v1"))
    args = ap.parse_args()
    splits = [int(x) for x in args.splits.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    abl_seeds = [int(x) for x in args.abl_seeds.split(",")]
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainingConfig(); git_sha = _git_sha()

    print("=" * 72)
    print(f"T2-v1 tag={args.tag} splits={splits} seeds={seeds} abl_seeds={abl_seeds} budget={args.budget}")
    print(f"controlled: 4q angle-RY Z-all, {T2.TOTAL_TRAINABLE_PARAMS} total params; "
          f"select on val {T2.SELECTION_METRIC}; primary test metric {T2.PRIMARY_TEST_METRIC}")
    print("=" * 72)

    all_runs, all_aux = [], {}
    for split in splits:
        tv = T2.build_t2(split); test = T2.build_t2_test(split)
        sd = run_dir / f"split_{split}"; sd.mkdir(parents=True, exist_ok=True)
        db = sd / "results.sqlite"
        for arm in ARMS:
            for seed in seeds:
                row = run_search_cell(db, arm, split, seed, args.budget, tv, test, cfg, git_sha)
                all_runs.append(row)
                bae = row["test"]["balanced_accuracy_error"] if row["test"] else None
                print(f"[{row['run_id']}] val_ll={row['selected_val_logloss']} test_bae={bae}")
        all_aux[str(split)] = run_aux(split, seeds, abl_seeds, tv, test, sd / "aux_results.json")
        print(f"[split {split}] aux done (classical + paired ablations)")

    summary = {"tag": args.tag, "splits": splits, "seeds": seeds, "abl_seeds": abl_seeds,
               "budget": args.budget, "git_sha": git_sha, "arms": list(ARMS), "runs": all_runs, "aux": all_aux}
    (run_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {run_dir}/{args.tag}_summary.json")


if __name__ == "__main__":
    main()
