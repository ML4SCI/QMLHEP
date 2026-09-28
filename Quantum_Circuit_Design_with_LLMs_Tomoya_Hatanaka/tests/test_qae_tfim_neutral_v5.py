"""Offline invariants for the QAE-TFIM v5 free-form redesign benchmark."""
import numpy as np
import pandas as pd

from llm_vqc.experiments.qae_tfim import neutral_v5
from llm_vqc.experiments.qae_tfim.pilot import state_sets


def test_redesign_prompt_contains_incumbent_and_freedom_but_no_one_change_rule():
    rng = np.random.default_rng(3)
    incumbent = neutral_v5.sample_neutral_arch(rng)
    prompt = neutral_v5.redesign_prompt(incumbent, 0.9312)
    assert "Current best architecture" in prompt
    assert "0.9312" in prompt
    assert "redesign the architecture freely" in prompt
    assert "entire architecture" in prompt
    # the v5 point: no one-change restriction, no stay-close instruction
    assert "exactly one" not in prompt.lower()
    assert "one structural" not in prompt.lower()
    assert "stay close" not in prompt.lower()
    # no angles, no test leakage
    assert "Do not propose numerical rotation angles" in prompt
    assert "protected" not in prompt.lower()
    # incumbent gates are actually serialized into the prompt
    first = incumbent[0]
    token = ('"g":"R' + first["axis"]) if first["type"] == "R" else '"g":"CNOT"'
    assert token in prompt.replace(" ", "")


def test_redesign_prompt_requests_strategy_metadata():
    rng = np.random.default_rng(4)
    prompt = neutral_v5.redesign_prompt(neutral_v5.sample_neutral_arch(rng), 0.9)
    for token in ("strategy", "local_adjustment", "global_redesign", "rationale",
                  "preserved", "changed", "analysis only"):
        assert token in prompt


def test_greedy_v5_is_4_plus_4_one_change_and_deterministic():
    train, val, test = state_sets(0)
    rows_a = neutral_v5.run_greedy_seed(0, train, val, test)
    rows_b = neutral_v5.run_greedy_seed(0, train, val, test)
    assert [r["val_loss"] for r in rows_a] == [r["val_loss"] for r in rows_b]
    assert [r["phase"] for r in rows_a] == ["explore"] * 4 + ["refine"] * 4
    # one-change moves: edit distance of each refinement proposal from its
    # incumbent is at most 2/16 slots (a swap changes two positions)
    for r in rows_a[4:]:
        assert r["edit_distance"] <= 2 / 16 + 1e-9


def test_random_v5_stream_differs_from_v4():
    from llm_vqc.experiments.qae_tfim import neutral_v4

    train, val, test = state_sets(0)
    v4_rows = neutral_v4.run_random_seed(0, train, val, test)
    v5_rows = neutral_v5.run_random_seed(0, train, val, test)
    assert [r["val_loss"] for r in v4_rows] != [r["val_loss"] for r in v5_rows]


def test_redesign_diagnostics_locality_split():
    frame = pd.DataFrame([
        {"seed": 0, "method": "LLM-Closed", "order": 1, "phase": "explore",
         "fallback": False, "invalid_errors": "", "edit_distance": np.nan,
         "strategy": "", "val_loss": 0.30, "val_fid": 0.70, "test_fid": 0.70},
        # accepted local proposal (2 slots changed)
        {"seed": 0, "method": "LLM-Closed", "order": 5, "phase": "refine",
         "fallback": False, "invalid_errors": "", "edit_distance": 2 / 16,
         "strategy": "local_adjustment", "val_loss": 0.20, "val_fid": 0.80,
         "test_fid": 0.80},
        # rejected global proposal (12 slots changed)
        {"seed": 0, "method": "LLM-Closed", "order": 6, "phase": "refine",
         "fallback": False, "invalid_errors": "", "edit_distance": 12 / 16,
         "strategy": "global_redesign", "val_loss": 0.40, "val_fid": 0.60,
         "test_fid": 0.60},
    ])
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        for seed in neutral_v5.VERIFY_SEEDS:
            (destination / f"llm_closed_seed{seed}.json").write_text(
                json.dumps({"rows": [], "architectures": [], "refine_failures": []})
            )
        d = neutral_v5.redesign_diagnostics(frame, destination)
    assert d["local_proposals_le_4_slots"] == {"n": 1, "accepted": 1}
    assert d["global_proposals_gt_4_slots"] == {"n": 1, "accepted": 0}
    assert d["strategy_stats"]["local_adjustment"]["accepted"] == 1
    assert d["strategy_stats"]["global_redesign"]["accepted"] == 0


def test_arch_to_gate_list_round_trip():
    rng = np.random.default_rng(9)
    arch = neutral_v5.sample_neutral_arch(rng)
    gates = neutral_v5._arch_to_gate_list(arch)
    parsed, errors = neutral_v5.parse_candidate({"name": "x", "gates": gates})
    assert errors == []
    from llm_vqc.experiments.qae_tfim.pilot import architecture_key
    assert architecture_key(parsed) == architecture_key(arch)
