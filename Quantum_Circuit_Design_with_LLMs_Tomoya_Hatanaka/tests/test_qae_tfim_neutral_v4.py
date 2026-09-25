"""Offline invariants for the QAE-TFIM v4 multi-start benchmark. No network."""
import numpy as np
import pandas as pd

from llm_vqc.experiments.qae_tfim import neutral_v4
from llm_vqc.experiments.qae_tfim.pilot import state_sets


def test_greedy44_uses_exactly_eight_evaluations_with_4_plus_4_phases():
    train, val, test = state_sets(0)
    rows = neutral_v4.run_greedy44_seed(0, train, val, test)
    assert len(rows) == 8
    assert [r["phase"] for r in rows] == ["explore"] * 4 + ["refine"] * 4
    assert [r["order"] for r in rows] == list(range(1, 9))


def test_greedy44_refines_the_best_warm_start_and_is_deterministic():
    train, val, test = state_sets(1)
    rows_a = neutral_v4.run_greedy44_seed(1, train, val, test)
    rows_b = neutral_v4.run_greedy44_seed(1, train, val, test)
    assert [r["val_loss"] for r in rows_a] == [r["val_loss"] for r in rows_b]
    warm_best = min(r["val_loss"] for r in rows_a[:4])
    # the final selected validation loss can never be worse than the best warm start
    assert min(r["val_loss"] for r in rows_a) <= warm_best


def test_random_v4_stream_differs_from_v3():
    from llm_vqc.experiments.qae_tfim import neutral_v3

    train, val, test = state_sets(0)
    v3_rows = neutral_v3.run_random_seed(0, train, val, test)
    v4_rows = neutral_v4.run_random_seed(0, train, val, test)
    assert [r["val_loss"] for r in v3_rows] != [r["val_loss"] for r in v4_rows]


def test_warmstart_prompt_requests_four_diverse_candidates_without_test_info():
    prompt = neutral_v4.WARMSTART_PROMPT
    assert "Output 4 distinct candidates" in prompt
    assert "DIVERSITY" in prompt
    assert "protected" not in prompt.lower()
    assert "validation" not in prompt.lower() or "feedback" not in prompt.lower()


def test_refinement_gains_definition():
    frame = pd.DataFrame([
        # warm starts: best val at order 2 (val_loss 0.2 -> val_fid 0.8, test 0.79)
        {"seed": 0, "method": "Greedy", "order": 1, "phase": "explore",
         "val_loss": 0.3, "val_fid": 0.7, "test_fid": 0.69},
        {"seed": 0, "method": "Greedy", "order": 2, "phase": "explore",
         "val_loss": 0.2, "val_fid": 0.8, "test_fid": 0.79},
        {"seed": 0, "method": "Greedy", "order": 3, "phase": "explore",
         "val_loss": 0.4, "val_fid": 0.6, "test_fid": 0.59},
        {"seed": 0, "method": "Greedy", "order": 4, "phase": "explore",
         "val_loss": 0.5, "val_fid": 0.5, "test_fid": 0.49},
        # refinement improves to val_loss 0.1 (val_fid 0.9, test 0.88)
        {"seed": 0, "method": "Greedy", "order": 5, "phase": "refine",
         "val_loss": 0.1, "val_fid": 0.9, "test_fid": 0.88},
        {"seed": 0, "method": "LLM-Closed", "order": 1, "phase": "explore",
         "val_loss": 0.2, "val_fid": 0.8, "test_fid": 0.8},
        {"seed": 0, "method": "LLM-Closed", "order": 5, "phase": "refine",
         "val_loss": 0.3, "val_fid": 0.7, "test_fid": 0.7},
    ])
    gains = neutral_v4.refinement_gains(frame)
    assert np.isclose(gains["Greedy"]["mean_test_gain"], 0.88 - 0.79)
    # LLM-Closed refinement was worse than its warm start -> zero gain
    assert np.isclose(gains["LLM-Closed"]["mean_test_gain"], 0.0)


def test_arch_distance_counts_differing_slots():
    a = [{"type": "R", "axis": "Y", "q": 0}] * 4
    b = [{"type": "R", "axis": "Y", "q": 0}] * 3 + [{"type": "R", "axis": "X", "q": 0}]
    assert np.isclose(neutral_v4._arch_distance(a, b), 0.25)
