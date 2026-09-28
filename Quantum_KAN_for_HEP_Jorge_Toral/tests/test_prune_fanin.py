# Covers src/architectures/hep_kan.py::prune_fanin
"""
Unit tests for the additive max-fan-in cap applied during pruning.

Uses a lightweight fake model (plain objects exposing only the attributes
prune_fanin() actually touches: act_fun[i].mask.data, edge_scores, log_history)
instead of a real HEPKAN/pykan model, so the test is deterministic and fast --
edge_scores are hand-picked so the "greatest attribution contribution" edges
are known in advance, rather than depending on pykan's real attribution math.
"""
import types

import torch

from src.architectures.hep_kan import prune_fanin


def _make_fake_model(mask_layer0, scores_layer0):
    """Builds a minimal duck-typed model exposing what prune_fanin needs."""
    layer0 = types.SimpleNamespace(mask=types.SimpleNamespace(data=mask_layer0))
    layer1_mask = torch.ones(2, 2)
    layer1 = types.SimpleNamespace(mask=types.SimpleNamespace(data=layer1_mask))

    calls = []
    return types.SimpleNamespace(
        act_fun=[layer0, layer1],
        edge_scores=[scores_layer0, None],
        log_history=lambda name: calls.append(name),
        _log_history_calls=calls,
    )


def test_prune_fanin_keeps_top_k_by_attribution_score():
    # in_dim=4 (inputs), out_dim=3 (raw hidden neurons).
    # col 0: 4 active edges (over the cap -> must be trimmed to top-2 by score)
    # col 1: 2 active edges (already at the cap -> must be left untouched)
    # col 2: 1 active edge  (below the cap -> must be left untouched)
    mask = torch.tensor([
        [1., 1., 1.],
        [1., 0., 0.],
        [1., 1., 0.],
        [1., 0., 0.],
    ])
    # edge_scores shape is [out_dim, in_dim] (pykan convention, pre-permute).
    scores = torch.tensor([
        [0.1, 0.9, 0.3, 0.7],  # col 0: highest scores are input 1 (0.9) and input 3 (0.7)
        [0.2, 0.5, 0.4, 0.1],  # col 1: irrelevant, already within cap
        [0.9, 0.1, 0.1, 0.1],  # col 2: irrelevant, already within cap
    ])
    model = _make_fake_model(mask, scores)

    result = prune_fanin(model, max_inputs=2, layer_index=0)

    new_mask = result.act_fun[0].mask.data
    # Column 0: only inputs 1 and 3 (the 2 highest-scored) should survive --
    # not input 0/2 (which would be a "first-by-index" or arbitrary selection).
    assert new_mask[:, 0].tolist() == [0., 1., 0., 1.]
    # Columns already at/below the cap are untouched.
    assert new_mask[:, 1].tolist() == [1., 0., 1., 0.]
    assert new_mask[:, 2].tolist() == [1., 0., 0., 0.]


def test_prune_fanin_does_not_touch_other_layers():
    mask = torch.tensor([[1., 1.], [1., 1.], [1., 1.]])  # 3 active edges on col 0 and 1
    scores = torch.tensor([[0.5, 0.1, 0.9], [0.2, 0.8, 0.3]])
    model = _make_fake_model(mask, scores)

    original_layer1_mask = model.act_fun[1].mask.data.clone()
    result = prune_fanin(model, max_inputs=2, layer_index=0)

    assert torch.equal(result.act_fun[1].mask.data, original_layer1_mask)
    assert result._log_history_calls == ["prune_fanin"]
