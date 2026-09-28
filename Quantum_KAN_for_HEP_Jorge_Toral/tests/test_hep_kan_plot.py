# Covers HEPKAN.plot input labelling when len(in_vars) != width_in[0].
"""
Regression tests for the IndexError in HEPKAN.plot() when the model has more
inputs than names in `in_vars` (e.g. width[0] > 22 while the feature-name list
has 22 entries), plus correct labelling of pruned models.
"""
import types

import torch

from src.architectures.hep_kan import HEPKAN
from src.utils.hyperparams import get_hyperparams

FEATURES = get_hyperparams()["features"]


def _model_with_data(n_in, hidden=(2, 0)):
    model = HEPKAN(width=[n_in, list(hidden), 1], grid=3, k=3, seed=0, auto_save=False)
    x = torch.rand(64, n_in) * 2 - 1
    model(x)
    return model


def test_plot_with_more_inputs_than_names(tmp_path):
    model = _model_with_data(len(FEATURES) + 8)
    model.plot(
        folder=str(tmp_path / "edges"),
        save_path=str(tmp_path / "graph.png"),
        in_vars=FEATURES,
        varscale=0.5,
    )
    assert (tmp_path / "graph.png").exists()


def test_resolve_in_vars_pads_missing_names():
    model = types.SimpleNamespace()
    labels = HEPKAN._resolve_in_vars(model, ["a", "b"], 4)
    assert labels == ["a", "b", "x_3", "x_4"]


def test_resolve_in_vars_truncates_extra_names():
    model = types.SimpleNamespace()
    assert HEPKAN._resolve_in_vars(model, ["a", "b", "c"], 2) == ["a", "b"]


def test_resolve_in_vars_uses_surviving_ids_for_pruned_model():
    model = types.SimpleNamespace(input_id=torch.tensor([0, 5, 9]))
    assert HEPKAN._resolve_in_vars(model, FEATURES, 3) == [FEATURES[0], FEATURES[5], FEATURES[9]]


def test_resolve_in_vars_ignores_stale_input_id():
    # input_id points past the name list -> fall back to padding, no crash
    model = types.SimpleNamespace(input_id=torch.tensor([0, 40]))
    assert HEPKAN._resolve_in_vars(model, ["a", "b"], 3) == ["a", "b", "x_3"]


def test_config_width_matches_feature_names():
    cfg = get_hyperparams()
    assert cfg["width"][0] == len(cfg["features"])
