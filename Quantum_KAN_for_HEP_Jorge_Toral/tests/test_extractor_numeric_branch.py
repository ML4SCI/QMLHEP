# Covers src/architectures/extractor.py::_evaluate_isolated_edges
"""
Unit tests for extraction reading the numeric spline branch (act_fun)
instead of the symbolic branch (symbolic_fun), and for the subnode_bias/
subnode_scale dimensional handling.

Uses a real, tiny HEPKAN model (width=[3,[1,1],1]) instead of a full trained
checkpoint, so the test runs in well under a second. This width has a
surviving multiplication node, so width_out[1] (=3: 1 sum + 2*1 mult) !=
width_in[1] (=2: 1 sum + 1 mult) at layer 0 -- exactly the shape mismatch that
the old node_bias/node_scale code would crash on, and that subnode_bias/
subnode_scale (both sized width_out[1]) handles correctly.
"""
import numpy as np
import torch

from src.architectures.hep_kan import HEPKAN
from src.architectures.extractor import SymbolicWarmStartExtractor


def _make_tiny_model():
    torch.manual_seed(0)
    model = HEPKAN([3, [1, 1], 1], grid=3, k=3, symbolic_enabled=True, auto_save=False)
    model.eval()
    return model


def test_evaluate_isolated_edges_reads_numeric_branch_without_shape_error():
    model = _make_tiny_model()
    extractor = SymbolicWarmStartExtractor({"chebyshev_r2_threshold": 0.95, "chebyshev_max_degree": 4, "qkan_dynamic_range_threshold": 1e-3})
    # Force CPU regardless of what CUDA devices this machine reports -- the
    # extractor auto-picks cuda whenever torch.cuda.is_available() is True,
    # even on a GPU whose compute capability this torch build doesn't support.
    # The model built above lives on CPU (never moved), so the extractor must
    # match it for this test to be about the numeric-branch fix, not the GPU.
    extractor.device = torch.device("cpu")
    x_vals = np.linspace(-1, 1, 20)

    # layer_index=0 has a surviving mult node (width_out[1]=3 != width_in[1]=2),
    # which is exactly what the old node_bias/node_scale code shape-mismatched
    # on. This call must not raise.
    y_vals = extractor._evaluate_isolated_edges(
        model, layer_index=0, input_index=0, output_index=0, x_vals=x_vals
    )

    assert y_vals.shape == (20,)
    # symbolic_fun is still pykan's all-zero placeholder on a freshly
    # constructed (never fix_symbolic'd) model. If this were still reading the
    # symbolic branch, y_vals would be identically zero -- so a nonzero
    # dynamic range proves the numeric (act_fun) branch is what's being read.
    assert float(np.max(y_vals) - np.min(y_vals)) > 0.0


def test_symbolic_branch_is_still_the_zero_placeholder_on_a_fresh_model():
    # Sanity check for the test above's premise: confirms symbolic_fun really
    # is inert at this stage, so the nonzero-dynamic-range assertion actually
    # discriminates between the two branches instead of passing by accident.
    model = _make_tiny_model()
    assert torch.all(model.symbolic_fun[0].mask == 0)
