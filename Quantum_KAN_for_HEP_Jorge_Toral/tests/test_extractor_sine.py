# Covers src/architectures/sine_basis.py and the sine branch of QKANModel
import numpy as np
import torch

from src.architectures.sine_basis import build_sine_grid, sinefit
from src.architectures.qkan_model import QKANModel


def test_sine_grid_shapes_and_first_layer_freq():
    freq, phase = build_sine_grid(4, is_first=True)
    assert freq.shape == (4,) and phase.shape == (4,)
    np.testing.assert_allclose(freq, [1, 2, 3, 4])


def test_sinefit_recovers_sine_edge():
    freq, phase = build_sine_grid(4)
    x = np.linspace(-1, 1, 200)
    amps_true = np.array([0.5, -0.3, 0.2, 0.1])
    y = np.sin(np.outer(x, freq) + phase[None, :]) @ amps_true
    amps, y_hat = sinefit(x, y, 4)
    np.testing.assert_allclose(amps, amps_true, atol=1e-6)
    assert np.max(np.abs(y - y_hat)) < 1e-8


def _tiny_graph(basis, degree=4):
    coefs = [0.3, -0.2, 0.1, 0.05, 0.0]  # degree+1
    return {
        "n_qubits": 1, "active_inputs": [0], "degree": degree, "basis": basis,
        "hidden_nodes": [{"type": "sum", "edge_groups": [[{"input_idx": 0, "col": 0, "wire": 0, "coefs": coefs}]]}],
        "output_edges": [{"hidden_idx": 0, "coefs": coefs}],
    }


def test_qkan_sine_forward_pass(tmp_path):
    path = tmp_path / "g.pt"
    torch.save(_tiny_graph("sine"), path)
    model = QKANModel(str(path), backend_mode="ideal")
    x = torch.rand(6, 1) * 2 - 1
    out = model(x)
    assert out.shape == (6,)
    assert torch.all(out.abs() <= 1.0 + 1e-3)
    assert model.basis == "sine"


def test_qkan_default_basis_is_chebyshev(tmp_path):
    g = _tiny_graph("chebyshev")
    del g["basis"]  # legacy graphs have no basis key
    path = tmp_path / "g.pt"
    torch.save(g, path)
    assert QKANModel(str(path), backend_mode="ideal").basis == "chebyshev"
