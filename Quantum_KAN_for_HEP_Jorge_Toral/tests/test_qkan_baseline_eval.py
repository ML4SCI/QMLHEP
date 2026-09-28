# Covers src/architectures/quantum_kan.py::evaluate / evaluate_baseline
"""
Unit tests for the baseline (pre-training) VQC evaluation.

Builds a QuantumKANTrainer without going through __init__ (which would build a
real PennyLane circuit via QKANModel) -- instead attaches a lightweight fake
`model` exposing only what evaluate()/evaluate_baseline() touch
(backend_mode, dev, qnode, _initialize_device, __call__, eval), and monkeypatches
pennylane.QNode so the backend-restore path doesn't need a real quantum device.
This keeps the test fast (no circuit simulation) while covering the two things
that matter: (a) baseline=True routes to *_baseline_{backend} config keys
instead of the plain ones, and (b) evaluate_baseline() restores
model.backend_mode back to train_backend afterwards.
"""
import types

import pennylane as qml
import torch

from src.architectures.quantum_kan import QuantumKANTrainer


class _FakeModel:
    """Always predicts a constant logit; backend_mode/dev/qnode mimic QKANModel."""

    def __init__(self, backend_mode):
        self.backend_mode = backend_mode
        self.dev = "fake-device"
        self.qnode = "fake-qnode"
        self._circuit = lambda x: x  # referenced (never actually called) by the stubbed qml.QNode

    def _initialize_device(self):
        return "fake-device"

    def eval(self):
        pass

    def __call__(self, x):
        return torch.zeros(x.shape[0])


def _make_trainer(tmp_path, monkeypatch, train_backend="ideal"):
    # evaluate()'s backend-switch path does `qml.QNode(...)` against a real
    # device when eval_backend != model.backend_mode; stub it out so switching
    # to "noisy" in these tests doesn't require a real Aer/noise-model device.
    monkeypatch.setattr(qml, "QNode", lambda *a, **kw: "fake-qnode")

    trainer = QuantumKANTrainer.__new__(QuantumKANTrainer)
    trainer.train_backend = train_backend
    trainer.criterion = torch.nn.BCEWithLogitsLoss()
    trainer.model = _FakeModel(backend_mode=train_backend)

    config = {"qkan_batch_size": 4}
    for suffix in ("", "_baseline", "_random"):
        for backend in ("ideal", "noisy", "shots"):
            for metric, ext in (("roc", "png"), ("cm", "png"), ("pr", "png"), ("metrics", "json")):
                key = f"{metric}_qkan{suffix}_{backend}"
                path = tmp_path / f"{key}.{ext}"
                path.parent.mkdir(parents=True, exist_ok=True)
                config[key] = str(path)
            cm_normalized_key = f"cm_qkan{suffix}_{backend}_normalized"
            path = tmp_path / f"{cm_normalized_key}.png"
            config[cm_normalized_key] = str(path)
            for eval_data_kind in ("true", "probs", "binary"):
                key = f"qkan_eval_data_{eval_data_kind}{suffix}_{backend}"
                config[key] = str(tmp_path / f"{key}.npy")
    trainer.config = config
    return trainer


def _toy_test_set():
    X_test = torch.zeros(4, 3)
    y_test = torch.tensor([0., 1., 0., 1.])
    return X_test, y_test


def test_evaluate_baseline_true_routes_to_baseline_keys(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    X_test, y_test = _toy_test_set()

    trainer.evaluate(X_test, y_test, eval_backend="ideal", baseline=True)

    assert (tmp_path / "metrics_qkan_baseline_ideal.json").exists()
    assert not (tmp_path / "metrics_qkan_ideal.json").exists()


def test_evaluate_baseline_false_routes_to_plain_keys(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    X_test, y_test = _toy_test_set()

    trainer.evaluate(X_test, y_test, eval_backend="ideal", baseline=False)

    assert (tmp_path / "metrics_qkan_ideal.json").exists()
    assert not (tmp_path / "metrics_qkan_baseline_ideal.json").exists()


def test_evaluate_baseline_restores_train_backend(tmp_path, monkeypatch):
    # train_backend="ideal", eval_backend="noisy" -> evaluate() switches
    # model.backend_mode to "noisy" internally; evaluate_baseline() must
    # restore it to "ideal" before returning, so a subsequent fit() call
    # still trains on the intended backend.
    trainer = _make_trainer(tmp_path, monkeypatch, train_backend="ideal")
    X_test, y_test = _toy_test_set()

    trainer.evaluate_baseline(X_test, y_test, eval_backend="noisy")

    assert trainer.model.backend_mode == "ideal"


def test_evaluate_random_init_routes_to_random_keys_and_saves_eval_arrays(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    X_test, y_test = _toy_test_set()

    trainer.evaluate(X_test, y_test, eval_backend="ideal", random_init=True)

    assert (tmp_path / "metrics_qkan_random_ideal.json").exists()
    assert (tmp_path / "qkan_eval_data_probs_random_ideal.npy").exists()
    assert not (tmp_path / "metrics_qkan_ideal.json").exists()
