# Guards ClassicKANTrainer's device propagation (src/architectures/classic_kan.py).
#
# This exercises the real code path (no mocking of torch/device) with a tiny
# synthetic model/dataset so it runs fast on both a CPU-only machine and a
# CUDA-enabled one: the expected device is always derived from
# torch.cuda.is_available(), so the assertions hold either way.
import torch

from src.architectures.classic_kan import ClassicKANTrainer


def _tiny_config():
    return {
        "width": [4, 4, 1],
        "grid": 3,
        "k": 3,
    }


def _expected_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_trainer_init_selects_and_applies_expected_device():
    trainer = ClassicKANTrainer(_tiny_config())

    expected = _expected_device()
    assert trainer.device == expected
    assert next(trainer.model.parameters()).device.type == expected.type


def test_train_kan_model_keeps_model_on_trainer_device(tmp_path):
    config = _tiny_config()
    trainer = ClassicKANTrainer(config)

    torch.manual_seed(0)
    X_train = torch.rand(32, 4)
    y_train = torch.randint(0, 2, (32, 1)).float()
    X_val = torch.rand(16, 4)
    y_val = torch.randint(0, 2, (16, 1)).float()

    trainer.train_kan_model(
        width=config["width"],
        grid=config["grid"],
        k=config["k"],
        learning_rate=1e-3,
        num_epochs=2,
        batch_size=8,
        early_stop_patience=2,
        early_stop_min_delta=1e-6,
        model_save_path=str(tmp_path / "model"),
        X_train_tensor=X_train,
        y_train_tensor=y_train,
        X_val_tensor=X_val,
        y_val_tensor=y_val,
        num_workers=0,
    )

    expected = _expected_device()
    assert next(trainer.model.parameters()).device.type == expected.type
