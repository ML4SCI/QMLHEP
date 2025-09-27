import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

from config import config
from data import train_dataset, test_dataset
from tasks import meta_tasks, test_meta_tasks

from models import CNNFeatureExtractor, PQCModel, HybridModel, freeze_bn
from meta import inner_loop_adaptation, outer_loop_meta_update, outer_loop_qmaml
from utils import plot_training_results, plot_comparison

# Helper: flatten PQC-only params

def _pqc_only_flat(pqc: nn.Module) -> torch.Tensor:
    """Flatten PQC weights + fc params (stable shape across inner loop)."""
    return torch.cat([
        pqc.weights.detach().flatten().cpu(),
        pqc.fc.weight.detach().flatten().cpu(),
        pqc.fc.bias.detach().flatten().cpu(),
    ])

# Warmup PQC head

def warmup_pqc_head(model: nn.Module, meta_tasks: List[Dict], steps: int = 100, lr: float = 1e-2):
    """Light warm-up of PQC head before full training."""
    opt = torch.optim.SGD(
        [{"params": [model.pqc.weights], "lr": lr},
         {"params": model.pqc.fc.parameters(), "lr": lr}],
        momentum=0.9
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for i in range(steps):
        task = meta_tasks[i % len(meta_tasks)]
        opt.zero_grad()
        logits = model(task["support_X"])
        loss = loss_fn(logits, task["support_y"])
        loss.backward()
        opt.step()

# Main Experiment Loop

def run_all_inits():
    initialization_types = ["gaussian", "uniform", "qmaml_learner", "zero", "pi"]
    results: Dict[str, Dict[str, List[float]]] = {}

    for init_type in initialization_types:
        print(f"\n=== Testing initialization: {init_type} ===")
        torch.manual_seed(42)
        np.random.seed(42)

        # --- Build CNN ---
        cnn_extractor = CNNFeatureExtractor(config.CNN_OUTPUT_DIM, config.NUM_QUBITS)
        freeze_bn(cnn_extractor)

        # --- Build PQC ---
        pqc_init = "zero" if init_type == "qmaml_learner" else init_type
        pqc_model = PQCModel(
            config.NUM_QUBITS,
            config.Q_DEPTH,
            init_type=pqc_init,
            bound_angles=True,
            verbose=True
        )
        hybrid_model = HybridModel(cnn_extractor, pqc_model)

        # --- Probes ---
        _sx = meta_tasks[0]["support_X"][:8]
        _sy = meta_tasks[0]["support_y"][:8]
        _qx = meta_tasks[0]["query_X"]

        # 7A) Gradient flow
        for p in hybrid_model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        _loss = nn.CrossEntropyLoss()(hybrid_model(_sx), _sy)
        _loss.backward()
        def _gnorm(p): return None if (p.grad is None) else float(p.grad.norm().item())
        try:
            _cnn_fc0 = cnn_extractor.backbone.fc[0].weight
            print("[Probe 7A]", init_type,
                  " grad||pqc.weights:", _gnorm(pqc_model.weights),
                  " grad||pqc.fc.weight:", _gnorm(pqc_model.fc.weight),
                  " grad||cnn.backbone.fc[0].weight:", _gnorm(_cnn_fc0))
        except Exception:
            print("[Probe 7A]", init_type,
                  " grad||pqc.weights:", _gnorm(pqc_model.weights),
                  " grad||pqc.fc.weight:", _gnorm(pqc_model.fc.weight))

        # 7B) Inner loop PQC-only test
        w_before = _pqc_only_flat(pqc_model)
        _ = inner_loop_adaptation(
            hybrid_model,
            meta_tasks[0]["support_X"], meta_tasks[0]["support_y"],
            inner_steps=2, inner_lr=0.01
        )
        w_after = _pqc_only_flat(pqc_model)
        print("[Probe 7B]", init_type, "||Δθ_PQC|| =", torch.norm(w_after - w_before).item())

        # 7C) CNN feature stats
        with torch.no_grad():
            _f = cnn_extractor(meta_tasks[0]["support_X"])
        print("[Probe 7C]", init_type, " feat mean:", _f.mean().item(), " feat std:", _f.std().item())

        # 7D) Prediction distribution
        with torch.no_grad():
            _p = torch.softmax(hybrid_model(_qx[:16]), dim=1)[:, 1]
        print("[Probe 7D]", init_type, " p1[min,mean,max]:",
              _p.min().item(), _p.mean().item(), _p.max().item())

        # --- Warm-up PQC head (all inits) ---
        warmup_pqc_head(hybrid_model, meta_tasks, steps=100, lr=1e-2)

        # --- Train ---
        if init_type == "qmaml_learner":
            training_results = outer_loop_qmaml(
                hybrid_model,
                meta_tasks,
                test_meta_tasks,
                config.OUTER_LR,
                config.EVAL_METRICS,
                ckpt_name="best_qmaml_learner.pth",
            )
        else:
            training_results = outer_loop_meta_update(
                hybrid_model,
                meta_tasks,
                test_meta_tasks,
                config.OUTER_LR,
                config.EVAL_METRICS,
                ckpt_name=f"best_{init_type}.pth",
            )

        results[init_type] = training_results
        plot_training_results(training_results, init_type)

    plot_comparison(results)


if __name__ == "__main__":
    run_all_inits()
