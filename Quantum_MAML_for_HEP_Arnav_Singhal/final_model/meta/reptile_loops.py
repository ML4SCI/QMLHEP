# meta/reptile_loops.py
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any

from config import config
from models.hybrid import HybridModel
from models.cnn_extractor import freeze_bn, set_cnn_trainable

# Inner Loop (CNN frozen, adapt PQC only)

def inner_loop_adaptation(
    model: nn.Module,
    support_X: torch.Tensor,
    support_y: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
) -> (nn.Module, List[float]):
    """
    Reptile-style inner loop with CNN frozen:
      - Freeze all CNN params (BN kept in eval)
      - Adapt only PQC: {weights, fc}
      - Returns a wrapper that shares params with `model`
    """
    # 1) Freeze CNN completely
    for p in model.cnn.parameters():
        p.requires_grad = False
    try:
        freeze_bn(model.cnn)   # keep BN eval+frozen
    except NameError:
        pass
    model.cnn.eval()

    # 2) Ensure PQC trainables
    model.pqc.weights.requires_grad = True
    for p in model.pqc.fc.parameters():
        p.requires_grad = True

    # 3) Optimize only PQC
    adapted_model = HybridModel(model.cnn, model.pqc)  # shares params
    opt = torch.optim.SGD(
        [
            {"params": [model.pqc.weights], "lr": inner_lr},
            {"params": model.pqc.fc.parameters(), "lr": inner_lr},
        ],
        momentum=0.9, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    grad_norms: List[float] = []
    for _ in range(inner_steps):
        opt.zero_grad()
        logits = adapted_model(support_X)
        loss = loss_fn(logits, support_y)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_([model.pqc.weights], 1.0)
        torch.nn.utils.clip_grad_norm_(model.pqc.fc.parameters(), 1.0)

        # log grad norm
        g2 = 0.0
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g2 += float(p.grad.norm(2).item()) ** 2
        grad_norms.append(g2 ** 0.5)

        opt.step()

    return adapted_model, grad_norms


# Outer Loop (Reptile + thin-slice CNN update)

def outer_loop_meta_update(
    model: nn.Module,
    meta_tasks: List[Dict[str, Any]],
    test_meta_tasks: List[Dict[str, Any]],
    outer_lr: float,
    eval_metrics: bool,
    ckpt_name: str = "best_model.pth",
) -> Dict[str, List[float]]:
    """
    Reptile outer loop with frozen-CNN inner adaptation:
      - Inner loop updates only PQC (weights + fc)
      - Reptile interpolation on all params (CNN unchanged if frozen)
      - PLUS: a small-gradient outer update on CNN {layer4 convs + to_angles}
              using the query set (BN stays eval/frozen).
    """
    loss_fn = nn.CrossEntropyLoss()

    # Make only layer4 (convs) + to_angles trainable across tasks
    try:
        set_cnn_trainable(model.cnn, train_layer4=True, train_to_angles=True)
        freeze_bn(model.cnn)
    except NameError:
        for p in model.cnn.backbone.parameters():
            p.requires_grad = False
        for name, p in model.cnn.backbone.named_parameters():
            if "layer4" in name and "bn" not in name:
                p.requires_grad = True
        for p in model.cnn.to_angles.parameters():
            p.requires_grad = True
        try:
            freeze_bn(model.cnn)
        except Exception:
            pass
    model.cnn.eval()

    # Collect CNN fast params
    cnn_fast_params = []
    for n, p in model.cnn.named_parameters():
        if p.requires_grad and ("to_angles" in n or "layer4" in n):
            cnn_fast_params.append(p)

    # Optimizer for CNN slice
    outer_opt = torch.optim.Adam(
        [{"params": cnn_fast_params, "lr": outer_lr * 0.2}],
        betas=(0.9, 0.999), weight_decay=0.0
    )

    # Histories
    meta_loss_hist, train_loss_hist, val_loss_hist, grad_hist = [], [], [], []
    train_pvar_hist, val_pvar_hist = [], []
    metrics = {"train_accuracy": [], "val_accuracy": [], "precision": [], "recall": [], "f1_score": []}
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        meta_loss, tr_loss = 0.0, 0.0
        epoch_grad_norms, epoch_train_p1 = [], []
        train_accs: List[float] = []

        # ---------- Train loop ----------
        for task in meta_tasks:
            # snapshot θ0
            start_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # inner adapt (CNN frozen)
            adapted_model, grad_norms = inner_loop_adaptation(
                model, task["support_X"], task["support_y"], config.INNER_STEPS, config.INNER_LR
            )
            epoch_grad_norms.extend(grad_norms)

            # query eval
            with torch.no_grad():
                qlogits = adapted_model(task["query_X"])
                qloss = loss_fn(qlogits, task["query_y"])
                meta_loss += qloss.item()
                tr_loss += qloss.item()

                qprob = torch.softmax(qlogits, dim=1)[:, 1].cpu().numpy().tolist()
                epoch_train_p1.extend(qprob)
                if eval_metrics:
                    preds = torch.argmax(qlogits, dim=1)
                    train_accs.append((preds == task["query_y"]).float().mean().item())

            # Reptile interpolation
            with torch.no_grad():
                thetaT = model.state_dict()
                new_state = {k: start_state[k] + outer_lr * (thetaT[k] - start_state[k]) for k in thetaT.keys()}
                model.load_state_dict(new_state)

            # Thin-slice CNN update
            prev_flag = model.pqc.weights.requires_grad
            model.pqc.weights.requires_grad_(False)

            outer_opt.zero_grad()
            q_feats = model.cnn(task["query_X"])
            if not q_feats.requires_grad:
                q_feats.requires_grad_(True)
            qlogits_cnn = model.pqc(q_feats, weights_override=model.pqc.weights)
            qloss_cnn = loss_fn(qlogits_cnn, task["query_y"])
            qloss_cnn.backward()

            if cnn_fast_params:
                torch.nn.utils.clip_grad_norm_(cnn_fast_params, 1.0)
            outer_opt.step()

            model.pqc.weights.requires_grad_(prev_flag)

        # ---------- Validation ----------
        model.eval()
        vloss, val_accs, val_precs, val_recs, val_f1s, epoch_val_p1 = 0.0, [], [], [], [], []

        base_state = {k: v.clone() for k, v in model.state_dict().items()}

        for t in test_meta_tasks:
            adapted_model, _ = inner_loop_adaptation(
                model, t["support_X"], t["support_y"], config.INNER_STEPS, config.INNER_LR
            )
            with torch.no_grad():
                qlogits = adapted_model(t["query_X"])
                vloss += loss_fn(qlogits, t["query_y"]).item()
                epoch_val_p1.extend(torch.softmax(qlogits, dim=1)[:, 1].cpu().numpy().tolist())

                if eval_metrics:
                    vpreds = torch.argmax(qlogits, dim=1)
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    val_accs.append((vpreds == t["query_y"]).float().mean().item())
                    val_precs.append(precision_score(t["query_y"].cpu(), vpreds.cpu(), zero_division=0))
                    val_recs.append(recall_score(t["query_y"].cpu(), vpreds.cpu(), zero_division=0))
                    val_f1s.append(f1_score(t["query_y"].cpu(), vpreds.cpu(), zero_division=0))

        model.load_state_dict(base_state)

        # ---------- Logging ----------
        n_tasks = max(1, len(meta_tasks))
        meta_loss_hist.append(meta_loss / n_tasks)
        train_loss_hist.append(tr_loss / n_tasks)
        val_loss_hist.append(vloss / max(1, len(test_meta_tasks)))
        grad_hist.append(float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0)

        train_pvar_hist.append(float(np.var(epoch_train_p1)) if epoch_train_p1 else 0.0)
        val_pvar_hist.append(float(np.var(epoch_val_p1)) if epoch_val_p1 else 0.0)

        if eval_metrics:
            metrics["train_accuracy"].append(float(np.mean(train_accs)) if train_accs else 0.0)
            metrics["val_accuracy"].append(float(np.mean(val_accs)) if val_accs else 0.0)
            metrics["precision"].append(float(np.mean(val_precs)) if val_precs else 0.0)
            metrics["recall"].append(float(np.mean(val_recs)) if val_recs else 0.0)
            metrics["f1_score"].append(float(np.mean(val_f1s)) if val_f1s else 0.0)
        else:
            for k in metrics:
                metrics[k].append(0.0)

        avg_val_acc = metrics["val_accuracy"][-1]
        if config.SAVE_BEST_MODEL and avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, ckpt_name))

        print(f"[Reptile*] Epoch {epoch+1}/{config.EPOCHS} | "
              f"Meta-loss {meta_loss_hist[-1]:.4f} | "
              f"Val Loss {val_loss_hist[-1]:.4f} | "
              f"Train Acc {metrics['train_accuracy'][-1]:.4f} | "
              f"Val Acc {metrics['val_accuracy'][-1]:.4f} | "
              f"Train p-var {train_pvar_hist[-1]:.4f} | Val p-var {val_pvar_hist[-1]:.4f}")

    return {
        "meta_loss": meta_loss_hist,
        "training_loss": train_loss_hist,
        "validation_loss": val_loss_hist,
        "gradient_norms": grad_hist,
        "train_pvar": train_pvar_hist,
        "val_pvar": val_pvar_hist,
        **metrics,
        "test_metrics": {"accuracy": [], "precision": [], "recall": [], "f1_score": []},
    }


__all__ = [
    "inner_loop_adaptation",
    "outer_loop_meta_update",
]