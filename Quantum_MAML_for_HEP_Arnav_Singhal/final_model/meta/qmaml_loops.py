import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any

from config import config
from models.learner import Learner, compute_task_embedding
from models.cnn_extractor import set_cnn_trainable, freeze_bn

# Q-MAML Inner Loop

def inner_loop_adaptation_qmaml(
    model: nn.Module,
    start_weights: torch.Tensor,
    support_X: torch.Tensor,
    support_y: torch.Tensor,
    steps: int,
    lr: float,
) -> torch.Tensor:
    if config.FREEZE_CNN_DURING_META:
        for p in model.cnn.parameters():
            p.requires_grad = False
        with torch.no_grad():
            support_feats = model.cnn(support_X)
    else:
        support_feats = model.cnn(support_X)

    w = nn.Parameter(start_weights.clone().detach().to(torch.float64), requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = model.pqc(support_feats, weights_override=w)
        loss = loss_fn(logits, support_y)
        loss.backward()
        opt.step()

    return w.detach()


# Q-MAML Outer Loop

def outer_loop_qmaml(
    model: nn.Module,
    meta_tasks: List[Dict[str, Any]],
    test_meta_tasks: List[Dict[str, Any]],
    outer_lr: float,
    eval_metrics: bool,
    ckpt_name: str = "best_qmaml_learner.pth",
) -> Dict[str, List[float]]:

    loss_fn = nn.CrossEntropyLoss()

    # ----- Learner -----
    pqc_shape = tuple(model.pqc.weights.shape)
    learner = Learner(config.CNN_OUTPUT_DIM, pqc_shape, config.LEARNER_HIDDEN).double()
    learner_dtype = next(learner.parameters()).dtype

    # ----- Trainable CNN parts -----
    if config.FREEZE_CNN_DURING_META:
        try:
            set_cnn_trainable(model.cnn, train_layer4=True, train_to_angles=True)
            freeze_bn(model.cnn)
        except Exception:
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

    cnn_fast_params = [p for n, p in model.cnn.named_parameters()
                       if p.requires_grad and ("to_angles" in n or "layer4" in n)]

    # ----- Optimizer -----
    outer_opt = torch.optim.Adam(
        [
            {"params": learner.parameters(),      "lr": outer_lr},
            {"params": model.pqc.fc.parameters(), "lr": outer_lr},
            {"params": cnn_fast_params,           "lr": outer_lr * 0.2},
        ]
    )

    # ----- Histories -----
    meta_loss_hist, train_loss_hist, val_loss_hist, grad_hist = [], [], [], []
    train_pvar_hist, val_pvar_hist = [], []
    metrics = {"train_accuracy": [], "val_accuracy": [], "precision": [], "recall": [], "f1_score": []}
    best_acc = 0.0

    def _grad_norm_preclip(*param_iters) -> float:
        s = 0.0
        for it in param_iters:
            for p in it:
                g = p.grad
                if g is None:
                    continue
                s += float(g.detach().pow(2).sum().item())
        return s ** 0.5

    # ----- Training loop -----
    for epoch in range(config.EPOCHS):
        model.train()
        meta_loss_epoch, epoch_grad_norms = 0.0, []
        epoch_train_p1, train_accs = [], []

        for task in meta_tasks:
            outer_opt.zero_grad()

            with torch.no_grad():
                emb = compute_task_embedding(model.cnn, task["support_X"]).to(learner_dtype)
            w0 = config.W0_SCALE * learner(emb)

            wT = inner_loop_adaptation_qmaml(
                model, w0, task["support_X"], task["support_y"], config.INNER_STEPS, config.INNER_LR
            )
            w_query = (wT - w0).detach() + w0

            q_feats = model.cnn(task["query_X"])
            qlogits = model.pqc(q_feats, weights_override=w_query)
            qloss = loss_fn(qlogits, task["query_y"])
            qloss.backward()

            gn = _grad_norm_preclip(
                list(learner.parameters()),
                list(model.pqc.fc.parameters()),
                cnn_fast_params
            )
            epoch_grad_norms.append(gn)

            torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.pqc.fc.parameters(), 1.0)
            if cnn_fast_params:
                torch.nn.utils.clip_grad_norm_(cnn_fast_params, 1.0)

            outer_opt.step()
            meta_loss_epoch += qloss.item()

            with torch.no_grad():
                qprob = torch.softmax(qlogits, dim=1)[:, 1].cpu().numpy().tolist()
                epoch_train_p1.extend(qprob)
                if eval_metrics:
                    qpreds = torch.argmax(qlogits, dim=1)
                    train_accs.append((qpreds == task["query_y"]).float().mean().item())

        # ----- Validation -----
        model.eval()
        vloss, epoch_val_p1 = 0.0, []
        val_accs, val_precs, val_recs, val_f1s = [], [], [], []

        for t in test_meta_tasks:
            with torch.no_grad():
                emb = compute_task_embedding(model.cnn, t["support_X"]).to(learner_dtype)
                w0 = config.W0_SCALE * learner(emb)
            wT = inner_loop_adaptation_qmaml(
                model, w0, t["support_X"], t["support_y"], config.INNER_STEPS, config.INNER_LR
            )
            with torch.no_grad():
                q_feats = model.cnn(t["query_X"])
                logits = model.pqc(q_feats, weights_override=wT)
                vloss += loss_fn(logits, t["query_y"]).item()
                epoch_val_p1.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())

                if eval_metrics:
                    preds = torch.argmax(logits, dim=1)
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    val_accs.append((preds == t["query_y"]).float().mean().item())
                    val_precs.append(precision_score(t["query_y"].cpu(), preds.cpu(), zero_division=0))
                    val_recs.append(recall_score(t["query_y"].cpu(), preds.cpu(), zero_division=0))
                    val_f1s.append(f1_score(t["query_y"].cpu(), preds.cpu(), zero_division=0))

        # ----- Logging -----
        n_tasks = max(1, len(meta_tasks))
        meta_loss_hist.append(meta_loss_epoch / n_tasks)
        train_loss_hist.append(meta_loss_epoch / n_tasks)
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
            for k in metrics: metrics[k].append(0.0)

        avg_val_acc = metrics["val_accuracy"][-1]
        if config.SAVE_BEST_MODEL and avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(learner.state_dict(), os.path.join(config.CHECKPOINT_DIR, ckpt_name))

        print(f"[Q-MAML] Epoch {epoch+1}/{config.EPOCHS} | "
              f"Meta-loss {meta_loss_hist[-1]:.4f} | "
              f"Val Loss {val_loss_hist[-1]:.4f} | "
              f"Train Acc {metrics['train_accuracy'][-1]:.4f} | "
              f"Val Acc {metrics['val_accuracy'][-1]:.4f}")

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
    "inner_loop_adaptation_qmaml",
    "outer_loop_qmaml",
]
