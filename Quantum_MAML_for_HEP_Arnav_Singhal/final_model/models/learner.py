import torch
import torch.nn as nn
from typing import Tuple

def _prod(shape):
    p = 1
    for s in shape: 
        p *= int(s)
    return p

class Learner(nn.Module):
    """Classical meta-learner that predicts initial PQC angles for a task."""
    def __init__(self, in_dim: int, pqc_shape: Tuple[int,int,int], hidden: int):
        super().__init__()
        self.pqc_shape = pqc_shape
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, _prod(pqc_shape))
        )

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        out = self.net(task_emb)                     # (prod,)
        return out.view(*self.pqc_shape)             # (depth, nq, 3)

@torch.no_grad()
def compute_task_embedding(cnn: nn.Module, support_X: torch.Tensor) -> torch.Tensor:
    """Mean 512-D embedding over support set."""
    if hasattr(cnn, "embed"):
        feats = cnn.embed(support_X)         # (S, 512)
    else:
        feats = cnn(support_X)               # fallback
    return feats.mean(dim=0)                 # (512,)
