import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from config import Config, config

class JetDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, pt: np.ndarray, m0: np.ndarray):
        self.X  = X.astype(np.float32, copy=False)    # (N,H,W,C)
        self.y  = y.astype(np.int64,   copy=False)    # (N,)
        self.pt = pt.astype(np.float32, copy=False)
        self.m0 = m0.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], int(self.y[idx])

def _describe_dataset(name: str, ds: JetDataset):
    N = len(ds)
    yc = np.bincount(ds.y, minlength=2)
    print(f"[{name}] N={N}  y-counts={yc.tolist()}  frac(class1)={yc[1]/max(1,yc.sum()):.3f}")
    for feat in ("pt","m0"):
        a = getattr(ds, feat)
        print(f"  {feat}: min={a.min():.3f} p10={np.percentile(a,10):.3f} "
              f"p50={np.percentile(a,50):.3f} p90={np.percentile(a,90):.3f} max={a.max():.3f}")
    Xs = ds.X[:5]
    print(f"  X sample: shape={Xs.shape}, dtype={Xs.dtype}, min={Xs.min():.3f}, max={Xs.max():.3f}")

def load_datasets(cfg: Config) -> Tuple[JetDataset, JetDataset]:
    def _load(path: str, take: int) -> JetDataset:
        with h5py.File(path, 'r') as f:
            X  = f['X_jets'][:take]
            y  = f['y'][:take]
            pt = f['pt'][:take]
            m0 = f['m0'][:take]
        return JetDataset(X, y, pt, m0)

    train = _load(cfg.TRAIN_PATH, cfg.SAMPLES)
    test  = _load(cfg.TEST_PATH, cfg.SAMPLES)
    _describe_dataset("TRAIN(raw)", train)
    _describe_dataset("TEST(raw)",  test)
    return train, test

# Train-only per-channel normalization (applied to both splits)
train_dataset, test_dataset = load_datasets(config)
GLOBAL_NORM = {
    "mean": train_dataset.X.mean(axis=(0,1,2), keepdims=True),
    "std":  train_dataset.X.std(axis=(0,1,2),  keepdims=True) + 1e-6
}


__all__ = [
    "JetDataset", "load_datasets", "train_dataset", "test_dataset", "GLOBAL_NORM"
]