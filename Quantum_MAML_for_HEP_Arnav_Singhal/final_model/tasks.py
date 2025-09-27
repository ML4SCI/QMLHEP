import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import config
from data import JetDataset, train_dataset, test_dataset, GLOBAL_NORM

# Normalization helper

def normalize_images(X: np.ndarray) -> np.ndarray:
    """Apply global channel-wise normalization (from train split)."""
    return (X - GLOBAL_NORM["mean"]) / GLOBAL_NORM["std"]

# PCA extractor + calibration

class PCAFeatureExtractor:
    """PCA fitted on normalized images; transform expects normalized inputs (same pipeline)."""
    def __init__(self, n_components: int):
        self.n = int(n_components)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=self.n, svd_solver="randomized",
                       whiten=False, random_state=42)
        self.fitted = False

    def fit_from_dataset(self, ds: JetDataset, max_samples: int = 15000, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        N = len(ds.y); k = min(max_samples, N)
        idx = rng.choice(np.arange(N), size=k, replace=False)
        X = ds.X[idx]                        # NHWC
        X = normalize_images(X)              # <-- IMPORTANT
        X = X.reshape(X.shape[0], -1)
        Xs = self.scaler.fit_transform(X)
        self.pca.fit(Xs)
        self.fitted = True
        return self

    def embed(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """Return PCA features as float32 torch tensor (B, n_components)."""
        assert self.fitted, "PCA not fitted; call fit_from_dataset first."
        if x_bchw.dim() == 4:  # (B,C,H,W)
            B = x_bchw.shape[0]
            x = (x_bchw.detach()
                 .permute(0, 2, 3, 1)
                 .contiguous()
                 .view(B, -1)
                 .cpu()
                 .numpy())
        elif x_bchw.dim() == 3:  # (C,H,W)
            x = (x_bchw.detach()
                 .permute(1, 2, 0)
                 .contiguous()
                 .view(1, -1)
                 .cpu()
                 .numpy())
        else:
            raise ValueError("Expected BCHW or CHW tensor.")
        xs = self.scaler.transform(x)
        z = self.pca.transform(xs)
        return torch.from_numpy(z.astype(np.float32))

# Fit PCA guard extractor
pca_guard_extractor = PCAFeatureExtractor(
    n_components=config.PCA_COMPONENTS
).fit_from_dataset(train_dataset, max_samples=30000)

@torch.no_grad()
def _proto_acc_pca_cos(Xs_n: np.ndarray, ys: np.ndarray,
                       Xq_n: np.ndarray, yq: np.ndarray) -> float:
    def to_t(x): return torch.from_numpy(x.transpose(0,3,1,2).copy()).float()
    Fs = pca_guard_extractor.embed(to_t(Xs_n))
    Fq = pca_guard_extractor.embed(to_t(Xq_n))
    mu_s = Fs.mean(0, keepdim=True); sd_s = Fs.std(0, keepdim=True).clamp_min(1e-6)
    Fs = (Fs - mu_s) / sd_s; Fq = (Fq - mu_s) / sd_s
    m0 = Fs[torch.from_numpy(ys)==0].mean(0, keepdim=True)
    m1 = Fs[torch.from_numpy(ys)==1].mean(0, keepdim=True)
    def _n(u): return u / u.norm(dim=1, keepdim=True).clamp_min(1e-6)
    d0 = 1 - (_n(Fq) @ _n(m0).T)
    d1 = 1 - (_n(Fq) @ _n(m1).T)
    pred = (d1 < d0).long().squeeze(1).cpu().numpy()
    return float((pred == yq).mean())

def _calibrate_pca_guard(ds: JetDataset, support_size=8, query_size=8,
                         trials=60, seed=7) -> float:
    rng = np.random.default_rng(seed)
    accs=[]
    for _ in range(trials):
        idx0 = np.where(ds.y==0)[0]; idx1 = np.where(ds.y==1)[0]
        s0 = rng.choice(idx0, support_size//2, replace=False)
        s1 = rng.choice(idx1, support_size//2, replace=False)
        q0 = rng.choice(np.setdiff1d(idx0, s0), query_size//2, replace=False)
        q1 = rng.choice(np.setdiff1d(idx1, s1), query_size//2, replace=False)
        s_idx = np.concatenate([s0,s1]); q_idx = np.concatenate([q0,q1])
        Xs_n = normalize_images(ds.X[s_idx]); Xq_n = normalize_images(ds.X[q_idx])
        ys = ds.y[s_idx]; yq = ds.y[q_idx]
        accs.append(_proto_acc_pca_cos(Xs_n, ys, Xq_n, yq))
    raw = float(np.quantile(accs, config.PCA_GUARD_PERCENTILE))
    thr = float(np.clip(raw, config.PCA_GUARD_CLAMP_MIN, config.PCA_GUARD_CLAMP_MAX))
    print(f"[PCA guard] raw q={config.PCA_GUARD_PERCENTILE:.0%}: {raw:.3f}  -> clamped: {thr:.3f} "
          f"(mean={np.mean(accs):.3f}, min={np.min(accs):.3f}, max={np.max(accs):.3f})")
    return thr

PCA_GUARD_THRESHOLD = _calibrate_pca_guard(
    train_dataset,
    support_size=config.SUPPORT_SIZE,
    query_size=config.QUERY_SIZE,
    trials=config.PCA_GUARD_CAL_TRIALS
)


# Helper utilities

def _gather_h5safe(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx).reshape(-1)
    order = np.argsort(idx)
    inv = np.empty_like(order); inv[order] = np.arange(order.size)
    return arr[idx[order]][inv]

def _hist_jsd(a: np.ndarray, b: np.ndarray, bins: int = 24) -> float:
    if a.size == 0 or b.size == 0: return np.nan
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if not np.isfinite([lo,hi]).all() or lo >= hi: return 0.0
    pa, edges = np.histogram(a, bins=bins, range=(lo,hi), density=True)
    pb, _     = np.histogram(b, bins=edges, density=True)
    pa = (pa + 1e-12); pb = (pb + 1e-12)
    pa /= pa.sum(); pb /= pb.sum()
    m = 0.5*(pa+pb)
    return 0.5*(np.sum(pa*np.log(pa/m)) + np.sum(pb*np.log(pb/m)))

def _quantile_edges(x: np.ndarray, bin_count: int) -> np.ndarray:
    qs = np.linspace(0, 1, bin_count+1)
    edges = np.quantile(x, qs).astype(np.float64)
    for i in range(1, edges.size):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)
    return edges

def _choose_query_joint_match(rem_idx: np.ndarray,
                              anchor_pt: np.ndarray, anchor_m0: np.ndarray,
                              k: int, pt_all: np.ndarray, m0_all: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    if rem_idx.size <= k: return rem_idx.copy()
    qs = np.linspace(0,1,k+2)[1:-1]
    tgt_pt = np.quantile(anchor_pt, qs); tgt_m0 = np.quantile(anchor_m0, qs)
    mu_pt, sd_pt = float(np.mean(anchor_pt)), float(np.std(anchor_pt)+1e-6)
    mu_m0, sd_m0 = float(np.mean(anchor_m0)), float(np.std(anchor_m0)+1e-6)
    R = rem_idx.copy()
    z_pt = (pt_all[R] - mu_pt)/sd_pt; z_m0 = (m0_all[R] - mu_m0)/sd_m0
    chosen=[]; used = np.zeros(R.size, dtype=bool)
    for tp, tm in zip(tgt_pt, tgt_m0):
        tpz = (tp - mu_pt)/sd_pt; tmz = (tm - mu_m0)/sd_m0
        d = np.abs(z_pt - tpz) + np.abs(z_m0 - tmz)
        d[used] = np.inf
        j = int(np.argmin(d)); used[j] = True
        chosen.append(R[j])
    if len(chosen) < k:
        d = np.abs(z_pt) + np.abs(z_m0); d[used] = np.inf
        order = np.argsort(d); need = k - len(chosen)
        chosen.extend(list(R[order[:need]]))
    return np.asarray(chosen[:k])

def _logreg_auc_guard(pt_s, m0_s, y_s, pt_q, m0_q, y_q,
                      lam=1e-2, iters=6) -> float:
    def _std2(a):
        mu, sd = a.mean(0, keepdim=True), a.std(0, keepdim=True)
        sd = np.clip(sd, 1e-6, None)
        return (a - mu) / sd, mu, sd
    Xs = np.stack([pt_s, m0_s], 1); Xq = np.stack([pt_q, m0_q], 1)
    Xs, mu, sd = _std2(Xs); Xq = (Xq - mu) / sd
    y = y_s.astype(np.float64).reshape(-1,1)
    X = np.concatenate([np.ones((Xs.shape[0],1)), Xs], 1)
    w = np.zeros((X.shape[1],1), dtype=np.float64)
    for _ in range(iters):
        z = X @ w; p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        g = X.T @ (p - y) + lam * np.r_[np.zeros((1,1)), w[1:]]
        S = (p*(1-p)).flatten(); H = X.T @ (X * S[:,None]); H[1:,1:] += lam * np.eye(H.shape[0]-1)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
    pq = 1.0 / (1.0 + np.exp(-np.clip(np.c_[np.ones((Xq.shape[0],1)), Xq] @ w, -20, 20)))
    auc = roc_auc_score(y_q, pq.ravel())
    return float(max(auc, 1.0-auc))  # orientation-agnostic

def _zscore_arr(a: np.ndarray) -> np.ndarray:
    mu, sd = np.mean(a), np.std(a)
    return (a - mu) / max(sd, 1e-6)

def _build_score(pt: np.ndarray, m0: np.ndarray) -> np.ndarray:
    zp, zm = _zscore_arr(pt), _zscore_arr(m0)
    return (1.0 * zm) + (-0.35 * zp)

# generate_meta_tasks  (with adaptive m0 cap + warm-up PCA guard)

def generate_meta_tasks(
    dataset: JetDataset,
    meta_task_types: List[str],
    bin_count: int,
    support_size: int,
    query_size: int,
    num_tasks_per_bin: int = 4,
    max_tasks: int = 128,
    seed: int = 42,
    target_jsd_max: float = 0.30,
    jsd_max_tries: int = 12,
) -> List[Dict[str, Any]]:
    """
      • Supports from class-conditional tails of score = z(m0) - 0.35*z(pt)
      • Weighted JSD with m0 emphasis (w_pt=0.3, w_m0=0.7)
      • Adaptive m0 cap: loose until ADAPTIVE_M0_MIN_ACCEPT tasks in a bin
      • Warm-up PCA guard: slightly lower threshold for first WARMUP_ACCEPT_N accepted tasks
    """
    rng = np.random.default_rng(seed)
    X, y = dataset.X, dataset.y
    pt = getattr(dataset, "pt"); m0 = getattr(dataset, "m0")
    assert support_size % 2 == 0 and query_size % 2 == 0
    half_s, half_q = support_size // 2, query_size // 2
    need_per_class = half_s + half_q

    score = _build_score(pt, m0)
    w_pt, w_m0 = 0.30, 0.70

    metas: List[Dict[str, Any]] = []
    total = 0
    accepted_total = 0
    accepted_count: Dict[Tuple[str,int], int] = {}

    def _edges_for(a: np.ndarray) -> np.ndarray:
        return _quantile_edges(a, bin_count)

    def _sample_support_from_tails(c_idx: np.ndarray, label: int) -> np.ndarray:
        if c_idx.size < half_s: return np.array([], dtype=int)
        svals = score[c_idx]; order = np.argsort(svals)
        tail_frac = 0.25; k_tail = max(int(np.ceil(tail_frac * c_idx.size)), half_s + 2)
        if label == 0: pool = c_idx[order][-k_tail:]
        else:          pool = c_idx[order][: k_tail]
        if pool.size < half_s: pool = c_idx
        return rng.choice(pool, size=half_s, replace=False)

    for feature_type in meta_task_types:
        active = getattr(dataset, feature_type)
        edges = _edges_for(active)

        for b in range(edges.size - 1):
            lo, hi = edges[b], edges[b + 1]
            mask = (active >= lo) & (active <= hi if b == edges.size - 2 else active < hi)
            idx = np.where(mask)[0]
            if idx.size < support_size + query_size: continue
            c0 = idx[y[idx] == 0]; c1 = idx[y[idx] == 1]
            if (c0.size < need_per_class) or (c1.size < need_per_class): continue

            max_here = min(num_tasks_per_bin,
                           int(len(c0) // (half_s + half_q)),
                           int(len(c1) // (half_s + half_q)))
            if max_here <= 0: continue

            for _ in range(max_here):
                if (c0.size < need_per_class) or (c1.size < need_per_class): break

                s0 = _sample_support_from_tails(c0, label=0)
                s1 = _sample_support_from_tails(c1, label=1)
                rem0 = np.setdiff1d(c0, s0, assume_unique=False)
                rem1 = np.setdiff1d(c1, s1, assume_unique=False)
                if (s0.size < half_s) or (s1.size < half_s) or (rem0.size < half_q) or (rem1.size < half_q): break

                q0 = _choose_query_joint_match(rem0, pt[s0], m0[s0], half_q, pt, m0, rng)
                q1 = _choose_query_joint_match(rem1, pt[s1], m0[s1], half_q, pt, m0, rng)

                tries, accepted = 0, False
                while tries < jsd_max_tries:
                    tries += 1
                    s_idx = np.concatenate([s0, s1]); q_idx = np.concatenate([q0, q1])

                    jsd_pt = _hist_jsd(pt[s_idx], pt[q_idx], bins=24)
                    jsd_m0 = _hist_jsd(m0[s_idx], m0[q_idx], bins=24)
                    wjsd   = w_pt * jsd_pt + w_m0 * jsd_m0

                    # adaptive m0 cap (looser until a few accepted in this bin)
                    key = (feature_type, int(b))
                    cap_m0_eff = (config.ADAPTIVE_M0_LOOSE_CAP
                                  if accepted_count.get(key, 0) < config.ADAPTIVE_M0_MIN_ACCEPT
                                  else 0.35)
                    cap_pt_eff = config.TG_CAP_PT

                    if not (np.isfinite(wjsd) and wjsd <= target_jsd_max and
                            (not np.isfinite(jsd_pt) or jsd_pt <= cap_pt_eff) and
                            (not np.isfinite(jsd_m0) or jsd_m0 <= cap_m0_eff)):
                        # try re-matching queries then re-pick supports from tails
                        if tries % 2 == 1:
                            q0 = _choose_query_joint_match(rem0, pt[s0], m0[s0], half_q, pt, m0, rng)
                            q1 = _choose_query_joint_match(rem1, pt[s1], m0[s1], half_q, pt, m0, rng)
                        else:
                            s0 = _sample_support_from_tails(c0, 0); s1 = _sample_support_from_tails(c1, 1)
                            rem0 = np.setdiff1d(c0, s0, assume_unique=False)
                            rem1 = np.setdiff1d(c1, s1, assume_unique=False)
                            if (rem0.size < half_q) or (rem1.size < half_q): break
                            q0 = _choose_query_joint_match(rem0, pt[s0], m0[s0], half_q, pt, m0, rng)
                            q1 = _choose_query_joint_match(rem1, pt[s1], m0[s1], half_q, pt, m0, rng)
                        continue

                    auc = _logreg_auc_guard(pt[s0], m0[s0], y[s0], pt[q_idx], m0[q_idx], y[q_idx])
                    if auc < config.GUARD_LOGREG_AUC_MIN:
                        s0 = _sample_support_from_tails(c0, 0); s1 = _sample_support_from_tails(c1, 1)
                        rem0 = np.setdiff1d(c0, s0, assume_unique=False)
                        rem1 = np.setdiff1d(c1, s1, assume_unique=False)
                        if (rem0.size < half_q) or (rem1.size < half_q): break
                        q0 = _choose_query_joint_match(rem0, pt[s0], m0[s0], half_q, pt, m0, rng)
                        q1 = _choose_query_joint_match(rem1, pt[s1], m0[s1], half_q, pt, m0, rng)
                        continue

                    # PCA guard warm-up (normalize only for guard; store RAW for CNN)
                    Xs_raw = _gather_h5safe(X, s_idx)
                    Xq_raw = _gather_h5safe(X, q_idx)
                    Xs_n = normalize_images(Xs_raw)
                    Xq_n = normalize_images(Xq_raw)
                    acc_img = _proto_acc_pca_cos(Xs_n, y[s_idx], Xq_n, y[q_idx])
                    thr_base = PCA_GUARD_THRESHOLD
                    thr_warm = max(0.52, PCA_GUARD_THRESHOLD - config.WARMUP_PCA_DELTA)
                    thr_use  = thr_warm if accepted_total < config.WARMUP_ACCEPT_N else thr_base

                    if (acc_img >= thr_use) and (acc_img <= config.PCA_GUARD_CLAMP_MAX):
                        accepted = True
                        break
                    else:
                        if tries % 2 == 1:
                            q0 = _choose_query_joint_match(rem0, pt[s0], m0[s0], half_q, pt, m0, rng)
                            q1 = _choose_query_joint_match(rem1, pt[s1], m0[s1], half_q, pt, m0, rng)
                        else:
                            s0 = _sample_support_from_tails(c0, 0); s1 = _sample_support_from_tails(c1, 1)
                            rem0 = np.setdiff1d(c0, s0, assume_unique=False)
                            rem1 = np.setdiff1d(c1, s1, assume_unique=False)
                            if (rem0.size < half_q) or (rem1.size < half_q): break
                            q0 = _choose_query_joint_match(rem0, pt[s0], m0[s0], half_q, pt, m0, rng)
                            q1 = _choose_query_joint_match(rem1, pt[s1], m0[s1], half_q, pt, m0, rng)

                if not accepted:
                    continue

                sX = torch.from_numpy(Xs_raw.transpose(0,3,1,2).copy()).float().contiguous()
                qX = torch.from_numpy(Xq_raw.transpose(0,3,1,2).copy()).float().contiguous()
                sy = torch.from_numpy(y[s_idx]).long()
                qy = torch.from_numpy(y[q_idx]).long()

                metas.append({
                    "support_X": sX, "support_y": sy,
                    "query_X":   qX, "query_y":   qy,
                    "support_idx": s_idx, "query_idx": q_idx,
                    "feature_tag": feature_type, "bin_id": int(b)
                })
                total += 1; accepted_total += 1
                accepted_count[key] = accepted_count.get(key, 0) + 1

                # light depletion to control reuse
                c0 = np.setdiff1d(c0, np.concatenate([s0, q0]), assume_unique=False)
                c1 = np.setdiff1d(c1, np.concatenate([s1, q1]), assume_unique=False)

                if total >= max_tasks:
                    print(f"Total meta-tasks generated: {total}")
                    print(f"Total meta-tasks actually sending: {len(metas)}")
                    return metas

    print(f"Total meta-tasks generated: {total}")
    print(f"Total meta-tasks actually sending: {len(metas)}")
    return metas

__all__ = [
    "normalize_images",
    "PCAFeatureExtractor",
    "pca_guard_extractor",
    "PCA_GUARD_THRESHOLD",
    "generate_meta_tasks",
]
