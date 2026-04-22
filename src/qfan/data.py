"""


Dataset loading and subset selection.
Particularly useful when running on hardware


Jamal Slim
jamal.slim@desy.de

"""

import os
import numpy as np
from sklearn.cluster import KMeans


def _resolve_subset_n(N: int, subset_n=None, subset_frac=None) -> int:
    if subset_n is not None:
        return int(max(1, min(N, subset_n)))
    if subset_frac is not None:
        return int(max(1, min(N, int(np.floor(N * float(subset_frac))))))
    return N


def select_subset_indices(X, subset_n, mode: str = "random", seed: int = 123,
                          kmeans_max_dim: int = 256, kmeans_ninit: int = 5) -> np.ndarray:
    """Return np.int64 indices selecting `subset_n` rows of X by the chosen mode."""
    X = np.asarray(X)
    N = X.shape[0]
    subset_n = int(max(1, min(N, subset_n)))
    if subset_n >= N:
        return np.arange(N, dtype=np.int64)
    prng = np.random.default_rng(seed)

    if mode == "random":
        return prng.choice(N, size=subset_n, replace=False).astype(np.int64)

    if mode == "stride":
        step = max(1, int(np.floor(N / subset_n)))
        idx = np.arange(0, N, step, dtype=np.int64)
        if idx.size > subset_n:
            idx = idx[:subset_n]
        elif idx.size < subset_n:
            extra = prng.choice(N, size=(subset_n - idx.size), replace=False)
            idx = np.unique(np.concatenate([idx, extra]).astype(np.int64))
        return idx[:subset_n]

    if mode == "kmeans":
        d = X.shape[1]
        dd = min(d, int(kmeans_max_dim))
        Xr = X[:, :dd].astype(np.float64, copy=False)
        mu = Xr.mean(axis=0, keepdims=True)
        sig = Xr.std(axis=0, keepdims=True) + 1e-12
        Z = (Xr - mu) / sig
        km = KMeans(n_clusters=subset_n, n_init=int(kmeans_ninit), random_state=seed)
        labels = km.fit_predict(Z)
        C = km.cluster_centers_
        idx = []
        for k in range(subset_n):
            members = np.where(labels == k)[0]
            if members.size == 0:
                continue
            dif = Z[members] - C[k][None, :]
            idx.append(int(members[np.argmin(np.sum(dif * dif, axis=1))]))
        idx = np.unique(np.array(idx, dtype=np.int64))
        if idx.size < subset_n:
            remaining = np.setdiff1d(np.arange(N, dtype=np.int64), idx, assume_unique=False)
            add = prng.choice(remaining, size=(subset_n - idx.size), replace=False)
            idx = np.concatenate([idx, add]).astype(np.int64)
        return idx[:subset_n]

    raise ValueError(f"Unknown SUBSET_MODE={mode}")


def load_dataset(data_path: str, fallback_d: int = 25, fallback_N: int = 2500,
                 fallback_seed: int = 7):
    """
    Load a .npy calorimeter dataset from `data_path`. If missing, synthesize
    a stand-in (documented, not a substitute). Returns float64 array.
    """
    if os.path.exists(data_path):
        return np.load(data_path, allow_pickle=True).astype(np.float64)
    print(f"[WARN] data {data_path} not found; generating synthetic stand-in "
          f"(d={fallback_d}, N={fallback_N}).")
    rng = np.random.default_rng(fallback_seed)
    X = rng.exponential(2.0, size=(fallback_N, fallback_d))
    for i in range(fallback_d - 1):
        X[:, i + 1] += 0.20 * X[:, i]
    X += 0.02 * rng.normal(size=X.shape)
    return X.astype(np.float64)
