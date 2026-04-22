"""
Residual sampler (paper Sec. III-F).


Jamal Slim
jamal.slim@desy.de


This is the CLASSICAL variant: K-means on whitened training residuals into
M = 2^r pools, plus a multinomial logistic gate that predicts pool
membership from [sketch, features, ridge_mean]. At generation time the
gate outputs P(pool | conditioning), one pool is sampled, and one residual
is drawn uniformly from that pool.

The paper's Sec. III-F explicitly names this construction as the drop-in
point for a parameterized quantum Born machine (the M = 2^r choice is
deliberate). Swap this class for BARG/QBG from v2.2 if you want a fully
quantum residual sampler; the training pipeline is unchanged because the
residual sampler is always fit POST-theta-training (paper Sec. III-F last
paragraph).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .utils import _power_of_two_floor, _whiten


class SafeResidualGate:
    def __init__(self, resid_q: int = 3, resid_max_clusters: int = 64,
                 n_init: int = 10,
                 whiten: bool = True, eps: float = 1e-6, shrink: float = 1e-2,
                 temp: float = 1.0, jitter: float = 0.0):
        self.resid_q = int(resid_q); self.resid_max_clusters = int(resid_max_clusters)
        self.n_init = int(n_init); self.whiten = bool(whiten)
        self.eps = float(eps); self.shrink = float(shrink)
        self.temp = float(temp); self.jitter = float(jitter)
        self.kmeans = None; self.pools = None; self.gate = None

    def _choose_M_power2(self, n_rows: int) -> int:
        target = min(2 ** self.resid_q, self.resid_max_clusters, n_rows)
        return int(max(2, _power_of_two_floor(target)))

    def fit(self, Xg: np.ndarray, R: np.ndarray) -> "SafeResidualGate":
        R = np.asarray(R, np.float64); Xg = np.asarray(Xg, np.float64)
        n = R.shape[0]; M = self._choose_M_power2(n)
        Rw = _whiten(R, eps=self.eps, shrink=self.shrink)[0] if self.whiten else R
        km = KMeans(n_clusters=M, n_init=self.n_init, random_state=0)
        labels = km.fit_predict(Rw).astype(np.int64)
        pools = [np.where(labels == m)[0] for m in range(M)]
        gate = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=500)
        gate.fit(Xg, labels)
        self.kmeans = km; self.pools = pools; self.gate = gate
        return self

    def sample(self, Xg: np.ndarray, R: np.ndarray, rng_np: np.random.Generator) -> np.ndarray:
        Xg = np.asarray(Xg, np.float64); R = np.asarray(R, np.float64)
        n = Xg.shape[0]
        if self.gate is None or self.pools is None:
            idx = rng_np.integers(0, R.shape[0], size=n)
            eps = R[idx]
            if self.jitter > 0:
                eps = eps + self.jitter * rng_np.normal(size=eps.shape)
            return eps
        P = self.gate.predict_proba(Xg); M = P.shape[1]
        eps = np.empty((n, R.shape[1]), dtype=np.float64)
        temp = max(1e-6, self.temp)
        for i in range(n):
            pi_raw = np.clip(P[i], 0.0, 1.0)
            if temp != 1.0:
                logp = np.log(pi_raw + 1e-12) / temp
                pi_ = np.exp(logp - logp.max())
            else:
                pi_ = pi_raw
            pi_ = pi_ / (pi_.sum() + 1e-12)
            m = int(rng_np.choice(M, p=pi_))
            pool = self.pools[m]
            j = int(pool[rng_np.integers(0, pool.size)]) if pool.size > 0 \
                else int(rng_np.integers(0, R.shape[0]))
            eps[i] = R[j]
        if self.jitter > 0:
            eps = eps + self.jitter * rng_np.normal(size=eps.shape)
        return eps


@dataclass
class BlockModel:
    """Per-block assembled model: ridge W, training residuals R, residual gate."""
    W: np.ndarray
    R: np.ndarray
    gate: SafeResidualGate
    start: int
    bsz: int
