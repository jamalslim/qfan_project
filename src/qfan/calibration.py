"""
Empirical Gaussian-copula post-hoc calibration.

Jamal Slim
jamal.slim@desy.de

For each dimension j:  u_j = F^{-1}_gen,j(F_gen,j(Y_j))   (quantile standardization)
                        z_j = Phi^{-1}(u_j)                (to Gaussian scores)
                        z'  = A z,  A = Sigma_true^{1/2} Sigma_gen^{-1/2}
                        Y' = F^{-1}_true,j(Phi(z'_j))      (back to data support)

This computes the joint-correlation structure of generated samples while
preserving per-dimension marginals exactly. It is independent of quantum
training; paper Sec. III-F notes that independent post-hoc correction is
acceptable as long as it doesn't enter the SPSA/exact objective.
"""

import numpy as np
from scipy.stats import norm

from .utils import _psd_sqrt, _psd_invsqrt, corr_nan_safe


class EmpiricalGaussianCopulaCalibrator:
    def __init__(self, shrink: float = 0.05, eps: float = 1e-6,
                 clip_nonnegative: bool = True):
        self.shrink = float(shrink); self.eps = float(eps)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.sorted_gen = None; self.sorted_true = None
        self.A = None; self.fitted = False

    def _gaussianize(self, X, sorted_ref):
        X = np.asarray(X, np.float64); n, d = X.shape
        Z = np.zeros_like(X); m = sorted_ref.shape[0]; denom = float(m + 1)
        for j in range(d):
            pos = np.searchsorted(sorted_ref[:, j], X[:, j], side='right')
            u = (pos + 0.5) / denom
            u = np.clip(u, self.eps, 1.0 - self.eps)
            Z[:, j] = norm.ppf(u)
        return Z

    def _degaussianize(self, Z, sorted_tgt):
        Z = np.asarray(Z, np.float64); n, d = Z.shape
        X = np.zeros_like(Z); m = sorted_tgt.shape[0]
        grid = np.arange(m, dtype=np.float64)
        for j in range(d):
            u = np.clip(norm.cdf(Z[:, j]), self.eps, 1.0 - self.eps)
            q = u * (m - 1)
            X[:, j] = np.interp(q, grid, sorted_tgt[:, j])
        if self.clip_nonnegative:
            X = np.maximum(X, 0.0)
        return X

    def fit(self, Y_gen_ref: np.ndarray, Y_true_ref: np.ndarray) -> "EmpiricalGaussianCopulaCalibrator":
        Y_gen_ref = np.asarray(Y_gen_ref, np.float64)
        Y_true_ref = np.asarray(Y_true_ref, np.float64)
        self.sorted_gen = np.sort(Y_gen_ref, axis=0)
        self.sorted_true = np.sort(Y_true_ref, axis=0)
        Zg = self._gaussianize(Y_gen_ref, self.sorted_gen)
        Zt = self._gaussianize(Y_true_ref, self.sorted_true)
        Cg = corr_nan_safe(Zg); Ct = corr_nan_safe(Zt)
        d_ = Cg.shape[0]
        Cg = (1.0 - self.shrink) * Cg + self.shrink * np.eye(d_)
        Ct = (1.0 - self.shrink) * Ct + self.shrink * np.eye(d_)
        self.A = _psd_sqrt(Ct, eps=self.eps) @ _psd_invsqrt(Cg, eps=self.eps)
        self.fitted = True
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Copula not fitted.")
        Y = np.asarray(Y, np.float64)
        Z = self._gaussianize(Y, self.sorted_gen)
        Zc = Z @ self.A.T
        return self._degaussianize(Zc, self.sorted_true)
