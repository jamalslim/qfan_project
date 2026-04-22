"""


Numerical utilities shared across the pipeline.
Jamal Slim
jamal.slim@desy.de

"""

import numpy as np


def corr_nan_safe(M: np.ndarray) -> np.ndarray:
    """np.corrcoef, zero-filled for columns with zero variance (dead pixels)."""
    with np.errstate(all="ignore"):
        C = np.corrcoef(M, rowvar=False)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _power_of_two_floor(n: int) -> int:
    if n < 1:
        return 0
    return 1 << (int(n).bit_length() - 1)


def _whiten(Z, eps: float = 1e-6, shrink: float = 0.0):
    """
    Return (Zw, mu, W) with Zw = (Z - mu) W, where W = C^{-1/2} for
    C = (Z-mu)^T(Z-mu)/n shrunk toward identity.
    """
    Z = np.asarray(Z, float)
    mu = Z.mean(axis=0, keepdims=True)
    X = Z - mu
    C = (X.T @ X) / max(1, X.shape[0])
    d = C.shape[0]
    if shrink > 0:
        C = (1.0 - shrink) * C + shrink * np.eye(d)
    w, V = np.linalg.eigh(C + eps * np.eye(d))
    w = np.clip(w, eps, None)
    Wm12 = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return X @ Wm12, mu, Wm12


def _psd_sqrt(C, eps: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(C, np.float64))
    w = np.clip(w, eps, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _psd_invsqrt(C, eps: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(C, np.float64))
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T
