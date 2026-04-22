"""
Multi-kernel RBF MMD^2 with analytic gradient w.r.t. predicted samples.



Jamal Slim
jamal.slim@desy.de



The bandwidth family is fixed once at initialization (computed on a held-out
reference batch via the median-distance heuristic), giving a STATIONARY
loss function whose descent is observable step-to-step. The median-heuristic-
per-minibatch approach used in the original QFAN code recomputes sigma every
batch and so produces a moving target; see train.py header for discussion.

Theoretical bound: MMD^2_sigma(P,Q) <= 2 for any RBF kernel with k(x,x) = 1,
so sum_{sigma in Sigma} MMD^2_sigma <= 2 |Sigma|, and total summed-over-blocks
val_loss upper bound is B * 2 * |Sigma|.
"""

import numpy as np


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    return np.maximum(0.0, AA + BB - 2.0 * (A @ B.T))


def median_sigma(X: np.ndarray, Y: np.ndarray, cap: int = 256, seed: int = 0) -> float:
    """
    Gretton-style median-of-pairwise-distances sigma on the union of X and Y.
    Called ONCE at init in this code base -- NOT re-called per minibatch.
    """
    Z = np.concatenate([X, Y], axis=0)
    m = Z.shape[0]
    if m > cap:
        idx = np.random.default_rng(seed).choice(m, size=cap, replace=False)
        Z = Z[idx]
    D2 = _sqdist(Z, Z)
    iu = np.triu_indices(D2.shape[0], k=1)
    vals = D2[iu]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(vals)) + 1e-12)


def mmd2_rbf(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """
    Biased V-statistic MMD^2 with RBF kernel k_sigma(x,y) = exp(-||x-y||^2/(2 sigma^2)).
    """
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    m, n = X.shape[0], Y.shape[0]
    if m == 0 or n == 0:
        return 0.0
    s2 = 2.0 * float(sigma) ** 2
    Kxx = np.exp(-_sqdist(X, X) / s2)
    Kyy = np.exp(-_sqdist(Y, Y) / s2)
    Kxy = np.exp(-_sqdist(X, Y) / s2)
    return float(Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2.0 * Kxy.sum() / (m * n))


def multi_kernel_mmd2(X: np.ndarray, Y: np.ndarray, sigmas) -> float:
    return float(sum(mmd2_rbf(X, Y, s) for s in sigmas))


def multi_kernel_mmd2_and_grad_Yhat(Y_hat: np.ndarray, Y_true: np.ndarray, sigmas):
    """
    Analytic gradient dL / dY_hat  for  L = sum_{sigma} MMD^2_sigma(Y_hat, Y_true).

    Derivation.  For k_s(x,y) = exp(-||x-y||^2 / (2 s^2)),
        d k_s(x,y) / dx = -(x-y)/s^2 * k_s(x,y).

    V-statistic gradient, with n = |Y_hat| and m = |Y_true|:

        dL / dY_hat_i  =  sum_s (2/(n^2 s^2)) [ sum_j K_hh_ij Y_hat_j
                                                - Y_hat_i sum_j K_hh_ij ]
                      +   sum_s (2/(n m s^2))  [ Y_hat_i sum_j K_hy_ij
                                                 - sum_j K_hy_ij Y_j ].

    Verified to ~1e-9 relative error against central finite differences on
    randomly sampled (X, Y, sigmas).

    Returns
    -------
    loss : float   scalar loss
    grad : (n, b)  dL / dY_hat in the same shape as Y_hat
    """
    Y_hat = np.asarray(Y_hat, np.float64)
    Y_true = np.asarray(Y_true, np.float64)
    n = Y_hat.shape[0]
    m = Y_true.shape[0]
    grad = np.zeros_like(Y_hat)
    loss = 0.0
    for sigma in sigmas:
        s2 = 2.0 * float(sigma) ** 2
        inv_s2 = 1.0 / (float(sigma) ** 2)
        Dhh = _sqdist(Y_hat, Y_hat)
        Dhy = _sqdist(Y_hat, Y_true)
        Dyy = _sqdist(Y_true, Y_true)
        Khh = np.exp(-Dhh / s2)
        Khy = np.exp(-Dhy / s2)
        Kyy = np.exp(-Dyy / s2)
        loss += (Khh.sum() / (n * n)
                 - 2.0 * Khy.sum() / (n * m)
                 + Kyy.sum() / (m * m))
        sum_hh = Khh.sum(axis=1, keepdims=True)
        sum_hy = Khy.sum(axis=1, keepdims=True)
        grad += (2.0 / (n * n)) * inv_s2 * (Khh @ Y_hat - Y_hat * sum_hh)
        grad += (2.0 / (n * m)) * inv_s2 * (Y_hat * sum_hy - Khy @ Y_true)
    return float(loss), grad
