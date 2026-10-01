"""Closed-form ridge decoder (paper Sec. III-E, Eq. 5)."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def ridge_fit(F: np.ndarray, Y: np.ndarray, alpha: float):
    """
    W = argmin_W  ||F W - Y||_F^2 + alpha ||W||_F^2
      = (F^T F + alpha I)^{-1} F^T Y

    Returns
    -------
    W : (p_f, b)  ridge weights
    R : (n, b)    residuals Y - F W
    """
    F = np.asarray(F, np.float64); Y = np.asarray(Y, np.float64)
    _, p = F.shape
    # Cholesky rather than an explicit inverse: A is symmetric positive
    # definite for any alpha > 0, and this is inside the training loop.
    A = F.T @ F + float(alpha) * np.eye(p)
    B = F.T @ Y
    c, low = cho_factor(A, lower=True, check_finite=False)
    W = cho_solve((c, low), B, check_finite=False)
    resid = Y - F @ W
    return W, resid


def ridge_inverse_matrix(F: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return A = (F^T F + alpha I)^{-1} as a dense matrix. p_f is tiny
    (12, 30, 42, ...), so explicit inversion is fine.

    Used by the exact-gradient chain rule  dW/dtheta_k = A [G_k^T R - F^T G_k W].
    """
    F = np.asarray(F, np.float64)
    # Here we do want the dense inverse, because the gradient chain rule
    # needs A itself and not just A applied to one right-hand side. p_f is
    # 18 at n_q=3, so this costs nothing.
    p = F.shape[1]
    return np.linalg.inv(F.T @ F + float(alpha) * np.eye(p))
