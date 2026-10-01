"""
Correlation calibration.

A rank-preserving linear map that nudges the generated correlation matrix
toward the target without touching the marginals. Fitted on the training set
and applied at generation time.

Only used by the calibrated variant. The raw pipeline never calls it.
"""
import numpy as np


def corr_only_R(Y_ref, Y_tr, blocks):
    """Per-block correlation-only calibration matrices (identity on
    singletons). Preserves the model-reference variances."""
    Rs = []
    for (s, b) in blocks:
        if b == 1:
            Rs.append(np.eye(1))
            continue
        C_ref = np.cov(Y_ref[:, s:s + b], rowvar=False) + EPS_REG * np.eye(b)
        d_ref = np.sqrt(np.diag(C_ref))
        Corr_tr = corr_nan_safe(Y_tr[:, s:s + b])
        C_t = np.outer(d_ref, d_ref) * Corr_tr + EPS_REG * np.eye(b)
        Rs.append(np.linalg.cholesky(C_t)
                  @ np.linalg.inv(np.linalg.cholesky(C_ref)))
    return Rs


def apply_R(Y, Rs, blocks):
    out = Y.copy()
    for R, (s, b) in zip(Rs, blocks):
        if b > 1:
            out[:, s:s + b] = Y[:, s:s + b] @ R.T
    return out
