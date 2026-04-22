"""
Jamal Slim
jamal.slim@desy.de


Evaluation metrics: correlation-matrix error partitioned by block structure.


"""

import numpy as np


def _corr_masks(blocks, d: int):
    """Boolean masks for off-diagonal entries, partitioned into within-block
    vs cross-block. Shape: (d, d) booleans each."""
    blk = np.empty(d, dtype=np.int64)
    for bi, (s, b) in enumerate(blocks):
        blk[s:s + b] = bi
    ii, jj = np.indices((d, d))
    off = ii != jj
    within = (blk[ii] == blk[jj]) & off
    cross  = (blk[ii] != blk[jj]) & off
    return within, cross


def correlation_error_summary(C_true: np.ndarray, C_gen: np.ndarray, blocks) -> dict:
    """
    Partition |C_true - C_gen| into:
      - all off-diagonal (`corr_mae_offdiag`)
      - within-block (`corr_mae_within`)
      - cross-block  (`corr_mae_cross`)

    The cross-block value is the one that directly probes the sketch's
    ability to propagate information between autoregressive blocks
    (paper Sec. VII-B, "patterns in correlations").
    """
    d = C_true.shape[0]
    within, cross = _corr_masks(blocks, d)
    err = np.abs(C_true - C_gen)
    return {
        "corr_mae_all":     float(err.mean()),
        "corr_mae_offdiag": float(err[np.eye(d, dtype=bool) == 0].mean()),
        "corr_mae_within":  float(err[within].mean()) if np.any(within) else 0.0,
        "corr_mae_cross":   float(err[cross].mean())  if np.any(cross)  else 0.0,
    }
