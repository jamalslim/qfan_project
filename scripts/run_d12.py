#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run QFAN v3 on the paper's d=12 main-paper experiment (Sec. VII).

    python scripts/run_d12.py


Jamal Slim
jamal.slim@desy.de

Configuration:
  - d=12 pixels, b=6 -> B=2 blocks
  - nq=3 qubits, L=2 layers -> p_theta = 12 shared variational parameters
  - pf = 12 Pauli features, G=2 measurement groups
  - EXACT parameter-shift training, Adam + cosine LR decay, PR averaging
  - multi-kernel fixed-bandwidth MMD loss
  - rollout refinement (DAgger-lite) + Gaussian-copula calibration

Writes:
    outputs/qfan_v3_d12_loss_curve.npz      training curves
    outputs/qfan_v3_d12_results.npz         metrics, generated samples, theta
"""

import sys
import pathlib

SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qfan import d12_config
from _pipeline import run_pipeline


def main():
    cfg = d12_config()

    # To switch to the paper's SPSA baseline, uncomment:
    # cfg.train.mode = "spsa"

    # To enable Gauss-Newton preconditioning (free but sometimes slower at
    # small p_theta; use on ill-conditioned or high-shot-noise regimes):
    # cfg.train.use_precond = True

    data_dir = PROJECT_ROOT / "data"
    out_dir  = PROJECT_ROOT / "outputs"
    meta = run_pipeline(cfg, data_dir=data_dir, out_dir=out_dir, verbose=True)

    print("\n[final metrics]")
    for k in ("val_loss_first", "val_loss_last", "val_loss_min", "val_loss_pr",
              "w1_mean_raw", "w1_mean",
              "corr_mae_offdiag", "cal_corr_mae_offdiag",
              "corr_mae_cross",   "cal_corr_mae_cross",
              "mse_mean"):
        print(f"  {k:22s} : {meta[k]}")


if __name__ == "__main__":
    main()
