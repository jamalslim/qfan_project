#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run QFAN v3 on the paper's d=25 Appendix A extension.

Jamal Slim
jamal.slim@desy.de


    python scripts/run_d25.py

Configuration:
  - d=25 pixels, b=5 -> B=5 blocks
  - nq=3 qubits, L=3 layers -> p_theta = 18 shared variational parameters
  - pf = 12 Pauli features, G=2 measurement groups
  - EXACT parameter-shift training, Adam + cosine LR decay, PR averaging

Requires data/cal_shower_img_25q.npy (not included by default).

Writes:
    outputs/qfan_v3_d25_loss_curve.npz      training curves
    outputs/qfan_v3_d25_results.npz         metrics, generated samples, theta
"""

import sys
import pathlib

SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qfan import d25_config
from _pipeline import run_pipeline


def main():
    cfg = d25_config()
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
