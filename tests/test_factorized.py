#!/usr/bin/env python3
"""
Regression test: the factorized CF-MMD loss must reproduce the original
atom-table implementation to machine precision on every code path.

    python tests/test_factorized.py

Covers register sizes n_q = 2, 3, 4, block widths b = 1 and 2, record sharing
on and off, prefix conditioning on and off, and the risk term on and off. For
each configuration it compares the loss, the theta gradient, both encoding
gradients, the record-share gradient and the decoder weights.

The original enumerates 2^(G n_q) joint records and is only usable up to about
n_q = 5. The factorized version is what makes larger registers possible, and
this test is what licenses using it: if the two ever disagree, the scaled
results are no longer produced by the algorithm the paper describes.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qfan.born import (BornBankSpec, StatevectorBornBank,          # noqa: E402
                       cf_mmd_loss_and_grads)
from qfan.born_factorized import (                                  # noqa: E402
    cf_mmd_loss_and_grads_factorized)

TOL = 1e-9


def _abs_err(a, b):
    if a is None and b is None:
        return 0.0
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())


def main():
    names = ["loss", "grad_theta", "gA", "gb", "W", "grad_rho"]
    idx = [0, 1, 2, 3, 4, 5]
    worst = 0.0
    n_cases = 0
    for nq in (2, 3, 4):
        bank = StatevectorBornBank(
            16, BornBankSpec(n_qubits=nq, depth=2, angle_dim=8,
                             include_y=True), seed=5)
        for b in (1, 2):
            for split in ((False, True) if b == 2 else (False,)):
                for cond in (False, True):
                    for risk in (0.0, 5.0):
                        rng = np.random.default_rng(
                            nq * 100 + b * 10 + split * 3 + cond)
                        n = 24
                        S = rng.normal(size=(n, 16))
                        Y = 0.1 * rng.random((n, b)) + 0.05
                        th = 0.3 * rng.normal(size=bank.n_var)
                        Om = rng.normal(size=(40, b))
                        kw = dict(
                            k_shots=8, gamma_risk=risk, compute_grad=True,
                            compute_encoding_grad=True,
                            X_cond=rng.normal(size=(n, 3)) if cond else None,
                            Omega_cond=rng.normal(size=(40, 3)) if cond else None,
                            rho=0.4 if split else None)
                        a = cf_mmd_loss_and_grads(bank, S, Y, th, Om, 1e-3, **kw)
                        c = cf_mmd_loss_and_grads_factorized(
                            bank, S, Y, th, Om, 1e-3, **kw)
                        for nm, i in zip(names, idx):
                            e = _abs_err(c[i], a[i])
                            worst = max(worst, e)
                            if e > TOL:
                                print(f"  [FAIL] n_q={nq} b={b} split={split} "
                                      f"cond={cond} risk={risk}: {nm} "
                                      f"abs err {e:.2e}")
                                sys.exit(1)
                        n_cases += 1
    print(f"  [PASS] factorized == original on {n_cases} configurations, "
          f"worst absolute deviation {worst:.2e}")
    check_large_blocks()


def check_large_blocks():
    """Blocks of three or more pixels have no reference implementation (the
    original modelled record sharing only for b == 2), so they are checked
    two independent ways: the loss must describe what generation draws, and
    every gradient must match finite differences of the loss."""
    from qfan.born_factorized import _per_setting_factors
    bank = StatevectorBornBank(
        12, BornBankSpec(n_qubits=3, depth=2, angle_dim=8, include_y=True),
        seed=5)
    rng = np.random.default_rng(1)
    th = 0.4 * rng.normal(size=bank.n_var)
    bank.set_theta(th)

    # 1. the loss models generation: analytic CF vs records drawn the way the
    #    generator draws them, k_sh shared plus k_ex private per pixel
    P = bank.setting_probs(rng.normal(size=(1, 12)), th)
    k = 16
    for b, rho in ((3, 0.5), (5, 0.25)):
        W = 0.3 * rng.normal(size=(bank.pf, b))
        Om = rng.normal(size=(6, b))
        ksh = int(rho * k); kex = k - ksh
        Pa, _ = _per_setting_factors(bank, P, W, Om, float(k))
        logpsi = ksh * np.log(np.prod(Pa, 0))
        for j in range(b):
            m = np.zeros(b); m[j] = 1
            Pj, _ = _per_setting_factors(bank, P, W, Om * m, float(k))
            logpsi = logpsi + kex * np.log(np.prod(Pj, 0))
        psi = np.exp(logpsi)[0]
        R, chunk = 60000, 10000
        acc = np.zeros(len(Om), complex)
        for _ in range(R // chunk):
            rec = bank.sample_records(np.repeat(P, chunk, axis=1),
                                      ksh + b * kex, rng)
            f = np.concatenate([bank.T[rec[g]] for g in range(bank.G)], axis=2)
            yh = np.empty((chunk, b))
            for j in range(b):
                sl = np.r_[np.arange(ksh), ksh + j * kex + np.arange(kex)]
                yh[:, j] = (f[:, sl, :].sum(1) / k) @ W[:, j]
            acc += np.exp(1j * (yh @ Om.T)).sum(0)
        err = float(np.abs(acc / R - psi).max())
        tol = 6.0 / np.sqrt(R)
        if err > tol:
            print(f"  [FAIL] b={b}: loss does not model generation "
                  f"({err:.4f} > {tol:.4f})")
            sys.exit(1)

    # 2. every gradient, rho included, against finite differences
    for b in (3, 5):
        r2 = np.random.default_rng(10 + b)
        n = 16
        S = r2.normal(size=(n, 12)); Y = 0.15 * r2.random((n, b)) + 0.05
        t = 0.3 * r2.normal(size=bank.n_var); Om = 3.0 * r2.normal(size=(30, b))
        kw = dict(k_shots=8, gamma_risk=2.0, X_cond=r2.normal(size=(n, 3)),
                  Omega_cond=r2.normal(size=(30, 3)))
        _, gth, _, _, _, gr = cf_mmd_loss_and_grads_factorized(
            bank, S, Y, t, Om, 1e-3, compute_grad=True, rho=0.4, **kw)
        L = lambda tt, rr: cf_mmd_loss_and_grads_factorized(
            bank, S, Y, tt, Om, 1e-3, compute_grad=False, rho=rr, **kw)[0]
        h = 1e-5
        fr = (L(t, 0.4 + h) - L(t, 0.4 - h)) / (2 * h)
        ft = np.array([(L(t + h * e, 0.4) - L(t - h * e, 0.4)) / (2 * h)
                       for e in np.eye(t.size)])
        er = abs(gr - fr) / abs(fr)
        et = float(np.abs(gth - ft).max() / np.abs(ft).max())
        if er > 1e-6 or et > 1e-6:
            print(f"  [FAIL] b={b}: rho rel {er:.1e}, theta rel {et:.1e}")
            sys.exit(1)
    print("  [PASS] blocks of 3 and 5 pixels: loss matches generation within "
          "Monte Carlo noise, rho and theta gradients match finite differences")


if __name__ == "__main__":
    main()
