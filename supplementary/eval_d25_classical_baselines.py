"""
d=25 classical baseline table: QFAN against classical generative
baselines on the same data and protocol.

Same data protocol as every d=25 run (subset 6000 -> 4800/1200, split seed
42). All baselines are fit on TRAIN only and sampled free-running where
autoregressive; test is touched once per baseline for the reported row.

Baselines (matched or sub-quantum parameter budget; the quantum pipeline
has ~559 trained parameters -- theta 18 + A 512 + b 16 + rho 13 -- plus
closed-form decoders, which every baseline below is also granted):

  B1  Independence: per-pixel empirical marginals, independent sampling.
      (0 correlation parameters; the floor any correlation claim must beat.)
  B2  Gaussian copula: empirical marginals + full Gaussian copula
      (d(d-1)/2 = 300 correlation parameters).
  B3  Autoregressive ridge chain + Gaussian residuals: per-block ridge on
      the true prefix (closed form), residual covariance per block; free-
      running generation with the SAME nonnegative clip as the quantum
      pipeline (~625 mean + 38 covariance parameters).
  B4  Autoregressive ridge chain + copula residuals: as B3 but residuals
      sampled from their empirical per-block joint (rank-preserving),
      free-running.

Honest note printed with the table: B2's marginals are nonparametric
(memorized training quantiles) -- exactly the memorization the Born-record design
removed from the quantum pipeline; it is included because referees will
run it, not because the comparison is symmetric.
"""
import pathlib
import sys

import numpy as np
from scipy.stats import wasserstein_distance, norm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qfan import correlation_error_summary, corr_nan_safe      # noqa: E402
from train import build_problem, spearman_corr   # noqa: E402


def metrics(Y_te, Y, blocks, d):
    w1 = float(np.mean([wasserstein_distance(Y_te[:, j], Y[:, j])
                        for j in range(d)]))
    cs = correlation_error_summary(corr_nan_safe(Y_te), corr_nan_safe(Y),
                                   blocks)
    sp = correlation_error_summary(spearman_corr(Y_te), spearman_corr(Y),
                                   blocks)
    return dict(w1=w1, off=cs["corr_mae_offdiag"],
                win=cs["corr_mae_within"], crs=cs["corr_mae_cross"],
                sp=sp["corr_mae_offdiag"])


def empirical_quantile_sample(col_sorted, u):
    n = col_sorted.size
    idx = np.clip((u * n).astype(int), 0, n - 1)
    return col_sorted[idx]


def main():
    (cfg, Y_tr, Y_te, d, blocks, sketcher, cache, spec,
     bank) = build_problem()
    n_te = len(Y_te)
    rng = np.random.default_rng(cfg.data.seed)
    rows = {}

    tr_sorted = [np.sort(Y_tr[:, j]) for j in range(d)]

    # ---- B1 independence ----
    U = rng.random((n_te, d))
    Y1 = np.column_stack([empirical_quantile_sample(tr_sorted[j], U[:, j])
                          for j in range(d)])
    rows["B1 independence (marginals only)"] = metrics(Y_te, Y1, blocks, d)

    # ---- B2 Gaussian copula ----
    # ranks -> normal scores -> correlation -> sample -> back-transform
    Z = np.column_stack([
        norm.ppf((np.argsort(np.argsort(Y_tr[:, j])) + 0.5) / len(Y_tr))
        for j in range(d)])
    R = np.corrcoef(Z.T)
    Lc = np.linalg.cholesky(R + 1e-9 * np.eye(d))
    Zs = rng.standard_normal((n_te, d)) @ Lc.T
    Us = norm.cdf(Zs)
    Y2 = np.column_stack([empirical_quantile_sample(tr_sorted[j], Us[:, j])
                          for j in range(d)])
    rows["B2 Gaussian copula (300 corr params)"] = metrics(
        Y_te, Y2, blocks, d)

    # ---- B3 / B4 autoregressive ridge chain ----
    alpha = 1e-3
    coefs, resid_bank, n_par = [], [], 0
    for (s, bsz) in blocks:
        X = np.hstack([Y_tr[:, :s], np.ones((len(Y_tr), 1))])
        Yb = Y_tr[:, s:s + bsz]
        W = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]),
                            X.T @ Yb)
        res = Yb - X @ W
        coefs.append(W)
        resid_bank.append(res)
        n_par += W.size + bsz * (bsz + 1) // 2

    def chain_sample(mode):
        Y = np.zeros((n_te, d))
        for bi, (s, bsz) in enumerate(blocks):
            X = np.hstack([Y[:, :s], np.ones((n_te, 1))])
            mu = X @ coefs[bi]
            if mode == "gauss":
                C = np.cov(resid_bank[bi].T).reshape(bsz, bsz)
                Lb = np.linalg.cholesky(C + 1e-12 * np.eye(bsz))
                eps = rng.standard_normal((n_te, bsz)) @ Lb.T
            else:                            # empirical joint residuals
                take = rng.integers(0, len(resid_bank[bi]), size=n_te)
                eps = resid_bank[bi][take]
            Y[:, s:s + bsz] = np.maximum(mu + eps, 0.0)
        return Y

    rows[f"B3 AR ridge + Gaussian residuals (~{n_par} params)"] = metrics(
        Y_te, chain_sample("gauss"), blocks, d)
    rows["B4 AR ridge + empirical joint residuals"] = metrics(
        Y_te, chain_sample("emp"), blocks, d)

    # ---- quantum references, from shipped artifacts ----
    rq = np.load(PROJECT_ROOT / "outputs/samples_d25.npz",
                 allow_pickle=True)
    rows["Q  QFAN all-quantum (shipped)"] = metrics(
        Y_te, rq["Y_head"], blocks, d)

    print("=" * 96)
    print(f"{'model':46s}  {'W1':>7s} {'off':>7s} {'win':>7s} "
          f"{'crs':>7s} {'Sp off':>7s}")
    print("-" * 96)
    for name, m in rows.items():
        print(f"{name:46s}  {m['w1']:7.4f} {m['off']:7.4f} "
              f"{m['win']:7.4f} {m['crs']:7.4f} {m['sp']:7.4f}")
    print("-" * 96)
    print("NOTE: B2/B4 sample memorized training residuals/quantiles -- the")
    print("exact classical machinery the Born-record design removed. The quantum row")
    print("claims functional necessity within its own algorithm "
          "(certificates C1-C6), NOT superiority over these baselines.")
    np.savez_compressed(
        PROJECT_ROOT / "outputs/qfan_d25_classical_baselines.npz",
        meta=np.array([rows], dtype=object))
    print("[SAVE] outputs/qfan_d25_classical_baselines.npz")


if __name__ == "__main__":
    main()
