#!/usr/bin/env python3
"""
plot_k_sweep.py -- sweep the number of Born records per block and show that
k is a MODEL parameter, not a precision setting.

Two results, both against the usual intuition for a variational algorithm:

  (a) marginal fidelity is OPTIMIZED at finite k (minimum near k=64 for the
      d=25 QFAN state). Increasing k past the optimum makes the model worse.
  (b) the total-energy variance falls monotonically toward zero as k grows,
      crossing the data value near k~35. Certificate C1 (exact conditional
      means, i.e. k -> infinity) is the endpoint of this curve, not a
      separate phenomenon.

Caveat worth keeping in mind when reading the output: the off-diagonal
Pearson correlation error does NOT track (a) or (b). A correlation
coefficient is scale-invariant and therefore blind to a uniform contraction
of the generated distribution until that contraction is exact. Use
dispersion-sensitive observables (Var(E), W1) to see the approach to
determinism.

Usage:
    python scripts/plot_k_sweep.py                 # sweep + figure
    python scripts/plot_k_sweep.py --from-cache    # re-plot only
"""
import argparse, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve()
PROJECT_ROOT = next(c for c in [ROOT.parent] + list(ROOT.parents)
                    if (c / "src" / "qfan").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

KS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
      1024, 2048, 4096]
CACHE = PROJECT_ROOT / "outputs" / "k_sweep.npz"


def sweep(state, seeds=3, n=1200):
    from qfan import correlation_error_summary, corr_nan_safe
    from qfan.born import fit_all_blocks_born, sample_progressive_born
    from train import build_problem
    from scipy.stats import wasserstein_distance
    (cfg, Y_tr, Y_te, d, blocks, sk, cache, spec, bank) = build_problem()
    r = np.load(state, allow_pickle=True)
    bank.set_theta(r["theta_final"])
    bank.A = r["A_final"].copy(); bank.b = r["b_final"].copy()
    rho = r["rho_final"].copy()
    Ct = corr_nan_safe(Y_te)
    rows = []
    for K in KS:
        models = fit_all_blocks_born(bank, cache, Y_tr, blocks,
                                     cfg.bank.ridge_alpha, noise_aware=True,
                                     k_shots=K)
        w1s, ves, offs = [], [], []
        for s_ in range(seeds):
            rng = np.random.default_rng(7000 + s_)
            Y = sample_progressive_born(bank, models, d, blocks, sk, n, rng,
                                        k=K, record_share=rho)
            w1s.append(np.mean([wasserstein_distance(Y_te[:, j], Y[:, j])
                                for j in range(d)]))
            ves.append(Y.sum(1).var())
            offs.append(correlation_error_summary(
                Ct, corr_nan_safe(Y), blocks)["corr_mae_offdiag"])
        rows.append([K, np.mean(w1s), np.std(w1s), np.mean(ves),
                     np.std(ves), np.mean(offs), np.std(offs)])
        print(f"  k={K:>5}  W1={np.mean(w1s):.4f}  Var(E)={np.mean(ves):.4f}"
              f"  off={np.mean(offs):.4f}", flush=True)
    A = np.array(rows)
    np.savez(CACHE, A=A, var_data=float(Y_te.sum(1).var()), w1_floor=0.00159)
    return A, float(Y_te.sum(1).var()), 0.00159


def plot(A, var_data, floor, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k, w1, w1e, ve, vee, off, offe = A.T
    plt.rcParams.update({"font.size": 11, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False,
                         "xtick.minor.visible": True,
                         "ytick.minor.visible": True, "axes.linewidth": 1.0})
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
    a = ax[0]
    a.axvspan(3, 32, color="tab:blue", alpha=0.06)
    a.axvspan(128, 5000, color="tab:red", alpha=0.06)
    a.errorbar(k, w1, yerr=w1e, fmt="o-", ms=4, lw=1.6, color="#2a4b8d",
               capsize=2, zorder=3)
    i = int(np.argmin(w1))
    a.plot(k[i], w1[i], "*", ms=17, color="#d62728", zorder=5,
           label=f"optimum $k={int(k[i])}$")
    a.axhline(floor, ls=":", color="0.45", lw=1.1)
    a.text(4.6, floor * 1.25, "statistical floor", fontsize=8, color="0.35")
    a.text(7, 0.042, "too much\nsampling noise", fontsize=8.5,
           color="tab:blue", ha="center")
    a.text(900, 0.040, "too little:\napproaching\ndeterminism", fontsize=8.5,
           color="tab:red", ha="center")
    a.set_xscale("log"); a.set_xlabel("records per block, $k$")
    a.set_ylabel(r"marginal error  $\bar W_1$")
    a.set_title("(a)  fidelity is optimized at finite $k$", fontsize=10.5,
                loc="left")
    a.legend(fontsize=9, loc="lower left")
    a.set_xlim(3.4, 5200); a.set_ylim(0, 0.050)
    b = ax[1]
    b.errorbar(k, ve, yerr=vee, fmt="o-", ms=4, lw=1.6, color="#2a4b8d",
               capsize=2, zorder=3)
    b.axhline(var_data, color="k", lw=1.3, ls="--")
    b.text(4.6, var_data * 1.18, "MC data", fontsize=9)
    kx = np.interp(var_data, ve[::-1], k[::-1])
    b.plot([kx], [var_data], "*", ms=17, color="#d62728", zorder=5,
           label=f"matches data at $k\\approx{kx:.0f}$")
    b.annotate(r"$\mathrm{Var}(E)\to 0$" + "\n" + r"deterministic (C1)",
               xy=(k[-1], ve[-1]), xytext=(700, ve[-1] * 0.32), fontsize=9,
               color="#c0392b", ha="center",
               arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("records per block, $k$")
    b.set_ylabel(r"total-energy variance  $\mathrm{Var}(E)$")
    b.set_title("(b)  the sampling noise is the dispersion", fontsize=10.5,
                loc="left")
    b.legend(fontsize=9, loc="upper right"); b.set_xlim(3.4, 5200)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"[SAVE] {out}.pdf/.png   W1 min at k={int(k[i])}; "
          f"Var(E) matches data at k={kx:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state",
                    default=str(PROJECT_ROOT / "outputs" /
                                "model_d25.npz"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" /
                                         "fig_k_sweep"))
    args = ap.parse_args()
    if args.from_cache and CACHE.exists():
        d = np.load(CACHE)
        A, vd, fl = d["A"], float(d["var_data"]), float(d["w1_floor"])
    else:
        A, vd, fl = sweep(args.state, seeds=args.seeds)
    plot(A, vd, fl, args.out)


if __name__ == "__main__":
    main()
