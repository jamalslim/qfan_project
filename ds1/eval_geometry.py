#!/usr/bin/env python3
"""
Evaluate and plot a model trained on any geometry.

    python ds1/eval_geometry.py outputs/model_ds1_53_nq4_b2_s1.npz

Prints the metrics the paper uses and writes figures to plots_<tag>/.

The central question is whether the TRAINED circuit beats the UNTRAINED one on
both observables. Every results file already contains the untrained reference
(Y_ini: circuit frozen at initialization, classical stages refitted) and the
other ablations, so this script needs no extra training.

What it reports
  per-pixel W1               marginal fidelity, against the statistical floor
  off-diagonal correlation   joint structure, split within and across blocks
  mean |corr|                how much correlation the data has, and the model
  total energy               W1 of the sum, and its variance
  sparse pixels              zero fraction per pixel, and how those pixels fare
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve()
PROJECT_ROOT = next(c for c in [ROOT.parent] + list(ROOT.parents)
                    if (c / "src" / "qfan").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scipy.stats import wasserstein_distance, rankdata          # noqa: E402
from qfan.utils import corr_nan_safe                            # noqa: E402

VARIANTS = [
    ("Y_gen", "trained"),
    ("Y_ini", "untrained circuit"),
    ("Y_cls", "classical features"),
    ("Y_scr", "scrambled sketch"),
    ("Y_mean", "conditional means"),
    ("Y_cop", "calibrated"),
]


def blocks_for(d, b):
    out, s = [], 0
    while s < d:
        out.append((s, min(b, d - s)))
        s += b
    return out


def masks(d, blocks):
    off = ~np.eye(d, dtype=bool)
    win = np.zeros((d, d), bool)
    for s, n in blocks:
        win[s:s + n, s:s + n] = True
    win &= off
    return off, win, off & ~win


def spearman(Y):
    return corr_nan_safe(np.apply_along_axis(rankdata, 0, Y))


def metrics(Yt, Y, blocks, to_phys=None):
    d = Yt.shape[1]
    off, win, crs = masks(d, blocks)
    w1 = np.array([wasserstein_distance(Yt[:, j], Y[:, j]) for j in range(d)])
    deg = len(np.unique(np.round(Y, 12), axis=0)) <= 1
    Ct, C = corr_nan_safe(Yt), corr_nan_safe(Y)
    D = np.abs(C - Ct)
    tri = np.triu_indices(d, 1)
    ph = np.ones(d) if to_phys is None else to_phys
    Et, E = (Yt * ph).sum(1), (Y * ph).sum(1)
    return dict(
        w1=w1, w1_mean=float(w1.mean()), degenerate=deg,
        off=float(D[off].mean()), within=float(D[win].mean()),
        cross=float(D[crs].mean()),
        sp=float(np.abs(spearman(Y) - spearman(Yt))[off].mean()),
        abscorr=float(np.abs(C[tri]).mean()),
        w1E=float(wasserstein_distance(Et, E) / (Et.std() + 1e-300)),
        varE=float(E.var() / (Et.var() + 1e-300)), C=C)


def w1_floor(Yt, seed=0, reps=20):
    rng = np.random.default_rng(seed)
    n, d = Yt.shape
    v = []
    for _ in range(reps):
        p = rng.permutation(n)
        a, b = Yt[p[:n // 2]], Yt[p[n // 2:]]
        v.append(np.mean([wasserstein_distance(a[:, j], b[:, j])
                          for j in range(d)]))
    return float(np.mean(v))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="outputs/model_<tag>.npz")
    ap.add_argument("--data", default=None,
                    help="the .npy the model was trained on. Needed for models "
                         "trained before the data path was recorded, to find "
                         "the layer widths and the normalization.")
    a = ap.parse_args()
    path = pathlib.Path(a.model)
    r = np.load(path, allow_pickle=True)
    meta = r["meta"][0] if "meta" in r.files else {}
    Yt = r["Y_te"]
    d = Yt.shape[1]
    b = int(meta.get("block_size", 2))
    # use the layout the model was actually trained with
    blocks = ([tuple(x) for x in meta["blocks"]] if meta.get("blocks")
              else blocks_for(d, b))
    layer_widths = meta.get("layer_widths")
    tag = path.stem.replace("model_", "")

    # Invert the preprocessing so energy observables are in physical units.
    # With per-pixel normalization, summing the stored pixels would weight a
    # nearly empty layer as heavily as the dominant one.
    to_phys = np.ones(d)
    norm_used = "unknown"
    src = a.data or meta.get("data_path") or ""
    side = pathlib.Path(src).with_suffix(".json") if src else None
    if side is not None and side.exists():
        import json
        info = json.loads(side.read_text())
        norm_used = info.get("norm", "global")
        if info.get("pixel_scale"):
            to_phys = 1.0 / np.asarray(info["pixel_scale"], float)
        elif info.get("energy_scale"):
            to_phys = np.full(d, 1.0 / float(info["energy_scale"]))
        if layer_widths is None:
            layer_widths = info.get("layer_widths")
    if layer_widths is None:
        print("  NOTE: layer widths unknown, so the per-layer table is skipped. "
              "Pass --data <the .npy>.")

    floor = w1_floor(Yt)
    tri = np.triu_indices(d, 1)
    data_abscorr = float(np.abs(corr_nan_safe(Yt)[tri]).mean())

    print(f"\n{'='*78}\n  {tag}\n{'='*78}")
    print(f"  d = {d}   block size {b}   blocks {len(blocks)}   "
          f"theta {r['theta_final'].shape[0]}   test showers {len(Yt)}")
    sizes = sorted({n for _, n in blocks})
    print(f"  normalization {norm_used}   block sizes "
          f"{sizes[0] if len(sizes) == 1 else sizes}")
    print(f"  W1 floor (finite test statistics)  {floor:.4f}")
    print(f"  data: mean |corr| {data_abscorr:.3f}")
    print(f"  energy columns are in units of the data: W1(E) divided by the "
          f"data's std(E), Var(E) as a ratio to the data's (1.0 is exact)")

    print(f"\n  {'variant':22s} {'W1':>8} {'xfloor':>7} {'off':>7} "
          f"{'within':>7} {'cross':>7} {'Sp':>7} {'|corr|':>7} {'W1(E)':>7} "
          f"{'Var(E)':>8}")
    res = {}
    for key, label in VARIANTS:
        if key not in r.files:
            continue
        m = metrics(Yt, r[key], blocks, to_phys)
        res[key] = m
        if m["degenerate"]:
            print(f"  {label:22s} {m['w1_mean']:8.4f} {m['w1_mean']/floor:7.1f}"
                  f"   degenerate: every sample identical, correlation "
                  f"undefined")
            continue
        print(f"  {label:22s} {m['w1_mean']:8.4f} {m['w1_mean']/floor:7.1f} "
              f"{m['off']:7.4f} {m['within']:7.4f} {m['cross']:7.4f} "
              f"{m['sp']:7.4f} {m['abscorr']:7.3f} {m['w1E']:7.4f} "
              f"{m['varE']:8.4f}")

    # ---- the verdict ----------------------------------------------------
    if "Y_gen" in res and "Y_ini" in res:
        t, u = res["Y_gen"], res["Y_ini"]
        print(f"\n  TRAINED vs UNTRAINED")
        print(f"    correlation error   {u['off']:.4f} -> {t['off']:.4f}   "
              f"({'better' if t['off'] < u['off'] else 'WORSE'})")
        print(f"    marginal W1         {u['w1_mean']:.4f} -> {t['w1_mean']:.4f}"
              f"   ({'better' if t['w1_mean'] < u['w1_mean'] else 'WORSE'})")
        print(f"    correlation carried {u['abscorr']:.3f} -> {t['abscorr']:.3f}"
              f"   of the data's {data_abscorr:.3f}")
        if data_abscorr < 0.15:
            print("    NOTE: the data itself has little inter-pixel correlation, "
                  "so the correlation\n    comparison has little room to "
                  "separate trained from untrained.")

    # ---- per-layer energy -----------------------------------------------
    if layer_widths and "Y_gen" in res:
        edges = np.cumsum([0] + list(layer_widths))
        Yg = r["Y_gen"]
        tot_t = (Yt * to_phys).sum(1)
        tot_g = (Yg * to_phys).sum(1)
        print(f"\n  per-layer energy (physical units)")
        print(f"    {'layer':>5} {'data share':>11} {'model share':>12} "
              f"{'W1/std':>8} {'corr with next, data':>22} {'model':>7}")
        Lt = [(Yt[:, a:c] * to_phys[a:c]).sum(1) for a, c in zip(edges, edges[1:])]
        Lg = [(Yg[:, a:c] * to_phys[a:c]).sum(1) for a, c in zip(edges, edges[1:])]
        for i in range(len(Lt)):
            sh_t = Lt[i].mean() / tot_t.mean()
            sh_g = Lg[i].mean() / tot_g.mean()
            w = wasserstein_distance(Lt[i], Lg[i]) / (Lt[i].std() + 1e-300)
            if i + 1 < len(Lt):
                ct = np.corrcoef(Lt[i], Lt[i + 1])[0, 1]
                cg = np.corrcoef(Lg[i], Lg[i + 1])[0, 1]
                nxt = f"{ct:+22.3f} {cg:+7.3f}"
            else:
                nxt = ""
            print(f"    {i:>5} {100*sh_t:10.1f}% {100*sh_g:11.1f}% {w:8.3f} {nxt}")
        print("    the last two columns test energy conservation: the data should "
              "show\n    negative layer-to-layer correlation, and the model should "
              "match its sign")

    # ---- sparse pixels --------------------------------------------------
    zf = (Yt == 0).mean(axis=0)
    if zf.max() > 0.05 and "Y_gen" in res:
        order = np.argsort(-zf)[:6]
        w1g = res["Y_gen"]["w1"]
        print(f"\n  sparsest pixels (zero fraction in the data, and model W1)")
        for j in order:
            if zf[j] <= 0.05:
                break
            print(f"    pixel {j:>3}   zeros {100*zf[j]:5.1f}%   W1 {w1g[j]:.4f}"
                  f"   ({w1g[j]/np.median(w1g):.1f}x the median pixel)")

    # ---- figures --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import mplhep as hep
            plt.style.use(hep.style.ROOT)
        except Exception:
            pass
    except ImportError:
        print("\n  matplotlib not available, skipping figures")
        return

    out = PROJECT_ROOT / f"plots_{tag}"
    out.mkdir(exist_ok=True)

    panels = [("MC truth", corr_nan_safe(Yt))]
    for key, label in (("Y_gen", "QFAN, trained"),
                       ("Y_ini", "untrained circuit")):
        if key in res and not res[key]["degenerate"]:
            panels.append((label, res[key]["C"]))
    fig, ax = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    ax = np.atleast_1d(ax)
    for a, (label, C) in zip(ax, panels):
        im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_title(label, fontsize=13)
        a.set_xlabel("pixel")
        a.set_ylabel("pixel")
    fig.colorbar(im, ax=ax, fraction=0.025, label="correlation")
    fig.savefig(out / "fig_corr.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    Eg = (r["Y_gen"] * to_phys).sum(1)
    Et = (Yt * to_phys).sum(1)
    fig, a = plt.subplots(figsize=(6.4, 4.6))
    bins = np.linspace(min(Et.min(), Eg.min()), max(Et.max(), Eg.max()), 40)
    a.hist(Et, bins, density=True, histtype="step", lw=1.8, color="k",
           label="MC truth")
    a.hist(Eg, bins, density=True, histtype="step", lw=1.8, color="#2a6fdb",
           label="QFAN")
    a.set_xlabel("total energy")
    a.set_ylabel("density")
    a.legend(frameon=False)
    fig.savefig(out / "fig_energy.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_energy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    show = np.linspace(0, d - 1, min(d, 12)).astype(int)
    cols = 4
    rows = int(np.ceil(len(show) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows))
    for a, j in zip(np.ravel(axs), show):
        lo, hi = np.percentile(np.r_[Yt[:, j], r["Y_gen"][:, j]], [0.5, 99.5])
        bj = np.linspace(lo, hi if hi > lo else lo + 1e-6, 30)
        a.hist(Yt[:, j], bj, density=True, histtype="step", color="k", lw=1.4)
        a.hist(r["Y_gen"][:, j], bj, density=True, histtype="step",
               color="#2a6fdb", lw=1.4)
        a.set_title(f"pixel {j}", fontsize=10)
        a.tick_params(labelsize=8)
    for a in np.ravel(axs)[len(show):]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_marginals.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figures written to {out}/")


if __name__ == "__main__":
    main()
