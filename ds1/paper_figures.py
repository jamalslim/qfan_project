#!/usr/bin/env python3
"""
Every figure for the Dataset 1 paper, each plot a separate figure.

    python ds1/paper_figures.py outputs/model_ds1_368_nq4_b2_s1_showers.npz --data data/ds1_368.npy

Reads a showers file from generate.py (MC data Y_te, QFAN Y_gen, untrained
circuit Y_ini) and writes, as PDF and PNG, in the style of the first QFAN
paper (mplhep ROOT style, MC data as black points with Poisson errors, models
as step histograms with their statistical band):

  figures/correlation/  corr_data, corr_qfan, corr_untrained
  figures/energy/       <q> for q = total, layer0, layer1, layer2, layer3, layer12
  figures/voxels/       voxel<j>
                        each: MC data, QFAN and the untrained circuit on one spectrum,
                        with the ratio of each model to MC data underneath
  figures/polar/        average_<layer>_<source>, single_<layer>_<source>
                        for source = data, qfan, untrained; average_grid: all layers,
                        MC data, QFAN and their ratio in one figure

Both figures of a quantity share the same x-axis; the polar plots of a layer
share one colour scale. --data points to the prepared data file so that the
polar plots are in MeV; without it they are in the prepared units.

Options: --voxels rings (default: first angular bin of every ring), all, or a
comma-separated list of voxel indices; --out figures.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
from scipy.stats import wasserstein_distance

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qfan.utils import corr_nan_safe                                  # noqa: E402

# ------------------------------------------------------------------
# style of the first QFAN paper (scripts/plot_paper_figures.py)
# ------------------------------------------------------------------
try:
    import mplhep as hep
    plt.style.use(hep.style.ROOT)
    HAVE_MPLHEP = True
except Exception:
    HAVE_MPLHEP = False
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 15, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 7, "ytick.major.size": 7,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "axes.linewidth": 1.1,
        "legend.frameon": False,
        "errorbar.capsize": 0,
        "axes.grid": False,
    })

C_DATA = "black"
C_QFAN = "#3b5bdb"
C_UNTR = "#e8590c"
MODELS = {"qfan": ("QFAN", C_QFAN), "untrained": ("QFAN untrained", C_UNTR)}

LAYERS = [(1, 8), (10, 16), (10, 19), (1, 5), (1, 5)]   # (angular, radial) bins
LAYER_NAMES = ["0", "1", "2", "3", "12"]
STARTS = np.cumsum([0] + [na * nr for na, nr in LAYERS])


def voxel_name(j):
    for li, (na, nr) in enumerate(LAYERS):
        if j < STARTS[li + 1]:
            return f"layer {LAYER_NAMES[li]}, ring {(j - STARTS[li]) // na + 1}"
    return ""


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# histograms with statistical (Poisson) errors
# ------------------------------------------------------------------
def hist_density(x, bins):
    """Density histogram with per-bin Poisson error: err = sqrt(k)/(N dx)."""
    k, edges = np.histogram(x, bins=bins)
    N = max(1, len(x))
    dx = np.diff(edges)
    return 0.5 * (edges[:-1] + edges[1:]), k / (N * dx), np.sqrt(k) / (N * dx), k


def spectrum_ratio_figure(x_data, models, bins, title, xlabel, folder, name):

    """One figure: the spectrum on top, with MC data as points with Poisson
    errors and every model in `models` (list of (key, samples)) as a step
    histogram with its statistical band; the ratio model / MC data of each
    model underneath, sharing the x-axis, as in the first paper. The y-axis
    is scaled to MC data and QFAN; a taller untrained spike runs off the top
    with its peak value marked, and ratios beyond the panel are drawn as
    triangles at its edge."""


    from matplotlib.gridspec import GridSpec
    c, dd, ed, kd = hist_density(x_data, bins)
    # gap between spectrum and ratio: HSPACE times the average panel height; the
    # figure grows with it so that both panels keep the size they have at 0.1
    HSPACE = 0.8
    fig = plt.figure(figsize=(7.4, 6.4 * (4.0 + 2.0 * HSPACE) / (4.0 + 2.0 * 0.1)))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.0], hspace=HSPACE)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    RMAX = 2.5
    axr.axhline(1.0, color="gray", lw=0.9, ls="--", zorder=1)
    top = dd.max() + ed.max()
    lines, hists = [], []
    for key, x in models:
        label, color = MODELS[key]
        _, dm, em, km = hist_density(x, bins)
        hists.append((key, dm, em, km, label, color))
        if key == "qfan":
            top = max(top, (dm + em).max())
        w1 = wasserstein_distance(x_data, x)
       # lines.append(f"{label}: $W_1$ = {w1:.4f}, spread {x.std() / max(x_data.std(), 1e-12):.2f}")
    ymax = 1.35 * top
    for key, dm, em, km, label, color in hists:
        ax.stairs(dm, bins, color=color, lw=1.8, label=label, zorder=4,
                  ls="-" if key == "qfan" else "--")
        ax.stairs(np.maximum(dm - em, 0), bins, baseline=dm + em, fill=True,
                  color=color, alpha=0.22, lw=0, zorder=3)
        if dm.max() > ymax:
            j = int(np.argmax(dm))
            right = c[j] > 0.5 * (bins[0] + bins[-1])
            ax.annotate(f"peak {dm.max():.1f}", xy=(c[j], 0.62 * ymax),
                        xytext=(-8 if right else 8, 0), textcoords="offset points",
                        color=color, fontsize=9, ha="right" if right else "left", va="center")
        r = (kd > 0) & (km > 0)
        ratio = dm[r] / dd[r]
        err = ratio * np.sqrt(1.0 / km[r] + 1.0 / kd[r])
        inside = ratio <= RMAX
        axr.errorbar(c[r][inside], ratio[inside], yerr=err[inside], fmt="o", ms=2.8,
                     lw=1.0, color=color, zorder=3)
        if (~inside).any():
            axr.plot(c[r][~inside], np.full((~inside).sum(), RMAX * 0.96), "^",
                     ms=4.5, color=color, zorder=3)
    m = dd > 0
    ax.errorbar(c[m], dd[m], yerr=ed[m], fmt="o", ms=3.5, lw=1.2, color=C_DATA,
                label="MC data", zorder=5)
   # ax.text(0.97, 0.95, title, transform=ax.transAxes, ha="right", va="top", fontsize=11)
    #ax.text(0.97, 0.84, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.set_ylabel("density")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, ymax)
    ax.legend(loc="best", fontsize=18)
    plt.setp(ax.get_xticklabels(), visible=False)
    axr.set_ylim(0.0, RMAX)
    axr.set_xlabel(xlabel)
    axr.set_ylabel("model / data", fontsize=18)
    save(fig, folder, name)


def two_plots(x_data, x_q, x_u, title, xlabel, folder, key, bins):
    """QFAN and the untrained circuit on the same spectrum and ratio panel."""
    spectrum_ratio_figure(x_data, [("qfan", x_q), ("untrained", x_u)], bins, title, xlabel,
                          folder, key)


def common_bins(xs, n):
    lo = min(np.quantile(x, 0.0005) for x in xs)
    hi = max(np.quantile(x, 0.9995) for x in xs)
    return np.linspace(lo, hi, n + 1)


# ------------------------------------------------------------------
# correlation matrices
# ------------------------------------------------------------------
def corr_figure(Y, title, folder, name):
    C = corr_nan_safe(Y)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for e in STARTS[1:-1]:
        ax.axhline(e - 0.5, color="k", lw=0.3, alpha=0.4)
        ax.axvline(e - 0.5, color="k", lw=0.3, alpha=0.4)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("voxel")
    ax.set_ylabel("voxel")
    ax.set_xticks(range(0, C.shape[0], 50))
    ax.set_yticks(range(0, C.shape[0], 50))
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("correlation")
    save(fig, folder, name)


# ------------------------------------------------------------------
# polar plots
# ------------------------------------------------------------------
def layer_blocks(Y):
    """(N, 368) ring-ordered -> per layer (N, n_angular, n_radial)."""
    out = []
    for li, (na, nr) in enumerate(LAYERS):
        out.append(Y[:, STARTS[li]:STARTS[li + 1]].reshape(-1, nr, na).transpose(0, 2, 1))
    return out


# colour-bar text sizes for the polar plots
CB_TICK = 14
CB_LABEL = 16


def log_colorbar_ticks(cb, vmin, vmax, size=CB_TICK):
    """Readable labels on a logarithmic colour bar in any style (the mplhep
    ROOT style enlarges tick labels). Decades are labelled when at least two
    fall inside the bar (at most four labels on a horizontal bar); otherwise
    1 and 3 times each decade on a horizontal bar, 1, 2 and 5 on a vertical
    one. Minor ticks are unlabelled, and all ticks get one explicit size."""
    horizontal = cb.orientation == "horizontal"
    decades = np.floor(np.log10(vmax)) - np.ceil(np.log10(vmin)) + 1
    if decades >= 2:
        cb.locator = LogLocator(base=10, subs=(1.0,), numticks=4 if horizontal else 8)
    else:
        cb.locator = LogLocator(base=10, subs=(1.0, 3.0) if horizontal else (1.0, 2.0, 5.0))
    cb.formatter = LogFormatterSciNotation(labelOnlyBase=False,
                                           minor_thresholds=(np.inf, np.inf))
    (cb.ax.xaxis if horizontal else cb.ax.yaxis).set_minor_formatter(NullFormatter())
    cb.ax.tick_params(which="both", labelsize=size)


def polar_figure(vals, na, nr, vmin, vmax, title, unit, folder, name):
    rep = int(np.ceil(72 / na))            # thin wedges so rings render smoothly
    th = np.linspace(0, 2 * np.pi, na * rep + 1)
    r_edges = np.arange(nr + 1, dtype=float)
    v = np.repeat(vals, rep, axis=0)
    v = np.where(v > 0, v, np.nan)
    fig = plt.figure(figsize=(5.2, 6.0))
    ax = fig.add_subplot(111, projection="polar")
    m = ax.pcolormesh(th, r_edges, v.T, cmap="viridis",
                      norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat", edgecolors="face", linewidth=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0, r_edges[-1])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, fontsize=13, pad=12)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05, shrink=0.9)
    log_colorbar_ticks(cb, vmin, vmax)
    cb.set_label(unit, fontsize=CB_LABEL)
    save(fig, folder, name)


def polar_figures(sources, unit_avg, unit_one, folder):
    rng = np.random.default_rng(0)
    for kind in ("average", "single"):
        blocks = {}
        for key, (name, Y) in sources.items():
            if kind == "average":
                blocks[key] = [b.mean(0) for b in layer_blocks(Y)]
            else:
                i = rng.integers(len(Y))
                blocks[key] = [b[0] for b in layer_blocks(Y[i:i + 1])]
        for li, (na, nr) in enumerate(LAYERS):
            pos = np.concatenate([blocks[k][li][blocks[k][li] > 0] for k in sources])
            vmax = pos.max() if pos.size else 1.0
            vmin = max(pos.min(), vmax * 1e-4) if pos.size else 1e-6
            for key, (name, _) in sources.items():
                what = "average shower" if kind == "average" else "one shower"
                polar_figure(blocks[key][li], na, nr, vmin, vmax,
                             f"{name}, layer {LAYER_NAMES[li]} ({what})",
                             unit_avg if kind == "average" else unit_one,
                             folder, f"{kind}_layer{LAYER_NAMES[li]}_{key}")


def polar_average_grid(Y_data, Y_model, unit, folder, name="average_grid"):
    """Average shower per layer in one figure: MC data (top row), QFAN
    (middle row) on a shared logarithmic colour scale per layer, and the
    ratio QFAN / MC data (bottom row). Rings are drawn with equal width."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm
    Bd = [b.mean(0) for b in layer_blocks(Y_data)]
    Bm = [b.mean(0) for b in layer_blocks(Y_model)]
    fig = plt.figure(figsize=(3.3 * len(LAYERS), 13.2))
    gs = GridSpec(3, len(LAYERS), figure=fig, hspace=0.28, wspace=0.12)
    for li, (na, nr) in enumerate(LAYERS):
        rep = int(np.ceil(72 / na))
        th = np.linspace(0, 2 * np.pi, na * rep + 1)
        re = np.arange(nr + 1, dtype=float)
        pos = np.r_[Bd[li][Bd[li] > 0], Bm[li][Bm[li] > 0]]
        vmax = pos.max(); vmin = max(pos.min(), vmax * 1e-4)
        ratio = np.where(Bd[li] > 0, Bm[li] / np.where(Bd[li] > 0, Bd[li], 1), np.nan)
        rows = [(Bd[li], LogNorm(vmin=vmin, vmax=vmax), "viridis", "MC data"),
                (Bm[li], LogNorm(vmin=vmin, vmax=vmax), "viridis", "QFAN"),
                (ratio, TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5), "RdBu_r", "QFAN / MC data")]
        for row, (vals, norm, cmap, rlabel) in enumerate(rows):
            ax = fig.add_subplot(gs[row, li], projection="polar")
            v = np.repeat(vals, rep, axis=0)
            if row < 2:
                v = np.where(v > 0, v, np.nan)
            m = ax.pcolormesh(th, re, v.T, cmap=cmap, norm=norm, shading="flat", edgecolors="face", linewidth=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_ylim(0, re[-1]); ax.spines["polar"].set_visible(False)
            if row == 0:
                ax.set_title(f"layer {LAYER_NAMES[li]}", fontsize=15, pad=12)
            if li == 0:
                ax.set_ylabel(rlabel, labelpad=30, fontsize=14)
                ax.yaxis.set_label_position("left")
            cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04, shrink=0.9)
            if row < 2:
                log_colorbar_ticks(cb, vmin, vmax)
            else:
                cb.set_ticks([0.6, 0.8, 1.0, 1.2, 1.4])
                cb.ax.tick_params(which="both", labelsize=CB_TICK)
            if li == len(LAYERS) // 2:
                cb.set_label(unit if row < 2 else "ratio", fontsize=CB_LABEL)
    save(fig, folder, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("showers", help="showers file from generate.py")
    ap.add_argument("--data", default=None,
                    help="the prepared data file (.npy with its .json), for MeV in the polar plots")
    ap.add_argument("--voxels", default="rings",
                    help="rings (default): first angular bin of every ring plus the paper's twelve voxels; all; or a list j1,j2,...")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    r = np.load(a.showers, allow_pickle=True)
    Y_te, Y_gen, Y_ini = r["Y_te"], r["Y_gen"], r["Y_ini"]
    if Y_te.shape[1] != 368:
        sys.exit("expects 368-voxel showers in ring order")
    out = ROOT / a.out
    folders = {k: out / k for k in ("correlation", "energy", "voxels", "polar")}
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)

    # correlation matrices
    corr_figure(Y_te, "MC data", folders["correlation"], "corr_data")
    corr_figure(Y_gen, "QFAN", folders["correlation"], "corr_qfan")
    corr_figure(Y_ini, "QFAN untrained", folders["correlation"], "corr_untrained")

    # energy sums
    sums = [("total", "total energy", lambda Y: Y.sum(1))]
    for li, ln in enumerate(LAYER_NAMES):
        sums.append((f"layer{ln}", f"layer {ln} energy",
                     (lambda a0, a1: (lambda Y: Y[:, a0:a1].sum(1)))(STARTS[li], STARTS[li + 1])))
    for key, title, fn in sums:
        xd, xq, xu = fn(Y_te), fn(Y_gen), fn(Y_ini)
        two_plots(xd, xq, xu, title, f"{title} (norm.)", folders["energy"], key,
                   common_bins((xd, xq, xu), 40))

    # single voxels
    if a.voxels == "rings":
        # first angular bin of every ring, plus the twelve voxels shown in the paper
        voxels = sorted({STARTS[li] + na * k for li, (na, nr) in enumerate(LAYERS) for k in range(nr)}
                        | {0, 33, 66, 100, 133, 166, 200, 233, 266, 300, 333, 367})
    elif a.voxels == "all":
        voxels = list(range(368))
    else:
        voxels = [int(v) for v in a.voxels.split(",")]
    for j in voxels:
        xd, xq, xu = Y_te[:, j], Y_gen[:, j], Y_ini[:, j]
        hi = max(np.quantile(x, 0.9995) for x in (xd, xq, xu))
        if hi <= 1e-12:
            continue
        two_plots(xd, xq, xu, f"voxel {j}, {voxel_name(j)}", "intensity (norm.)",
                   folders["voxels"], f"voxel{j:03d}", np.linspace(0.0, hi * 1.02, 33))

    # polar plots
    factor, unit_avg, unit_one = 1.0, "mean energy per voxel (norm.)", "energy per voxel (norm.)"
    if a.data:
        info = json.loads(pathlib.Path(a.data).with_suffix(".json").read_text())
        if info.get("energy_scale") and info.get("energy_mev"):
            factor = float(info["energy_mev"]) / float(info["energy_scale"])
            unit_avg, unit_one = "mean energy per voxel [MeV]", "energy per voxel [MeV]"
    sources = {"data": ("MC data", Y_te * factor), "qfan": ("QFAN", Y_gen * factor),
               "untrained": ("QFAN untrained", Y_ini * factor)}
    polar_figures(sources, unit_avg, unit_one, folders["polar"])
    polar_average_grid(Y_te * factor, Y_gen * factor, unit_avg, folders["polar"])

    n = sum(len(list(f.glob("*.pdf"))) for f in folders.values())
    print(f"{n} figures (PDF and PNG) written to {out}"
          f"   [{'mplhep ROOT style' if HAVE_MPLHEP else 'mplhep not installed: fallback style'}]")


if __name__ == "__main__":
    main()
