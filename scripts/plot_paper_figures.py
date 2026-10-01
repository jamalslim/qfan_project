"""
Paper figures for QFAN -- ROOT/HEP style, with statistical errors.

Style: `mplhep` with `hep.style.ROOT` when installed (pip install mplhep);
otherwise a faithful built-in ROOT-style fallback (ticks-in on all four
sides, minor ticks, Helvetica-like sans, no grid) so the script runs
everywhere. All model curves are the raw QFAN sampler
(all-quantum randomness); the copula-calibrated variant is never plotted.

Conventions (HEP standard):
  * MC "data" -> black points with Poisson error bars  sqrt(k)/(N dx)
  * QFAN      -> blue step histogram with statistical uncertainty band
  * untrained -> orange step histogram (certificate C4), where included
  * ratio panels QFAN/data with propagated errors under each spectrum

Produces (in plots/):
  fig_corr_data.png            correlation matrix, MC test data
  fig_corr_qfan.png            correlation matrix, QFAN raw
  fig_corr_untrained.png       correlation matrix, untrained circuit (C4)
  fig_marginal_pixel_XXXX.pdf/.png   ONE standalone figure per pixel
                                     (spectrum + ratio panel); _untr variants
                                     additionally overlay the untrained circuit
  fig_energy_sum.png           total-energy spectrum + ratio panel

Run after scripts/evaluate.py (and optionally
scripts/run_d25_born_ibm.py for the hardware overlays):
    python scripts/plot_paper_figures.py --d 25
"""

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import wasserstein_distance

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from qfan.utils import corr_nan_safe                                # noqa: E402

# ------------------------------------------------------------------
# style: real mplhep ROOT style if available, faithful fallback if not
# ------------------------------------------------------------------
try:
    import mplhep as hep
    plt.style.use(hep.style.ROOT)
    HAVE_MPLHEP = True
except Exception:
    HAVE_MPLHEP = False
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial",
                            "DejaVu Sans"],
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
import argparse                                                    # noqa: E402

_ap = argparse.ArgumentParser(description="Simulator versus MC paper figures.")
_ap.add_argument("--d", type=int, choices=[12, 25], default=25,
                 help="image size to plot")
_A = _ap.parse_args()

LABEL = f"QFAN raw (simulation)   d = {_A.d}"



# The two image sizes read different artifacts. d=12 plots straight from the
# training output; d=25 plots the raw evaluation, which is a separate
# configuration (quad features, k=32) and so does not match Table I.
if _A.d == 12:
    OUT = PROJECT_ROOT / "plots_d12"
    _MAIN = "model_d12.npz"
    _KEY = "Y_gen"
else:
    OUT = PROJECT_ROOT / "plots_d25"
    _MAIN = "samples_d25.npz"
    _KEY = "Y_head"
OUT.mkdir(exist_ok=True)

r = np.load(PROJECT_ROOT / "outputs" / _MAIN, allow_pickle=True)
Y_te, Y_gen = r["Y_te"], r[_KEY]

# untrained-circuit reference (certificate C4) lives in the training artifact
_ref = ("model_d12.npz" if _A.d == 12
        else "model_d25.npz")
_rr = np.load(PROJECT_ROOT / "outputs" / _ref, allow_pickle=True)
Y_ini = _rr["Y_ini"]
# Y_ini is the untrained-circuit reference: theta frozen at initialization
# with every classical stage refitted. It is plotted alongside the trained
# model because the gap between them is the paper's central claim.

# Hardware overlays are disabled. This script produces simulator versus MC
# figures only. The hardware path and its results are in _hardware_archive/.
Y_ibm = None
IBM_LABEL = None
C_IBM = "#c2255c"

d = Y_te.shape[1]


# ------------------------------------------------------------------
# histogram helpers with statistical (Poisson) errors
# ------------------------------------------------------------------
def hist_density(x, bins):
    """Density histogram with per-bin Poisson error: err = sqrt(k)/(N dx)."""
    k, edges = np.histogram(x, bins=bins)
    N = max(1, len(x))
    dx = np.diff(edges)
    dens = k / (N * dx)
    err = np.sqrt(np.maximum(k, 0)) / (N * dx)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, dens, err, edges


def draw_data_points(ax, x, bins, label="MC data"):
    """HEP convention: data as black markers with Poisson error bars."""
    c, dens, err, _ = hist_density(x, bins)
    mask = dens > 0
    ax.errorbar(c[mask], dens[mask], yerr=err[mask], fmt="o", ms=3.5,
                lw=1.2, color=C_DATA, label=label, zorder=5)
    return c, dens, err


def draw_model_step(ax, x, bins, color, label):
    """Model as step histogram + shaded statistical uncertainty band."""
    c, dens, err, edges = hist_density(x, bins)
    ax.stairs(dens, edges, color=color, lw=1.8, label=label, zorder=4)
    ax.stairs(np.maximum(dens - err, 0), edges, baseline=dens + err,
              fill=True, color=color, alpha=0.22, lw=0, zorder=3)
    return c, dens, err


def draw_ratio(ax, bins, dens_m, err_m, dens_d, err_d, color):
    """Model/data ratio with propagated errors."""
    c = 0.5 * (bins[:-1] + bins[1:])
    mask = (dens_d > 0) & (dens_m > 0)
    ratio = np.where(mask, dens_m / np.maximum(dens_d, 1e-300), np.nan)
    rerr = ratio * np.sqrt(
        np.where(mask, (err_m / np.maximum(dens_m, 1e-300)) ** 2
                 + (err_d / np.maximum(dens_d, 1e-300)) ** 2, 0.0))
    ax.axhline(1.0, color="gray", lw=0.9, ls="--", zorder=1)
    ax.errorbar(c[mask], ratio[mask], yerr=rerr[mask], fmt="o", ms=2.8,
                lw=1.0, color=color, zorder=3)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5])


def corner_label(ax):
    if HAVE_MPLHEP:
        hep.label.exp_text("", LABEL, loc=0, ax=ax, fontsize=11)
    else:
        ax.text(0.0, 1.015, LABEL, transform=ax.transAxes,
                fontsize=11, ha="left", va="bottom")


# ------------------------------------------------------------------
# 1) correlation matrices (one figure each)
# ------------------------------------------------------------------
def corr_figure(C, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.set_xticks(range(0, d, 2))
    ax.set_yticks(range(0, d, 2))
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("correlation")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path.name} (+ .pdf)")


corr_figure(corr_nan_safe(Y_te), "MC data", OUT / "fig_corr_data.png")
corr_figure(corr_nan_safe(Y_gen), "QFAN (Born sampler, no calibration)",
            OUT / "fig_corr_qfan.png")
corr_figure(corr_nan_safe(Y_ini), "QFAN, untrained circuit (cert. C4)",
            OUT / "fig_corr_untrained.png")


# ------------------------------------------------------------------
# Hardware overlays are disabled. This script produces simulator versus MC
# figures only. The hardware execution path and its results are kept in
# _hardware_archive/ and are not loaded here.
Y_ibm = None
IBM_LABEL = None
C_IBM = "#c2255c"

if Y_ibm is not None:
    corr_figure(corr_nan_safe(Y_ibm), f"QFAN on {IBM_LABEL} (Born sampler)",
                OUT / "fig_corr_ibm.png")


# ------------------------------------------------------------------
# 2) per-pixel marginal PDFs -- ONE STANDALONE FIGURE PER PIXEL
#    variants: ""      data vs QFAN (simulator)
#              "_untr" data vs QFAN vs untrained circuit
#              "_ibm"  data vs IBM hardware
#              "_all"  data vs QFAN (simulator) vs IBM hardware
# ------------------------------------------------------------------
def marginal_pixel_figure(j, series, tag):
    """series: list of (samples, color, label) model curves; MC data drawn
    on top as points. Saved as vector PDF (and PNG preview) -- drop-in
    subfloat files for the paper."""
    fig = plt.figure(figsize=(4.4, 4.4), dpi=200)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.9, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    hi = max([Y_te[:, j].max()] + [Y[:, j].max() for Y, _, _ in series])
    if hi <= 1e-12:
        ax.text(0.5, 0.5, f"pixel {j} (dead)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        for a in (ax, axr):
            a.set_xticks([]); a.set_yticks([])
    else:
        bins = np.linspace(0.0, hi * 1.02, 33)
        curves = [draw_model_step(ax, Y[:, j], bins, c, lab)
                  for Y, c, lab in series]
        _, dens_d, err_d = draw_data_points(ax, Y_te[:, j], bins)
        w1 = wasserstein_distance(Y_te[:, j], series[0][0][:, j])
        ax.text(0.96, 0.94, f"pixel {j}\n$W_1$ = {w1:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.set_ylabel("density")
        ax.legend(loc="best", fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
        for (Y, c, lab), (_, dens_m, err_m) in zip(series, curves):
            draw_ratio(axr, bins, dens_m, err_m, dens_d, err_d, c)
        axr.set_xlabel("energy (norm.)")
        axr.set_ylabel("model / data", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_marginal_pixel_{j:04d}{tag}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


for j in range(d):
    marginal_pixel_figure(j, [(Y_gen, C_QFAN, "QFAN")], "")
    marginal_pixel_figure(j, [(Y_gen, C_QFAN, "QFAN"),
                              (Y_ini, C_UNTR, "QFAN untrained")], "_untr")
    if Y_ibm is not None:
        marginal_pixel_figure(j, [(Y_ibm, C_IBM, f"QFAN {IBM_LABEL}")],
                              "_ibm")
        marginal_pixel_figure(j, [(Y_gen, C_QFAN, "QFAN sim"),
                                  (Y_ibm, C_IBM, f"QFAN {IBM_LABEL}")],
                              "_all")
print(f"[SAVE] fig_marginal_pixel_XXXX[.pdf/.png] variants: '', _untr"
      + (", _ibm, _all" if Y_ibm is not None else ""))


# ------------------------------------------------------------------
# 3) total-energy spectrum with ratio panel (same variants)
# ------------------------------------------------------------------
def energy_figure(series, tag):
    E_t = Y_te.sum(axis=1)
    fig = plt.figure(figsize=(7.4, 6.4), dpi=200)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.0], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    Es = [Y.sum(axis=1) for Y, _, _ in series]
    lo = min([E_t.min()] + [E.min() for E in Es])
    hi = max([E_t.max()] + [E.max() for E in Es])
    bins = np.linspace(lo, hi * 1.01, 40)
    curves = [draw_model_step(ax, E, bins, c, lab)
              for E, (Y, c, lab) in zip(Es, series)]
    _, dens_d, err_d = draw_data_points(ax, E_t, bins)
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=10)
    lines = [f"$W_1$({lab}) = {wasserstein_distance(E_t, E):.4f}"
             for E, (Y, c, lab) in zip(Es, series)]
    lines.append(f"data: mean {E_t.mean():.3f}, std {E_t.std():.3f}")
    ax.text(0.03, 0.60, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top")
    plt.setp(ax.get_xticklabels(), visible=False)
    corner_label(ax)
    for (E, (Y, c, lab)), (_, dens_m, err_m) in zip(zip(Es, series), curves):
        draw_ratio(axr, bins, dens_m, err_m, dens_d, err_d, c)
    axr.set_xlabel(r"total energy  $\sum_j y_j$  (norm.)")
    axr.set_ylabel("model / data", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_energy_sum{tag}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] fig_energy_sum{tag}.png/.pdf")


energy_figure([(Y_gen, C_QFAN, "QFAN"), (Y_ini, C_UNTR, "QFAN untrained")],
              "")
if Y_ibm is not None:
    energy_figure([(Y_ibm, C_IBM, f"QFAN {IBM_LABEL}")], "_ibm")
    energy_figure([(Y_gen, C_QFAN, "QFAN sim"),
                   (Y_ibm, C_IBM, f"QFAN {IBM_LABEL}")], "_all")

print(f"\nmplhep: {'ACTIVE (hep.style.ROOT)' if HAVE_MPLHEP else 'not installed -- built-in ROOT-style fallback used; pip install mplhep for the genuine style'}")
