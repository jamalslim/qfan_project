#!/usr/bin/env python3
"""
Polar shower plots, CaloChallenge style: GEANT4 and QFAN side by side.

    python ds1/plot_polar.py outputs/model_<tag>.npz --data data/ds1_53.npy
    python ds1/plot_polar.py outputs/model_<tag>.npz --data data/ds1_53.npy \\
           --binning binning_dataset_1_photons.xml

One column per calorimeter layer, drawn as a disc: radial bins are rings,
angular bins are sectors. Rows are GEANT4, QFAN, and their ratio. Each layer
column shares one logarithmic colour scale between GEANT4 and QFAN so the two
can be compared by eye.

Writes to plots_<tag>/:
    fig_polar_average.pdf    mean energy per voxel over all showers
    fig_polar_single.pdf     a few individual showers from each

RADIAL SCALE

The DS1 radial bins have variable widths. With --binning pointing at the
official binning_dataset_1_photons.xml from the CaloChallenge repository, the
rings are drawn at their true radii. The file is checked against the known
geometry (radial and angular bins per layer) before it is used; if it cannot be
read or does not match, the script says so and falls back to equal-width rings,
labelled as such. Without --binning the rings are equal-width.

WHAT THE PLOTS CAN AND CANNOT SHOW

QFAN models each layer summed over angle, so its showers are uniform around
each ring. The GEANT4 row is prepared the same way, collapsed over angle and
spread back evenly, so the average plots compare like with like. Individual
GEANT4 showers do have angular structure; that is exactly what the angle
collapse gives up, and single-shower plots will show both rows as smooth rings.
"""
import argparse
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve()
PROJECT_ROOT = next(c for c in [ROOT.parent] + list(ROOT.parents)
                    if (c / "src" / "qfan").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from export_calochallenge import LAYERS, to_voxels                # noqa: E402

LAYER_NAMES = ["layer 0", "layer 1", "layer 2", "layer 3", "layer 12"]


def radial_edges_from_xml(path):
    """Radial edges per layer from the official binning file, or None.

    Looks for elements carrying an r_edges attribute, in document order, and
    accepts them only if every layer's number of radial and angular bins
    matches the known photon geometry."""
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        print(f"  [binning] could not read {path}: {e}")
        return None
    layers = [el for el in root.iter() if "r_edges" in el.attrib]
    got = []
    for el in layers:
        try:
            edges = [float(x) for x in el.attrib["r_edges"].replace(" ", "").split(",") if x]
            n_a = int(el.attrib.get("n_bin_alpha", 1))
        except ValueError:
            return None
        got.append((n_a, np.asarray(edges)))
    shape_got = [(a, len(e) - 1) for a, e in got]
    if shape_got != LAYERS:
        print(f"  [binning] {path} does not match the photon geometry "
              f"(found {shape_got}, expected {LAYERS}); using equal-width rings")
        return None
    print(f"  [binning] true radial edges read from {path}")
    return [e for _, e in got]


def layer_blocks(voxels):
    """(N, 368) -> list of (N, n_alpha, n_r), one per layer."""
    out, c = [], 0
    for n_a, n_r in LAYERS:
        out.append(voxels[:, c:c + n_a * n_r].reshape(-1, n_a, n_r))
        c += n_a * n_r
    return out


def draw(fig, gs_row, blocks_mean, r_edges, vmins, vmaxs, cmap, norm_kind,
         row_label, cbar=True, title_row=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    axes = []
    for li, ((n_a, n_r), vals) in enumerate(zip(LAYERS, blocks_mean)):
        ax = fig.add_subplot(gs_row[li], projection="polar")
        # Draw each angular bin as several thin wedges. A single wedge spanning
        # the full circle has coinciding corners and renders as nothing, and
        # coarse wedges give polygons rather than rings.
        rep = int(np.ceil(72 / n_a))
        th = np.linspace(0, 2 * np.pi, n_a * rep + 1)
        vals = np.repeat(vals, rep, axis=0)
        re = r_edges[li]
        if norm_kind == "log":
            v = np.where(vals > 0, vals, np.nan)
            nrm = LogNorm(vmin=vmins[li], vmax=vmaxs[li])
        else:
            v = vals
            nrm = TwoSlopeNorm(vmin=vmins[li], vcenter=1.0, vmax=vmaxs[li])
        m = ax.pcolormesh(th, re, v.T, cmap=cmap, norm=nrm, shading="flat")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(0, re[-1])
        ax.spines["polar"].set_visible(False)
        if title_row:
            ax.set_title(LAYER_NAMES[li], fontsize=12, pad=12)
        if row_label and li == 0:
            ax.set_ylabel(row_label, labelpad=28, fontsize=11)
            ax.yaxis.set_label_position("left")
        if cbar:
            cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.06, shrink=0.75)
            cb.ax.tick_params(labelsize=7)
        axes.append(ax)
    return axes


def draw_one(vals, n_a, r_edge, vmin, vmax, title, path, label):
    """One polar plot of one layer, written to path (.pdf and .png)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rep = int(np.ceil(72 / n_a))
    th = np.linspace(0, 2 * np.pi, n_a * rep + 1)
    v = np.repeat(vals, rep, axis=0)
    v = np.where(v > 0, v, np.nan)
    fig = plt.figure(figsize=(3.6, 3.4))
    ax = fig.add_subplot(111, projection="polar")
    m = ax.pcolormesh(th, r_edge, v.T, cmap="viridis",
                      norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0, r_edge[-1])
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, fontsize=10, pad=10)
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.08, shrink=0.8)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def separate_figures(sources, edges, out):
    """Every polar plot as its own figure: the average shower and one example
    shower, per layer, for each source; one colour scale per layer and kind."""
    folder = out / "polar_separate"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for kind in ("average", "single"):
        blocks = {}
        for key, (name, vox) in sources.items():
            if kind == "average":
                blocks[key] = [b.mean(0) for b in layer_blocks(vox)]
            else:
                i = rng.integers(len(vox))
                blocks[key] = [b[0] for b in layer_blocks(vox[i:i + 1])]
        for li, (n_a, n_r) in enumerate(LAYERS):
            pos = np.concatenate([blocks[k][li][blocks[k][li] > 0] for k in sources])
            vmax = pos.max() if pos.size else 1.0
            vmin = max(pos.min(), vmax * 1e-4) if pos.size else 1e-6
            label = ("mean energy per voxel [MeV]" if kind == "average"
                     else "energy per voxel [MeV]")
            for key, (name, _) in sources.items():
                title = f"{name}, {LAYER_NAMES[li]}" + (" (average)" if kind == "average"
                                                          else " (one shower)")
                draw_one(blocks[key][li], n_a, edges[li], vmin, vmax, title,
                         folder / f"{kind}_{LAYER_NAMES[li].replace(' ', '')}_{key}", label)
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data", required=True, help="the .npy the model was trained on")
    ap.add_argument("--binning", default=None,
                    help="official binning_dataset_1_photons.xml, for true radii")
    ap.add_argument("--n-single", type=int, default=3)
    ap.add_argument("--separate", action="store_true",
                    help="also write every polar plot as its own figure (GEANT4, trained "
                         "and untrained circuit; average and one shower; each layer) to "
                         "plots_<tag>/polar_separate/")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    side = pathlib.Path(a.data).with_suffix(".json")
    if not side.exists():
        sys.exit(f"missing {side}; it is written by prepare_ds1.py")
    info = json.loads(side.read_text())
    r = np.load(a.model, allow_pickle=True)
    tag = pathlib.Path(a.model).stem.replace("model_", "")
    out = PROJECT_ROOT / f"plots_{tag}"
    out.mkdir(exist_ok=True)

    # say on the figure what is drawn: with the angular bins summed, every ring
    # is spread evenly over angle, for GEANT4 and QFAN alike
    what = ("53 rings: energy summed over angle and drawn evenly around each ring"
            if info.get("collapse") == "angle" else "all 368 voxels")
    geant = to_voxels(r["Y_te"], info)
    qfan = to_voxels(r["Y_gen"], info)

    edges = radial_edges_from_xml(a.binning) if a.binning else None
    true_radii = edges is not None
    if edges is None:
        edges = [np.arange(n_r + 1, dtype=float) for _, n_r in LAYERS]
    scale_note = ("true radial bin edges" if true_radii
                  else "equal-width rings (pass --binning for true radii)")

    # ---- average shower --------------------------------------------------
    Bg = [b.mean(0) for b in layer_blocks(geant)]
    Bq = [b.mean(0) for b in layer_blocks(qfan)]
    vmin, vmax = [], []
    for g, q in zip(Bg, Bq):
        pos = np.r_[g[g > 0], q[q > 0]]
        vmin.append(max(pos.min(), pos.max() * 1e-4) if pos.size else 1e-6)
        vmax.append(pos.max() if pos.size else 1.0)
    ratio = [np.where(g > 0, q / np.where(g > 0, g, 1), np.nan) for g, q in zip(Bg, Bq)]
    lo = [float(np.clip(np.nanmin(x), 0.2, 0.95)) for x in ratio]
    hi = [float(np.clip(np.nanmax(x), 1.05, 5.0)) for x in ratio]

    fig = plt.figure(figsize=(3.3 * len(LAYERS), 10.0))
    gs = GridSpec(3, len(LAYERS), figure=fig, hspace=0.30, wspace=0.45)
    draw(fig, [gs[0, i] for i in range(len(LAYERS))], Bg, edges, vmin, vmax,
         "viridis", "log", "GEANT4", title_row=True)
    draw(fig, [gs[1, i] for i in range(len(LAYERS))], Bq, edges, vmin, vmax,
         "viridis", "log", "QFAN")
    draw(fig, [gs[2, i] for i in range(len(LAYERS))], ratio, edges, lo, hi,
         "RdBu_r", "ratio", "QFAN / GEANT4")
    fig.suptitle(f"Average shower, mean energy per voxel [MeV]   ({scale_note})\n{what}",
                 fontsize=11, y=1.01)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_polar_average.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- single showers --------------------------------------------------
    n = min(a.n_single, len(geant), len(qfan))
    rng = np.random.default_rng(0)
    ig = rng.choice(len(geant), n, replace=False)
    iq = rng.choice(len(qfan), n, replace=False)
    fig = plt.figure(figsize=(3.0 * len(LAYERS), 3.1 * 2 * n))
    gs = GridSpec(2 * n, len(LAYERS), figure=fig, hspace=0.35, wspace=0.35)
    for k in range(n):
        sg = [b[0] for b in layer_blocks(geant[ig[k]:ig[k] + 1])]
        sq = [b[0] for b in layer_blocks(qfan[iq[k]:iq[k] + 1])]
        mn, mx = [], []
        for g, q in zip(sg, sq):
            pos = np.r_[g[g > 0], q[q > 0]]
            mn.append(max(pos.min(), pos.max() * 1e-4) if pos.size else 1e-6)
            mx.append(pos.max() if pos.size else 1.0)
        draw(fig, [gs[2 * k, i] for i in range(len(LAYERS))], sg, edges, mn, mx,
             "viridis", "log", f"GEANT4 #{k + 1}", title_row=(k == 0))
        draw(fig, [gs[2 * k + 1, i] for i in range(len(LAYERS))], sq, edges, mn, mx,
             "viridis", "log", f"QFAN #{k + 1}")
    fig.suptitle(what, fontsize=11, y=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_polar_single.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if a.separate:
        sources = {"geant4": ("GEANT4", geant), "trained": ("QFAN, trained", qfan)}
        if "Y_ini" in r.files:
            sources["untrained"] = ("untrained circuit", to_voxels(r["Y_ini"], info))
        folder = separate_figures(sources, edges, out)
        print(f"  wrote separate polar plots to {folder}")
    print(f"  wrote {out}/fig_polar_average.pdf and fig_polar_single.pdf")
    print(f"  radial scale: {scale_note}")


if __name__ == "__main__":
    main()
