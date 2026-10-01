#!/usr/bin/env python3
"""
Error bars across seeds: one table row per configuration.

    python ds1/aggregate.py outputs/model_ds1_53_nq4_b2_s*_showers.npz \\
           --name "QFAN, 53 rings" --table outputs/results.md

Each file is one seed, written by generate.py: its own training run and
train/test split. For every file the generated showers (Y_gen) and the untrained
circuit (Y_ini) are scored against that seed's test showers (Y_te), next to two
references on the same showers: the statistical floor (as many training showers
as test showers, drawn at random) and the no-correlation model (Y_null: the test
showers with every pixel shuffled independently, sent through the same inverse
preprocessing as the generated showers). Across files each metric is reported
as mean +/- standard deviation. With a single file the spread comes from
bootstrap resamples of the test and generated showers instead, which omits
training variability; the row says so.

Because the no-correlation model goes through the same inverse, it shows how
much of each metric that step produces by itself. Exact energy conservation, for
instance, produces much of the layer 1-2 anticorrelation for any model. The full
correlation matrix, against the untrained circuit and the no-correlation model,
is the fair test.

Metrics
  W1 dense / sparse   per-pixel Wasserstein distance over the truth's standard
                      deviation; sparse = pixels zero in over 5% of test showers
                      or with a coefficient of variation above 1
  zeros               fraction of exact zeros on the sparse pixels
  Pearson, Spearman   mean absolute error of the off-diagonal correlation matrix
  captured            share of the achievable Spearman improvement,
                      (null - model) / (null - floor)
  E spread            robust spread (median absolute deviation) of the total
                      energy over the truth's
  L1-L2               correlation between the energies of layers 1 and 2
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy.stats import rankdata, wasserstein_distance

ROOT = pathlib.Path(__file__).resolve()
PROJECT_ROOT = next(c for c in [ROOT.parent] + list(ROOT.parents)
                    if (c / "src" / "qfan").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from qfan.utils import corr_nan_safe                              # noqa: E402

KEYS = ["w1_dense", "w1_sparse", "zeros", "pearson", "spearman", "captured",
        "e_spread", "l12"]
HEAD = ["W1 dense", "W1 sparse", "zeros", "Pearson", "Spearman", "captured",
        "E spread", "L1-L2"]


def spearman_matrix(Y):
    return corr_nan_safe(np.apply_along_axis(rankdata, 0, Y))


def mad(x):
    return np.median(np.abs(x - np.median(x)))


def score(Y, ref, edges, sparse, off, Ct, St):
    w = np.array([wasserstein_distance(ref[:, j], Y[:, j]) / max(ref[:, j].std(), 1e-12)
                  for j in range(ref.shape[1])])
    L = np.stack([Y[:, a:b].sum(1) for a, b in zip(edges, edges[1:])], 1)
    return dict(w1_dense=w[~sparse].mean(), w1_sparse=w[sparse].mean(),
                zeros=(Y[:, sparse] == 0).mean(),
                pearson=np.abs(corr_nan_safe(Y) - Ct)[off].mean(),
                spearman=np.abs(spearman_matrix(Y) - St)[off].mean(),
                e_spread=mad(Y.sum(1)) / max(mad(ref.sum(1)), 1e-12),
                l12=np.corrcoef(L[:, 1], L[:, 2])[0, 1])


def evaluate(path, ref_rows=None, rng=None):
    r = np.load(path, allow_pickle=True)
    meta = r["meta"][0]
    Yte = r["Y_te"] if ref_rows is None else r["Y_te"][ref_rows]
    Ytr = r["Y_tr"]
    d = Yte.shape[1]
    widths = meta.get("layer_widths") or ([8, 16, 19, 5, 5] if d == 53 else [8, 160, 190, 5, 5])
    edges = np.cumsum([0] + list(widths))
    zf = (Yte == 0).mean(0)
    cv = Yte.std(0) / np.maximum(Yte.mean(0), 1e-12)
    sparse = (zf > 0.05) | (cv > 1.0)
    off = ~np.eye(d, dtype=bool)
    Ct, St = corr_nan_safe(Yte), spearman_matrix(Yte)
    rng = rng or np.random.default_rng(0)
    floor = score(Ytr[rng.integers(0, len(Ytr), len(Yte))], Yte, edges, sparse, off, Ct, St)
    if "Y_null" in r.files:
        # written by generate.py: shuffled in the prepared space and sent through
        # the same inverse as the model's showers
        null_Y = r["Y_null"]
        if ref_rows is not None:
            null_Y = null_Y[rng.integers(0, len(null_Y), len(null_Y))]
    else:
        null_Y = np.column_stack([rng.permutation(Yte[:, k]) for k in range(d)])
    null = score(null_Y, Yte, edges, sparse, off, Ct, St)
    out = {}
    for key in ("Y_gen", "Y_ini"):
        if key in r.files:
            Yg = r[key]
            if ref_rows is not None:            # bootstrap the generated showers too
                Yg = Yg[rng.integers(0, len(Yg), len(Yg))]
            m = score(Yg, Yte, edges, sparse, off, Ct, St)
            m["captured"] = (null["spearman"] - m["spearman"]) / max(
                null["spearman"] - floor["spearman"], 1e-12)
            out[key] = m
    floor["captured"], null["captured"] = 1.0, 0.0
    out["floor"], out["null"] = floor, null
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one model file per seed")
    ap.add_argument("--name", required=True, help="label for this configuration")
    ap.add_argument("--table", default=None, help="append the rows to this markdown file")
    ap.add_argument("--boot", type=int, default=8, help="bootstraps when only one file")
    a = ap.parse_args()

    runs = []
    if len(a.files) == 1:
        n = len(np.load(a.files[0], allow_pickle=True)["Y_te"])
        for b in range(a.boot):
            rng = np.random.default_rng(1000 + b)
            runs.append(evaluate(a.files[0], rng.integers(0, n, n), rng))
        how = f"1 seed, {a.boot} bootstraps: no training variability"
    else:
        for i, f in enumerate(a.files):
            runs.append(evaluate(f, None, np.random.default_rng(1000 + i)))
        how = f"{len(a.files)} seeds"

    rows = []
    for key, label in (("Y_gen", a.name), ("Y_ini", a.name + ", untrained circuit"),
                       ("floor", "floor (training showers)"), ("null", "no correlations")):
        vals = [run[key] for run in runs if key in run]
        if not vals:
            continue
        cells = []
        for k in KEYS:
            v = np.array([x[k] for x in vals])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            cells.append(f"{100*v.mean():.1f}% ± {100*sd:.1f}" if k in ("zeros", "captured")
                         else f"{v.mean():.3f} ± {sd:.3f}")
        rows.append((label, cells))

    width = max(len(r[0]) for r in rows)
    print(f"\n{how}")
    print(" " * width + "".join(f"{h:>18}" for h in HEAD))
    for label, cells in rows:
        print(f"{label:<{width}}" + "".join(f"{c:>18}" for c in cells))
    if a.table:
        t = pathlib.Path(a.table)
        new = not t.exists()
        with t.open("a") as fh:
            if new:
                fh.write("| configuration | " + " | ".join(HEAD) + " |\n")
                fh.write("|---" * (len(HEAD) + 1) + "|\n")
            for label, cells in rows:
                fh.write(f"| {label} ({how}) | " + " | ".join(cells) + " |\n")
        print(f"\nappended to {t}")


if __name__ == "__main__":
    main()
