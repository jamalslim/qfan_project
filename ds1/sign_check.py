#!/usr/bin/env python3
"""
Is the model's correlation structure a sign-flipped copy of the truth?

    python ds1/sign_check.py outputs/model_<tag>.npz --layers 8,16,19,5,5

For every pair of layers it compares the model's correlations with the truth
two ways, as they are and with the truth's sign reversed:

    as is     mean |C_model - C_truth|
    flipped   mean |C_model + C_truth|

A clean sign flip shows up as `flipped` far smaller than `as is`: the model has
the right magnitudes and the wrong signs. If both are similar, the model simply
has different structure there, which is a different failure with a different
fix. The last column is the strength of the true correlation in that region;
a region the truth barely correlates cannot show a meaningful flip.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve()
PROJECT_ROOT = next(c for c in [ROOT.parent] + list(ROOT.parents)
                    if (c / "src" / "qfan").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from qfan.utils import corr_nan_safe                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", default="8,16,19,5,5",
                    help="pixels per layer, in image order (DS1 photons, "
                         "collapsed over angle, is the default)")
    a = ap.parse_args()

    r = np.load(a.model, allow_pickle=True)
    Ct = corr_nan_safe(r["Y_te"])
    Cm = corr_nan_safe(r["Y_gen"])
    widths = [int(w) for w in a.layers.split(",")]
    if sum(widths) != Ct.shape[0]:
        sys.exit(f"layer widths sum to {sum(widths)}, data has {Ct.shape[0]}")
    edges = np.cumsum([0] + widths)
    names = [f"L{i}" for i in range(len(widths))]

    print(f"\n  {'pair':>9} {'as is':>8} {'flipped':>8} {'verdict':>22} "
          f"{'truth strength':>15}")
    for i in range(len(widths)):
        for j in range(i, len(widths)):
            a0, a1, b0, b1 = edges[i], edges[i + 1], edges[j], edges[j + 1]
            T = Ct[a0:a1, b0:b1]
            M = Cm[a0:a1, b0:b1]
            if i == j:                                    # drop the diagonal
                mask = ~np.eye(a1 - a0, dtype=bool)
                T, M = T[mask], M[mask]
            as_is = float(np.abs(M - T).mean())
            flip = float(np.abs(M + T).mean())
            strength = float(np.abs(T).mean())
            if strength < 0.1:
                verdict = "truth too weak to say"
            elif flip < 0.6 * as_is:
                verdict = "SIGN FLIPPED"
            elif as_is < 0.6 * flip:
                verdict = "sign right"
            else:
                verdict = "wrong, but not a flip"
            print(f"  {names[i] + '-' + names[j]:>9} {as_is:8.3f} {flip:8.3f} "
                  f"{verdict:>22} {strength:15.3f}")
    print("  A single sign-reversed layer flips every pair it takes part in "
          "while its own\n  internal pair stays right, so the flipped pairs "
          "point to the layer at fault.\n")


if __name__ == "__main__":
    main()
