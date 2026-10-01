#!/usr/bin/env python3
"""
Is the gap between the training and validation curves real?

    python ds1/diagnose.py outputs/model_<tag>.npz --data data/<prepared>.npy

The trainer reports the loss on batches of 256 training showers, with each
block's decoder refitted on the batch being scored, and the validation loss on
512 held-out showers. This scores TRAINING showers with the validation
estimator, at several batch sizes and ridge penalties. If training showers
score like validation showers, there is no generalisation gap: the model has
reached the capacity of its circuit, and the lower training curve only reflects
the smaller batch. If training showers score much lower, the decoders fit
their batch better than new showers, and more training showers or a larger
batch are the remedies.
"""
import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    import train as T
    r = np.load(a.model, allow_pickle=True)
    meta = r["meta"][0]
    depth = int(meta["depth"])
    n_q = r["theta_final"].size // (2 * depth)
    if meta.get("seed") is not None:
        T.SEED_OVERRIDE = int(meta["seed"])
    tag = "diag_" + pathlib.Path(a.model).stem
    T.BLOCK_SIZE = int(meta.get("block_size", 2))
    T.D_PIXELS = T.register_geometry(str(pathlib.Path(a.data).resolve()), n_q, depth,
                                     r["A_final"].shape[0], tag)
    T.TAG = tag
    cfg, Y_tr, Y_te, d, blocks, sk, cache, spec, bank = T.build_problem(tag)
    if not np.allclose(Y_tr, r["Y_tr"]):
        sys.exit("this data file does not reproduce the model's training split")
    prng, val_idx, train_pool, _, Om, Omc = T.setup_train(bank, cache, Y_tr, blocks, cfg.data.seed)
    bank.A, bank.b = r["A_final"].copy(), r["b_final"].copy()
    th = r["theta_final"]
    rho = np.clip(np.asarray(r["rho_final"], float), 1e-6, 1 - 1e-6)
    logit = np.log(rho / (1 - rho))
    a0 = cfg.bank.ridge_alpha

    def loss(idx, alpha):
        return T.total_val(bank, th, blocks, cache, idx, Y_tr, Om, Omc, alpha, logit)

    rng = np.random.default_rng(0)
    sizes = (128, 256, 512)
    subsets = {n: [rng.choice(train_pool, n, replace=False) for _ in range(3)] for n in sizes}
    print(f"loss with the validation estimator (ridge alpha {a0:g} in training)")
    print(f"{'alpha':>8}{'validation 512':>16}" + "".join(f"{'train ' + str(n):>12}" for n in sizes))
    for f in (1, 10, 100):
        al = a0 * f
        row = [loss(val_idx, al)] + [np.mean([loss(i, al) for i in subsets[n]]) for n in sizes]
        print(f"{al:>8.3g}{row[0]:>16.3f}" + "".join(f"{v:>12.3f}" for v in row[1:]))
    v, t = loss(val_idx, a0), np.mean([loss(i, a0) for i in subsets[512]])
    if t < 0.8 * v:
        print("\ntraining showers score much lower than validation showers at the same batch "
              "size: the decoders overfit their batch. Remedies: more training showers "
              "(all 10,000 at this energy, or both DS1 photon files) or a larger batch.")
    else:
        print("\ntraining and validation showers score alike at the same batch size: no "
              "generalisation gap. The plateau is the circuit's capacity; the training "
              "curve is lower only because its batches are smaller.")


if __name__ == "__main__":
    main()
