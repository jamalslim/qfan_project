#!/usr/bin/env python3
"""
Generate showers from a trained QFAN model.

    python ds1/generate.py outputs/model_<tag>.npz --data data/<prepared>.npy

The trained circuit samples showers block by block, each block conditioned on
the ones before it, from Born measurement records, decoded by the closed-form
ridge regressions fitted on training showers. The only thing applied afterwards
is the exact inverse of the preprocessing (for data prepared with --fractions:
the fractions are normalised and multiplied by the generated total). Nothing
is fitted to generated output.

Written to outputs/model_<tag>_showers.npz, all in image units:
  Y_gen   showers from the trained circuit
  Y_ini   showers from the same circuit at its untrained initialisation, with
          the same decoders fitted the same way: the control for training
  Y_null  test showers with every pixel shuffled independently in the prepared
          space and sent through the same inverse: the control for correlations
  Y_tr, Y_te   training and test showers
eval_geometry.py, plot_polar.py, sign_check.py and aggregate.py read this file.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(HERE.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="outputs/model_<tag>.npz from scripts/train.py")
    ap.add_argument("--data", required=True, help="the prepared .npy the model was trained on")
    ap.add_argument("--n", type=int, default=None, help="showers to generate (default: as many as test showers)")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--fit-k", action="store_true",
                    help="choose each block's number of measurement records after "
                         "training: the count minimising the training loss on training "
                         "showers, circuit fixed. Each circuit, trained and untrained, "
                         "gets its own. Lets models trained with fixed k use per-block "
                         "budgets without retraining")
    ap.add_argument("--out", default=None,
                    help="output file (default: outputs/model_<tag>_showers.npz); use it to keep "
                         "several generations of the same model side by side")
    a = ap.parse_args()

    import train as T
    from prepare_ds1 import from_fractions
    from qfan.born import StatevectorBornBank, fit_all_blocks_born, sample_progressive_born

    src = pathlib.Path(a.model)
    r = np.load(src, allow_pickle=True)
    meta = r["meta"][0]
    depth = int(meta["depth"])
    n_q = r["theta_final"].size // (2 * depth)
    angle_dim = r["A_final"].shape[0]
    K = np.asarray(meta.get("k_gen", 64))
    K = int(K) if K.size == 1 else K.astype(int)          # records per block

    # rebuild the problem exactly as the trainer built it
    if meta.get("seed") is not None:
        T.SEED_OVERRIDE = int(meta["seed"])
    tag = "gen_" + src.stem
    data = str(pathlib.Path(a.data).expanduser().resolve())
    T.D_PIXELS = T.register_geometry(data, n_q, depth, angle_dim, tag)
    T.TAG = tag
    T.BLOCK_SIZE = int(meta.get("block_size", max(int(n) for _, n in meta["blocks"])))
    cfg, Y_tr, Y_te, d, blocks, sk, cache, spec, bank = T.build_problem(tag)
    if [tuple(b) for b in meta["blocks"]] != [tuple(b) for b in blocks]:
        sys.exit("block layout differs from the one the model was trained with")
    if Y_tr.shape != r["Y_tr"].shape or not np.allclose(Y_tr, r["Y_tr"]):
        sys.exit("this data file does not reproduce the model's training split; "
                 "use the exact .npy the model was trained on")

    bank.set_theta(r["theta_final"])
    bank.A = r["A_final"].copy()
    bank.b = r["b_final"].copy()
    bank_ini = StatevectorBornBank(cfg.sketch.sketch_dim, spec, seed=cfg.data.seed + 13)
    rho = np.asarray(r["rho_final"], float)
    n = a.n or len(Y_te)
    rng = np.random.default_rng(a.seed)

    if a.fit_k:
        from scipy.optimize import minimize_scalar
        from qfan.born_factorized import cf_mmd_loss_and_grads_factorized as cf_loss
        _, _, _, _, Omega_pb, Omega_c_pb = T.setup_train(bank, cache, Y_tr, blocks, cfg.data.seed)
        gamma = float(meta.get("gamma_risk", 10.0))
        fit_rows = np.random.default_rng(0).choice(len(Y_tr), min(1024, len(Y_tr)), replace=False)

    def budgets(bk):
        """Records per block minimising the training loss, circuit fixed."""
        ks = []
        th = bk.get_theta()
        for bi, (s0, bsz) in enumerate(blocks):
            S = cache[bi, fit_rows].astype(np.float64)
            Yb = Y_tr[fit_rows, s0:s0 + bsz]
            Xc = Y_tr[fit_rows, :s0] if Omega_c_pb[bi] is not None else None
            f = lambda lk: cf_loss(bk, S, Yb, th, Omega_pb[bi], cfg.bank.ridge_alpha,
                                   k_shots=float(np.exp(lk)), gamma_risk=gamma,
                                   compute_grad=False, X_cond=Xc,
                                   Omega_cond=Omega_c_pb[bi], rho=float(rho[bi]))[0]
            res = minimize_scalar(f, bounds=(np.log(2.0), np.log(1024.0)),
                                  method="bounded", options=dict(xatol=0.02))
            ks.append(int(max(1, round(float(np.exp(res.x))))))
        return np.array(ks)

    used_k = {}

    def sample(bk, label):
        kb = budgets(bk) if a.fit_k else K
        used_k[label] = np.asarray(kb)
        if a.fit_k:
            print(f"[GENERATE] records per block ({label}): {kb.tolist()}", flush=True)
        models = fit_all_blocks_born(bk, cache, Y_tr, blocks, cfg.bank.ridge_alpha,
                                     noise_aware=True, k_shots=kb, intercept=True)
        return sample_progressive_born(bk, models, d, blocks, sk, n, rng,
                                       k=kb, record_share=rho)

    X_gen, X_ini = sample(bank, "trained"), sample(bank_ini, "untrained")
    X_null = np.column_stack([rng.permutation(Y_te[:, j]) for j in range(d)])

    info = json.loads(pathlib.Path(data).with_suffix(".json").read_text())
    if info.get("represent") == "fractions":
        inv = lambda X: from_fractions(X, info)
    else:
        inv = lambda X: np.asarray(X, np.float64)
    out = dict(Y_gen=inv(X_gen), Y_ini=inv(X_ini), Y_null=inv(X_null),
               Y_tr=inv(Y_tr), Y_te=inv(Y_te))
    for k in ("theta_final", "A_final", "b_final", "rho_final"):
        out[k] = r[k]
    width = out["Y_te"].shape[1]
    img_blocks = [(s, min(2, width - s)) for s in range(0, width, 2)]
    out["meta"] = np.array([dict(
        layer_widths=info.get("layer_widths"), blocks=img_blocks, block_size=2,
        depth=depth, k_gen=K, data_path=data, model=str(src),
        represent=info.get("represent", "image"), seed=meta.get("seed"),
        k_trained=used_k["trained"].tolist(), k_untrained=used_k["untrained"].tolist(),
        generation="trained circuit, closed-form ridge decoders fitted on training "
                   "showers, inverse preprocessing only")], dtype=object)
    dest = pathlib.Path(a.out) if a.out else src.with_name(src.stem + "_showers.npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    from qfan.utils import corr_nan_safe
    from scipy.stats import rankdata
    Yte = out["Y_te"]
    off = ~np.eye(width, dtype=bool)
    sp = lambda Y: corr_nan_safe(np.apply_along_axis(rankdata, 0, Y))
    St = sp(Yte)
    e = lambda Y: np.abs(sp(Y) - St)[off].mean()
    mad = lambda x: np.median(np.abs(x - np.median(x)))
    print(f"[GENERATE] wrote {dest}  ({n} showers)")
    print(f"           rank-correlation error: QFAN {e(out['Y_gen']):.4f}, untrained "
          f"circuit {e(out['Y_ini']):.4f}, no correlations {e(out['Y_null']):.4f}")
    print(f"           total-energy spread / truth (robust): QFAN "
          f"{mad(out['Y_gen'].sum(1)) / mad(Yte.sum(1)):.2f}")


if __name__ == "__main__":
    main()
