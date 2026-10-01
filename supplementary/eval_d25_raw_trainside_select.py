"""
Train-side re-selection of the RAW (calibration-free) pipeline config.

The raw headline's configuration (the bank +
quadratic features + k=32) was selected "head-to-head" over 8 configs on
the reported metrics, i.e. plausibly on TEST -- a mild but real selection
bias, and a violation of the rule that k_gen (and by extension
every generation-time choice) is calibrated on a TRAIN-side monitor.

This script redoes the selection cleanly:
  * grid: bank in { (angle_dim=8), QFAN (angle_dim=16)}
          x features in {linear, quadratic}  x  k in {32, 64}
  * selection metric: free-running off-diag correlation MAE against the
    TRAIN correlations (generation seed fixed, n = 1200); test never
    consulted during selection;
  * the winner's TEST row is then computed exactly once and reported as
    the certified raw headline.

Decoders are the closed-form noise-aware fits on train (both feature
sets); rho_b sharing follows each bank's trained rho_final, mirroring
evaluate.py's generation loop verbatim.
"""
import pathlib
import sys

import numpy as np
from scipy.stats import wasserstein_distance

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qfan import correlation_error_summary, corr_nan_safe      # noqa: E402
from qfan.born import (noise_aware_ridge_fit, fit_all_blocks_born,  # noqa: E402
                       sample_progressive_born)
from train import build_problem, spearman_corr   # noqa: E402
from quad_features import ExtFeatures                  # noqa: E402

BANKS = {
    "": ("qfan_d25_v41_results.npz", 8),
    "QFAN": ("model_d25.npz", 16),
}
KS = (32, 64)


def load_bank(tag):
    """Fresh problem + bank with the tagged checkpoint loaded."""
    from qfan.born import BornBankSpec, StatevectorBornBank
    (cfg, Y_tr, Y_te, d, blocks, sketcher, cache, spec,
     bank) = build_problem()
    fname, angle_dim = BANKS[tag]
    if angle_dim != 16:
        spec2 = BornBankSpec(n_qubits=cfg.bank.n_qubits,
                             depth=cfg.bank.q_depth,
                             angle_dim=angle_dim, include_y=True,
                             gen_shots_k=64)
        bank = StatevectorBornBank(cfg.sketch.sketch_dim, spec2,
                                   seed=cfg.data.seed + 13)
    r = np.load(PROJECT_ROOT / "outputs" / fname, allow_pickle=True)
    bank.set_theta(r["theta_final"])
    bank.A = r["A_final"].copy()
    bank.b = r["b_final"].copy()
    rho = r["rho_final"].copy()
    return cfg, Y_tr, Y_te, d, blocks, sketcher, cache, bank, rho


def generate_quad(bank, ext, models, blocks, sketcher, rho, d, n, k, rng,
                  batch=1200):
    """Verbatim generation loop of evaluate.py (quad features)."""
    out = np.zeros((n, d))
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        Sraw, cur = sketcher.init_state(hi - lo)
        for bi, (s, bsz) in enumerate(blocks):
            P = bank.setting_probs(sketcher.mixed(Sraw, cur))
            rs = float(rho[bi])
            k_sh = int(round(rs * k)) if (bsz > 1 and rs < 1.0) else k
            k_ex = k - k_sh if bsz > 1 else 0
            rec = bank.sample_records(P, k_sh + bsz * k_ex if k_ex else k,
                                      rng)
            rec_T = [ext.Text[rec[g]] for g in range(bank.G)]
            cols = []
            for j in range(bsz):
                sl = (np.arange(k) if k_ex == 0 else
                      np.concatenate([np.arange(k_sh),
                                      k_sh + j * k_ex + np.arange(k_ex)]))
                cols.append(ext.record_features(rec_T, sl)
                            @ models[bi][:, j])
            Yb = np.maximum(np.stack(cols, 1), 0.0)
            out[lo:hi, s:s + bsz] = Yb
            cur = sketcher.update_inplace(Sraw, cur, Yb)
    return out


def row(Y_ref, Y, blocks, d):
    w1 = float(np.mean([wasserstein_distance(Y_ref[:, j], Y[:, j])
                        for j in range(d)]))
    cs = correlation_error_summary(corr_nan_safe(Y_ref), corr_nan_safe(Y),
                                   blocks)
    sp = correlation_error_summary(spearman_corr(Y_ref), spearman_corr(Y),
                                   blocks)
    return dict(w1=w1, off=cs["corr_mae_offdiag"],
                win=cs["corr_mae_within"], crs=cs["corr_mae_cross"],
                sp=sp["corr_mae_offdiag"])


def main():
    results = {}
    gens = {}
    for tag in BANKS:
        (cfg, Y_tr, Y_te, d, blocks, sketcher, cache, bank,
         rho) = load_bank(tag)
        ridge_alpha = cfg.bank.ridge_alpha
        for feat in ("linear", "quad"):
            for k in KS:
                rng = np.random.default_rng(cfg.data.seed)
                if feat == "linear":
                    models = fit_all_blocks_born(
                        bank, cache, Y_tr, blocks, ridge_alpha,
                        noise_aware=True, k_shots=k)
                    Y = sample_progressive_born(
                        bank, models, d, blocks, sketcher, len(Y_te), rng,
                        k=k, record_share=rho)
                else:
                    ext = ExtFeatures(bank)
                    models = []
                    for bi, (s, bsz) in enumerate(blocks):
                        P = bank.setting_probs(
                            cache[bi].astype(np.float64))
                        W, _ = noise_aware_ridge_fit(
                            ext.expectation(P),
                            Y_tr[:, s:s + bsz].astype(np.float64),
                            ext.sigma_bar(P), ridge_alpha, k=k)
                        models.append(W)
                    Y = generate_quad(bank, ext, models, blocks, sketcher,
                                      rho, d, len(Y_te), k, rng)
                key = (tag, feat, k)
                results[key] = row(Y_tr, Y, blocks, d)   # TRAIN reference
                gens[key] = Y
                print(f"[train-side] {tag} {feat:6s} k={k:2d}: "
                      f"off={results[key]['off']:.4f}  "
                      f"W1={results[key]['w1']:.4f}", flush=True)

    # Selection rule (pre-specified): primary = off-diag; configs within
    # TIE_TOL of the best are a statistical tie (measured MC generation
    # noise at n=1200 is sigma ~ 0.003-0.005 per seed); ties are broken by
    # W1 (marginal fidelity). Report the tie set alongside the winner.
    TIE_TOL = 0.005
    best_off = min(v["off"] for v in results.values())
    tie_set = [kk for kk, v in results.items()
               if v["off"] <= best_off + TIE_TOL]
    winner = min(tie_set, key=lambda kk: results[kk]["w1"])
    print("-" * 78)
    if len(tie_set) > 1:
        print(f"TIE on primary metric (off within {TIE_TOL} of best "
              f"{best_off:.4f}): {tie_set}")
    print(f"TRAIN-SIDE WINNER: bank={winner[0]} features={winner[1]} "
          f"k={winner[2]}  (train off={results[winner]['off']:.4f}, "
          f"W1 tie-break={results[winner]['w1']:.4f})")

    # test touched once, for the winner only
    (_cfg, _Ytr, Y_te, d, blocks, _sk, _cache, _bank,
     _rho) = load_bank(winner[0])
    m = row(Y_te, gens[winner], blocks, d)
    print(f"CERTIFIED RAW HEADLINE (test, selected train-side): "
          f"W1={m['w1']:.5f}  off={m['off']:.4f}  win={m['win']:.4f}  "
          f"crs={m['crs']:.4f}  Sp off={m['sp']:.4f}")
    np.savez_compressed(
        PROJECT_ROOT / "outputs/qfan_d25_raw_trainside_selection.npz",
        meta=np.array([dict(train_side={str(k): v for k, v in
                                        results.items()},
                            winner=str(winner), test_row=m)],
                      dtype=object),
        Y_head=gens[winner])
    print("[SAVE] outputs/qfan_d25_raw_trainside_selection.npz")


if __name__ == "__main__":
    main()
