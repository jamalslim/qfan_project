#!/usr/bin/env python3
"""
Evaluate the trained d=25 model with no post-processing at all.

This is the "raw" pipeline. No correlation calibration, no monotone marginal
map, no copula step. What comes out is the untouched Born sampler, which is
what you want when the question is what the model itself produces rather than
what a post-hoc map can fix up.

It uses quadratic record features and k=32 generation shots, the best raw
configuration in a head-to-head over both banks, linear and quadratic
decoders, and k in {32, 64}. It reports

    W1 = 0.0141   off-diag = 0.1068   within = 0.0856   cross = 0.1077

against a marginal-fidelity floor of 0.0016 set by the finite test sample.

CAREFUL. These are NOT the d=25 numbers in the paper's Table I, which come
straight from training (native features, k=64) and read 0.0128 / 0.1282. Both
are legitimate QFAN results but they are different configurations, so the two
sets are not directly comparable.

Needs outputs/model_d25.npz, produced by

    python scripts/train.py --d 25 --loop
"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from scipy.stats import wasserstein_distance
from qfan import correlation_error_summary, corr_nan_safe
from qfan.born import noise_aware_ridge_fit, BornBlockModel
from train import build_problem, spearman_corr
from quad_features import ExtFeatures

# Generation shot count. This is a model parameter, not a precision knob:
# raising it narrows the sampling noise and so changes the distribution being
# generated, it does not simply reduce error. k=32 was selected train-side.
K_GEN = 32

def main():
    # Rebuild the exact problem the model was trained on: same data, same
    # split, same block layout, same sketch seeds. build_problem() is imported
    # from the trainer so there is one definition of all that, not two.
    (cfg, Y_tr, Y_te, d, blocks, sketcher, cache, spec, bank) = build_problem()
    r = np.load(ROOT / "outputs/model_d25.npz", allow_pickle=True)
    bank.set_theta(r["theta_final"]); bank.A = r["A_final"].copy()
    bank.b = r["b_final"].copy()
    RHO = r["rho_final"].copy()        # one trained share per block, 13 here
    # Quadratic record features: parity expectations augmented with exact
    # record-level second moments. This is what "quad" refers to above.
    ext = ExtFeatures(bank)

    # One ridge decoder per block, fitted on teacher-forced sketches. The fit
    # is noise-aware: the regressor input is a k-shot average, not an exact
    # expectation, so we add the known sampling covariance Sigma/k. Drop that
    # term and the decoder is biased toward zero, and the attenuation then
    # compounds along the chain.
    models = []
    for bi, (s, bsz) in enumerate(blocks):
        P = bank.setting_probs(cache[bi].astype(np.float64))
        W, _ = noise_aware_ridge_fit(ext.expectation(P),
                                     Y_tr[:, s:s+bsz].astype(np.float64),
                                     ext.sigma_bar(P), cfg.bank.ridge_alpha,
                                     k=K_GEN)
        models.append(BornBlockModel(W=W, start=s, bsz=bsz))

    def generate(n, rng, batch=1200):
        """Free-running generation. Unlike training, nothing here sees ground
        truth: each block is conditioned on the sketch of the model's own
        previous outputs. Batched only to bound memory."""
        out = np.zeros((n, d))
        for lo in range(0, n, batch):
            hi = min(lo + batch, n); Sraw, cur = sketcher.init_state(hi - lo)
            for bi, (s, bsz) in enumerate(blocks):
                P = bank.setting_probs(sketcher.mixed(Sraw, cur))
                # Record sharing. A trained fraction rho of the k records is
                # common to every pixel in the block, the rest is drawn
                # privately per pixel. Sharing records correlates their
                # sampling noise, which is how measurement noise becomes
                # physical intra-block correlation instead of being
                # suppressed. Single-pixel blocks have nothing to correlate.
                rs = float(RHO[bi])
                k_sh = int(round(rs * K_GEN)) if (bsz > 1 and rs < 1.0) else K_GEN
                k_ex = K_GEN - k_sh if bsz > 1 else 0
                rec = bank.sample_records(P, k_sh + bsz*k_ex if k_ex else K_GEN, rng)
                rec_T = [ext.Text[rec[g]] for g in range(bank.G)]
                cols = []
                for j in range(bsz):
                    # Pixel j sees the shared block [0, k_sh) plus its own
                    # private slice. Getting this indexing wrong silently
                    # destroys the intra-block correlation, so it is spelled
                    # out rather than done with fancy slicing.
                    sl = (np.arange(K_GEN) if k_ex == 0 else
                          np.concatenate([np.arange(k_sh),
                                          k_sh + j*k_ex + np.arange(k_ex)]))
                    cols.append(ext.record_features(rec_T, sl) @ models[bi].W[:, j])
                # Intensities are non-negative. Clipping here rather than in
                # the decoder keeps the ridge solve linear and exact.
                Yb = np.maximum(np.stack(cols, 1), 0.0)
                out[lo:hi, s:s+bsz] = Yb
                # Fold the block just generated into the running sketch so it
                # conditions the next one. This is the autoregressive step.
                cur = sketcher.update_inplace(Sraw, cur, Yb)
        return out

    # Same seed as training, so the printed number is reproducible. Generate
    # as many samples as the test set has, otherwise W1 compares distributions
    # estimated at different sample sizes.
    Y = generate(len(Y_te), np.random.default_rng(cfg.data.seed))

    # Three metrics of increasing difficulty. Per-pixel W1 tests the marginals,
    # the Pearson error tests joint structure, and the Spearman error repeats
    # that on ranks so it is insensitive to any monotone distortion.
    w1 = float(np.mean([wasserstein_distance(Y_te[:, j], Y[:, j]) for j in range(d)]))
    cs = correlation_error_summary(corr_nan_safe(Y_te), corr_nan_safe(Y), blocks)
    sp = correlation_error_summary(spearman_corr(Y_te), spearman_corr(Y), blocks)
    print(f"QFAN quad-feature k={K_GEN}: W1={w1:.5f}  off={cs['corr_mae_offdiag']:.4f}"
          f"  win={cs['corr_mae_within']:.4f}  crs={cs['corr_mae_cross']:.4f}"
          f"  Sp off={sp['corr_mae_offdiag']:.4f}")
    np.savez_compressed(ROOT / "outputs/samples_d25.npz",
                        Y_te=Y_te, Y_head=Y, rho_b=RHO, k=K_GEN,
                        meta=np.array([dict(w1=w1, **cs,
                            sp_off=sp['corr_mae_offdiag'])], dtype=object))
    print("[SAVE] outputs/samples_d25.npz")

if __name__ == "__main__":
    main()
