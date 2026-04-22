"""
Full QFAN generation pipeline (on top of a trained encoder theta).

Jamal Slim
jamal.slim@desy.de



  - fit_all_blocks:      per-block ridge + residual gate
  - sample_progressive:  free-running autoregressive generation
  - predict_future:      mean-only prediction of unseen dims from observed prefix
  - rollout_refine_models: DAgger-lite refinement to reduce teacher-forcing
                           exposure bias (paper Sec. III-F teacher/rollout note)
"""

import numpy as np

from .quantum import TrainableShotPauliBank, compute_features_chunked
from .ridge import ridge_fit
from .sketch import OnlineCountSketch, build_prefix_sketch_cache
from .decoder import SafeResidualGate, BlockModel
from .metrics import correlation_error_summary
from .utils import corr_nan_safe


def fit_all_blocks(bank: TrainableShotPauliBank,
                   sketch_cache: np.ndarray,
                   Y_train: np.ndarray,
                   blocks,
                   ridge_alpha: float,
                   use_safe_gate: bool = True,
                   resid_q: int = 3, resid_max_clusters: int = 64,
                   resid_kmeans_ninit: int = 10,
                   resid_whiten: bool = True,
                   resid_eps: float = 1e-6, resid_shrink: float = 1e-2,
                   resid_temp: float = 1.0, resid_jitter: float = 0.0,
                   feature_chunk_size: int = 256):
    """Fit the per-block ridge decoder + residual gate at the CURRENT bank.theta."""
    models = []
    for bi, (start, bsz) in enumerate(blocks):
        S = sketch_cache[bi].astype(np.float64, copy=False)
        Yb = Y_train[:, start:start + bsz].astype(np.float64, copy=False)
        F = compute_features_chunked(bank, S, chunk_size=feature_chunk_size)
        W, R = ridge_fit(F, Yb, alpha=ridge_alpha)
        Mu = F @ W
        gate = SafeResidualGate(
            resid_q=resid_q, resid_max_clusters=resid_max_clusters,
            n_init=resid_kmeans_ninit,
            whiten=resid_whiten, eps=resid_eps, shrink=resid_shrink,
            temp=resid_temp, jitter=resid_jitter,
        )
        if use_safe_gate:
            G = np.concatenate([S, F, Mu], axis=1)
            gate.fit(G, R)
        models.append(BlockModel(W=W, R=R, gate=gate, start=start, bsz=bsz))
    return models


def sample_progressive(bank: TrainableShotPauliBank,
                       models,
                       d: int, blocks,
                       sketcher: OnlineCountSketch,
                       n_samples: int,
                       rng_np: np.random.Generator,
                       clip_nonnegative: bool = True,
                       feature_chunk_size: int = 256) -> np.ndarray:
    """
    Free-running autoregressive generation. Differs from teacher-forced
    training in that the sketch is updated with MODEL outputs, not with
    ground truth (see paper Sec. III-F teacher/rollout note; this is the
    path where exposure bias, if any, manifests).
    """
    out = np.zeros((n_samples, d), dtype=np.float64)
    Sraw, cur_len = sketcher.init_state(n_samples)
    for bi, (start, bsz) in enumerate(blocks):
        Sprefix = sketcher.mixed(Sraw, cur_len)
        F = compute_features_chunked(bank, Sprefix, chunk_size=feature_chunk_size)
        Mu = F @ models[bi].W
        G = np.concatenate([Sprefix, F, Mu], axis=1)
        eps = models[bi].gate.sample(G, models[bi].R, rng_np)
        Yblk = Mu + eps
        if clip_nonnegative:
            Yblk = np.maximum(Yblk, 0.0)
        out[:, start:start + bsz] = Yblk
        cur_len = sketcher.update_inplace(Sraw, cur_len, Yblk)
    return out


def predict_future(bank: TrainableShotPauliBank,
                   models,
                   d: int, blocks,
                   sketcher: OnlineCountSketch,
                   Y_partial: np.ndarray,
                   clip_nonnegative: bool = True,
                   feature_chunk_size: int = 256) -> np.ndarray:
    """Mean-only prediction of unseen dims given the first k dims of Y_partial."""
    Y_partial = np.asarray(Y_partial, np.float64)
    n, k = Y_partial.shape
    out = np.zeros((n, d), dtype=np.float64)
    out[:, :k] = Y_partial
    Sraw, cur_len = sketcher.init_state(n)
    partial_bi = None
    for bi, (start, bsz) in enumerate(blocks):
        end = start + bsz
        if end <= k:
            cur_len = sketcher.update_inplace(Sraw, cur_len, Y_partial[:, start:end])
        elif start < k < end:
            partial_bi = bi; break
        else:
            break
    start_bi = 0 if partial_bi is None else partial_bi
    for bi in range(start_bi, len(blocks)):
        start, bsz = blocks[bi]
        end = start + bsz
        if end <= k:
            continue
        Sprefix = sketcher.mixed(Sraw, cur_len)
        F = compute_features_chunked(bank, Sprefix, chunk_size=feature_chunk_size)
        block = F @ models[bi].W
        if clip_nonnegative:
            block = np.maximum(block, 0.0)
        if start < k < end:
            obs_len = k - start
            block[:, :obs_len] = Y_partial[:, start:k]
        out[:, start:end] = block
        cur_len = sketcher.update_inplace(Sraw, cur_len, block)
    out[:, :k] = Y_partial
    return out


def rollout_refine_models(bank: TrainableShotPauliBank,
                          models,
                          Y_train: np.ndarray,
                          blocks,
                          sketcher: OnlineCountSketch,
                          ridge_alpha: float,
                          epochs: int = 6, max_rollout_ratio: float = 0.85,
                          monitor_n: int = 512, seed: int = 2025,
                          use_safe_gate: bool = True,
                          resid_q: int = 3, resid_max_clusters: int = 64,
                          resid_kmeans_ninit: int = 10,
                          resid_whiten: bool = True,
                          resid_eps: float = 1e-6, resid_shrink: float = 1e-2,
                          resid_temp: float = 1.0, resid_jitter: float = 0.0,
                          clip_nonnegative: bool = True,
                          feature_chunk_size: int = 256):
    """
    DAgger-lite rollout refinement: re-fit ridge + residual gate on a mixture
    of teacher-forced (truth) and free-running (rollout) prefixes. Mix ratio
    increases linearly from 0 to `max_rollout_ratio` over `epochs`.
    """
    if epochs <= 0:
        return models
    Y_train = np.asarray(Y_train, np.float64)
    n, d = Y_train.shape
    prng = np.random.default_rng(seed)
    print("=" * 78)
    print(f"ROLLOUT REFINEMENT  epochs={epochs}  max_ratio={max_rollout_ratio}")
    print("=" * 78)
    for ep in range(epochs):
        rr = float(max_rollout_ratio) * float(ep + 1) / float(epochs)
        Y_roll = sample_progressive(
            bank, models, d, blocks, sketcher, n_samples=n, rng_np=prng,
            clip_nonnegative=clip_nonnegative,
            feature_chunk_size=feature_chunk_size,
        )
        Y_cond = (1.0 - rr) * Y_train + rr * Y_roll
        if clip_nonnegative:
            Y_cond = np.maximum(Y_cond, 0.0)
        cache = build_prefix_sketch_cache(Y_cond, blocks, sketcher)
        models = fit_all_blocks(
            bank, cache, Y_train, blocks, ridge_alpha,
            use_safe_gate=use_safe_gate,
            resid_q=resid_q, resid_max_clusters=resid_max_clusters,
            resid_kmeans_ninit=resid_kmeans_ninit,
            resid_whiten=resid_whiten, resid_eps=resid_eps,
            resid_shrink=resid_shrink,
            resid_temp=resid_temp, resid_jitter=resid_jitter,
            feature_chunk_size=feature_chunk_size,
        )
        mon_n = min(int(monitor_n), n)
        idx = prng.choice(n, size=mon_n, replace=False)
        Y_ref = Y_train[idx]
        Y_mon = sample_progressive(
            bank, models, d, blocks, sketcher, n_samples=mon_n, rng_np=prng,
            clip_nonnegative=clip_nonnegative,
            feature_chunk_size=feature_chunk_size,
        )
        s = correlation_error_summary(corr_nan_safe(Y_ref), corr_nan_safe(Y_mon), blocks)
        print(f"  refine {ep+1:2d}/{epochs}  rr={rr:.3f}  "
              f"offdiag={s['corr_mae_offdiag']:.5f}  "
              f"within={s['corr_mae_within']:.5f}  "
              f"cross={s['corr_mae_cross']:.5f}")
    return models
