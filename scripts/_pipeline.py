"""
Shared end-to-end pipeline (used by run_d12.py and run_d25.py).


Jamal Slim
jamal.slim@desy.de


This is where configuration objects get turned into side-effect commands:
load data, build the sketch cache, train theta, fit ridge + residual gate,
rollout-refine, generate, optional copula calibration, compute metrics, save.
"""

import os
import sys
import pathlib

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split

# add src/ to sys.path so we can import the library
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qfan import (
    QFANConfig, load_dataset, _resolve_subset_n, select_subset_indices,
    OnlineCountSketch, build_blocks, build_prefix_sketch_cache,
    ShotBankSpec, TrainableShotPauliBank,
    train_theta_exact, train_theta_spsa,
    fit_all_blocks, sample_progressive, predict_future, rollout_refine_models,
    EmpiricalGaussianCopulaCalibrator,
    correlation_error_summary, corr_nan_safe,
)


def run_pipeline(cfg: QFANConfig, data_dir: pathlib.Path, out_dir: pathlib.Path,
                 verbose: bool = True) -> dict:
    """
    Execute the full QFAN v3 pipeline.

    Writes into out_dir:
        qfan_v3_<run_tag>_loss_curve.npz    training curves + theta checkpoints
        qfan_v3_<run_tag>_results.npz       generated samples, metrics, meta

    Returns a dict with the principal metrics, suitable for logging or CI.
    """
    data_dir = pathlib.Path(data_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.data.seed)

    print("=" * 78)
    print(f"QFAN v3 pipeline  run_tag={cfg.run_tag}")
    print(f"  data_dir = {data_dir}")
    print(f"  out_dir  = {out_dir}")
    print("=" * 78)

    # ---- load ----
    data_path = data_dir / cfg.data.data_path
    X_all = load_dataset(str(data_path), fallback_d=25, fallback_N=2500,
                         fallback_seed=cfg.data.seed)
    N_total = X_all.shape[0]

    subset_n = _resolve_subset_n(
        N_total, subset_n=cfg.data.subset_n, subset_frac=cfg.data.subset_frac)
    if cfg.data.use_subset and cfg.data.subset_apply == "before_split":
        idx_sub = select_subset_indices(
            X_all, subset_n=subset_n, mode=cfg.data.subset_mode,
            seed=cfg.data.subset_seed,
            kmeans_max_dim=cfg.data.subset_kmeans_max_dim,
            kmeans_ninit=cfg.data.subset_kmeans_ninit,
        )
        X_all = X_all[idx_sub]
        print(f"[SUBSET] before_split  mode={cfg.data.subset_mode}  "
              f"kept={X_all.shape[0]}/{N_total}")

    idx_all = np.arange(len(X_all))
    tr_idx, te_idx = train_test_split(idx_all, test_size=cfg.data.test_size,
                                      random_state=42)
    Y_tr, Y_te = X_all[tr_idx], X_all[te_idx]

    if cfg.data.use_subset and cfg.data.subset_apply == "train_only":
        Nt = Y_tr.shape[0]
        subset_n_tr = _resolve_subset_n(
            Nt, subset_n=cfg.data.subset_n, subset_frac=cfg.data.subset_frac)
        idx_tr_sub = select_subset_indices(
            Y_tr, subset_n=subset_n_tr, mode=cfg.data.subset_mode,
            seed=cfg.data.subset_seed,
            kmeans_max_dim=cfg.data.subset_kmeans_max_dim,
            kmeans_ninit=cfg.data.subset_kmeans_ninit,
        )
        Y_tr = Y_tr[idx_tr_sub]
        print(f"[SUBSET] train_only  kept_train={Y_tr.shape[0]}/{Nt}  "
              f"test={Y_te.shape[0]}")

    n, d = Y_tr.shape
    blocks = build_blocks(d, cfg.block_size)
    nb = len(blocks)
    print(f"[DATA] n_train={n}  n_test={len(Y_te)}  d={d}  "
          f"nb={nb}  b={cfg.block_size}")

    # ---- sketch cache ----
    sketcher = OnlineCountSketch(
        sketch_dim=cfg.sketch.sketch_dim,
        max_dim=max(cfg.sketch.max_dim_sketch, d),
        use_mixer=cfg.sketch.use_mixer,
        seed=cfg.data.seed,
        nonlinearity=cfg.sketch.nonlinearity,
        len_norm=cfg.sketch.len_norm,
    )
    print("[CACHE] building teacher-forced prefix sketch cache ...")
    sketch_cache = build_prefix_sketch_cache(Y_tr, blocks, sketcher)
    print(f"[CACHE] shape={sketch_cache.shape}  dtype={sketch_cache.dtype}")

    # ---- encoder bank ----
    bank_spec = ShotBankSpec(
        n_qubits=cfg.bank.n_qubits,
        depth=cfg.bank.q_depth,
        angle_dim=cfg.bank.angle_dim,
        measure_weight2=cfg.bank.measure_weight2,
        shots=cfg.bank.n_shots,
    )
    bank = TrainableShotPauliBank(
        sketch_dim=cfg.sketch.sketch_dim, spec=bank_spec,
        seed=cfg.data.seed + 13,
    )
    theta_init = bank.get_theta()
    print(f"[BANK] nq={cfg.bank.n_qubits}  depth={cfg.bank.q_depth}  "
          f"p_theta={theta_init.size}  pf={bank.feature_dim()}  G=2  "
          f"shots={cfg.bank.n_shots}")

    # ---- train theta ----
    if cfg.train.mode == "exact":
        hist = train_theta_exact(
            bank=bank, sketch_cache=sketch_cache, Y_train=Y_tr, blocks=blocks,
            ridge_alpha=cfg.bank.ridge_alpha,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch,
            val_batch_size=cfg.train.val_batch,
            lr=cfg.train.lr,
            beta1=cfg.train.beta1, beta2=cfg.train.beta2, eps=cfg.train.eps,
            grad_clip=cfg.train.grad_clip,
            lr_warmup=cfg.train.lr_warmup,
            lr_decay=cfg.train.lr_decay,
            lr_floor_frac=cfg.train.lr_floor_frac,
            ema_beta=cfg.train.ema_beta,
            pr_tail_frac=cfg.train.pr_tail_frac,
            use_precond=cfg.train.use_precond,
            precond_damp=cfg.train.precond_damp,
            precond_every=cfg.train.precond_every,
            all_blocks=cfg.train.all_blocks,
            block_subset=cfg.train.block_subset,
            sigma_mults=cfg.mmd.sigma_mults,
            feature_chunk_size=cfg.bank.feature_chunk_size,
            seed=cfg.data.seed, verbose=verbose,
        )
    elif cfg.train.mode == "spsa":
        hist = train_theta_spsa(
            bank=bank, sketch_cache=sketch_cache, Y_train=Y_tr, blocks=blocks,
            ridge_alpha=cfg.bank.ridge_alpha,
            steps=cfg.train.spsa_steps, batch_size=cfg.train.spsa_batch,
            a0=cfg.train.spsa_a0, c0=cfg.train.spsa_c0,
            alpha=cfg.train.spsa_alpha, gamma=cfg.train.spsa_gamma,
            sigma_mults=cfg.mmd.sigma_mults,
            val_batch_size=cfg.train.val_batch,
            feature_chunk_size=cfg.bank.feature_chunk_size,
            seed=cfg.data.seed, verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown train.mode={cfg.train.mode}")

    loss_curve_path = out_dir / f"qfan_v3_{cfg.run_tag}_loss_curve.npz"
    np.savez_compressed(
        str(loss_curve_path),
        train_loss=np.asarray(hist["train_loss"]),
        val_loss=np.asarray(hist["val_loss"]),
        val_loss_ema=np.asarray(hist["val_loss_ema"]),
        grad_norm=np.asarray(hist["grad_norm"]),
        step_norm=np.asarray(hist["step_norm"]),
        lr=np.asarray(hist["lr"]),
        sigmas=np.array(hist["sigmas"], dtype=object),
        theta_init=theta_init,
        theta_final=hist["theta_final"],
        theta_last_iter=hist["theta_last_iter"],
        val_loss_pr=float(hist.get("val_loss_pr", -1.0)),
        train_mode=cfg.train.mode,
        run_tag=cfg.run_tag,
    )
    print(f"[SAVE] {loss_curve_path}")

    # ---- fit ridge + residual gate ----
    print("=" * 78)
    print("FIT: teacher-forced ridge + classical residual gate at trained theta")
    print("=" * 78)
    models = fit_all_blocks(
        bank=bank, sketch_cache=sketch_cache, Y_train=Y_tr,
        blocks=blocks, ridge_alpha=cfg.bank.ridge_alpha,
        use_safe_gate=cfg.residual.use_safe_gate,
        resid_q=cfg.residual.resid_qubits,
        resid_max_clusters=cfg.residual.resid_max_clusters,
        resid_kmeans_ninit=cfg.residual.resid_kmeans_ninit,
        resid_whiten=cfg.residual.resid_whiten,
        resid_eps=cfg.residual.resid_whiten_eps,
        resid_shrink=cfg.residual.resid_whiten_shrink,
        resid_temp=cfg.residual.resid_sample_temp,
        resid_jitter=cfg.residual.resid_jitter_scale,
        feature_chunk_size=cfg.bank.feature_chunk_size,
    )

    # ---- rollout refinement ----
    models = rollout_refine_models(
        bank=bank, models=models, Y_train=Y_tr, blocks=blocks,
        sketcher=sketcher, ridge_alpha=cfg.bank.ridge_alpha,
        epochs=cfg.rollout.epochs,
        max_rollout_ratio=cfg.rollout.max_rollout_ratio,
        monitor_n=cfg.rollout.monitor_n,
        seed=cfg.rollout.reseed,
        use_safe_gate=cfg.residual.use_safe_gate,
        resid_q=cfg.residual.resid_qubits,
        resid_max_clusters=cfg.residual.resid_max_clusters,
        resid_kmeans_ninit=cfg.residual.resid_kmeans_ninit,
        resid_whiten=cfg.residual.resid_whiten,
        resid_eps=cfg.residual.resid_whiten_eps,
        resid_shrink=cfg.residual.resid_whiten_shrink,
        resid_temp=cfg.residual.resid_sample_temp,
        resid_jitter=cfg.residual.resid_jitter_scale,
        clip_nonnegative=cfg.sketch.clip_nonnegative,
        feature_chunk_size=cfg.bank.feature_chunk_size,
    )

    # ---- generation ----
    print("=" * 78); print("GENERATION"); print("=" * 78)
    Y_gen_raw = sample_progressive(
        bank, models, d, blocks, sketcher,
        n_samples=len(Y_te), rng_np=rng,
        clip_nonnegative=cfg.sketch.clip_nonnegative,
        feature_chunk_size=cfg.bank.feature_chunk_size,
    )

    copula = None
    Y_gen = Y_gen_raw.copy()
    if cfg.copula.enabled:
        print("[CAL] fitting Gaussian-copula calibrator ...")
        Y_ref_gen = sample_progressive(
            bank, models, d, blocks, sketcher,
            n_samples=len(Y_tr),
            rng_np=np.random.default_rng(cfg.data.seed + 555),
            clip_nonnegative=cfg.sketch.clip_nonnegative,
            feature_chunk_size=cfg.bank.feature_chunk_size,
        )
        copula = EmpiricalGaussianCopulaCalibrator(
            shrink=cfg.copula.shrink, eps=cfg.copula.eps,
            clip_nonnegative=cfg.sketch.clip_nonnegative,
        ).fit(Y_ref_gen, Y_tr)
        Y_gen = copula.transform(Y_gen_raw)

    # ---- metrics ----
    print("=" * 78); print("METRICS"); print("=" * 78)
    w1_per_dim_raw = np.array(
        [wasserstein_distance(Y_te[:, j], Y_gen_raw[:, j]) for j in range(d)],
        dtype=np.float64,
    )
    w1_mean_raw = float(w1_per_dim_raw.mean())
    w1_per_dim = np.array(
        [wasserstein_distance(Y_te[:, j], Y_gen[:, j]) for j in range(d)],
        dtype=np.float64,
    )
    w1_mean = float(w1_per_dim.mean())

    Y_pred = np.array([])
    mse_per_dim = np.array([])
    mse_mean = -1.0
    k_obs = -1
    if cfg.do_prediction_eval:
        k_obs = d // 2
        Y_pred_full = predict_future(
            bank, models, d, blocks, sketcher, Y_te[:, :k_obs],
            clip_nonnegative=cfg.sketch.clip_nonnegative,
            feature_chunk_size=cfg.bank.feature_chunk_size,
        )
        if copula is not None:
            Y_pred_full = copula.transform(Y_pred_full)
        mse_per_dim = ((Y_pred_full[:, k_obs:] - Y_te[:, k_obs:]) ** 2)\
            .mean(axis=0).astype(np.float64)
        mse_mean = float(mse_per_dim.mean())
        Y_pred = Y_pred_full

    C_test = corr_nan_safe(Y_te)
    C_gen_raw = corr_nan_safe(Y_gen_raw)
    C_gen = corr_nan_safe(Y_gen)
    corr_summary_raw = correlation_error_summary(C_test, C_gen_raw, blocks)
    corr_summary = correlation_error_summary(C_test, C_gen, blocks)

    # ---- save ----
    meta = dict(
        run_tag=cfg.run_tag,
        train_mode=cfg.train.mode,
        d=int(d), nb=int(nb), block_size=int(cfg.block_size),
        n_train=int(Y_tr.shape[0]), n_test=int(Y_te.shape[0]),
        nq=int(cfg.bank.n_qubits), depth=int(cfg.bank.q_depth),
        p_theta=int(theta_init.size), pf=int(bank.feature_dim()),
        shots=int(cfg.bank.n_shots),
        epochs=int(cfg.train.epochs), batch=int(cfg.train.batch),
        lr=float(cfg.train.lr),
        use_precond=bool(cfg.train.use_precond),
        pr_tail_frac=float(cfg.train.pr_tail_frac),
        rollout_epochs=int(cfg.rollout.epochs),
        use_copula=bool(cfg.copula.enabled),
        val_loss_first=float(hist["val_loss"][0]) if hist["val_loss"] else -1.0,
        val_loss_last=float(hist["val_loss"][-1]) if hist["val_loss"] else -1.0,
        val_loss_min=float(min(hist["val_loss"])) if hist["val_loss"] else -1.0,
        val_loss_pr=float(hist.get("val_loss_pr", -1.0)),
        w1_mean_raw=w1_mean_raw, w1_mean=w1_mean,
        mse_mean=mse_mean, k_obs=int(k_obs),
        **corr_summary_raw,
        **{f"cal_{k}": v for k, v in corr_summary.items()},
    )

    results_path = out_dir / f"qfan_v3_{cfg.run_tag}_results.npz"
    np.savez_compressed(
        str(results_path),
        Y_tr=Y_tr, Y_te=Y_te,
        Y_gen_raw=Y_gen_raw, Y_gen=Y_gen,
        C_test=C_test, C_gen_raw=C_gen_raw, C_gen=C_gen,
        w1_per_dim_raw=w1_per_dim_raw, w1_mean_raw=w1_mean_raw,
        w1_per_dim=w1_per_dim, w1_mean=w1_mean,
        k_obs=int(k_obs), Y_pred=Y_pred,
        mse_per_dim=mse_per_dim, mse_mean=mse_mean,
        theta_init=theta_init,
        theta_trained=hist["theta_final"],
        theta_last_iter=hist["theta_last_iter"],
        pauli_strings=np.array(bank.paulis, dtype=object),
        train_loss_curve=np.asarray(hist["train_loss"]),
        val_loss_curve=np.asarray(hist["val_loss"]),
        val_loss_ema_curve=np.asarray(hist["val_loss_ema"]),
        grad_norm_curve=np.asarray(hist["grad_norm"]),
        step_norm_curve=np.asarray(hist["step_norm"]),
        lr_curve=np.asarray(hist["lr"]),
        meta=np.array([meta], dtype=object),
    )

    print("\n" + "=" * 78)
    print(f"[SUMMARY]  run_tag = {cfg.run_tag}")
    print(f"  train_mode         : {cfg.train.mode}")
    print(f"  p_theta            : {theta_init.size}")
    if hist["val_loss"]:
        print(f"  val_loss first/last: "
              f"{hist['val_loss'][0]:.6f}  ->  {hist['val_loss'][-1]:.6f}")
        print(f"  val_loss min       : {min(hist['val_loss']):.6f}")
    print(f"  val_loss PR-averaged: {hist.get('val_loss_pr', -1):.6f}")
    print(f"  W1 mean  (raw/cal) : {w1_mean_raw:.6g} / {w1_mean:.6g}")
    print(f"  corr offdiag       : "
          f"{corr_summary_raw['corr_mae_offdiag']:.5f} / {corr_summary['corr_mae_offdiag']:.5f}")
    print(f"  corr within        : "
          f"{corr_summary_raw['corr_mae_within']:.5f} / {corr_summary['corr_mae_within']:.5f}")
    print(f"  corr cross         : "
          f"{corr_summary_raw['corr_mae_cross']:.5f} / {corr_summary['corr_mae_cross']:.5f}")
    if cfg.do_prediction_eval:
        print(f"  pred mean MSE (k_obs={k_obs}) : {mse_mean:.6g}")
    print(f"[SAVE] {results_path}")
    print("=" * 78)

    return meta
