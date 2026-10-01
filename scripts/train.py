#!/usr/bin/env python3
"""
QFAN training. The image size is an argument, matching
scripts/plot_paper_figures.py.

    python scripts/train.py --d 12 --loop
    python scripts/train.py --d 25 --loop

Same algorithm at both sizes. The circuit settings differ:

           depth   angle dim   theta   blocks
    d=12       2           8      12        6
    d=25       3          16      18       13

and everything else is shared, n_q = 3, block size b = 2, k = 64 records per
block, G = 3 measurement settings, 300 epochs, the CF-MMD objective with exact
analytic gradients, and one trained record share per block.

Training runs in resumable chunks. Each call trains until --budget-s expires,
then exits. Re-run the same command, or pass --loop, until it reports
TRAINING COMPLETE. Progress is checkpointed in
outputs/qfan_<tag>_train_ckpt.pkl and the checkpoint is discarded if its
block count does not match the selected image size.

Requires data/cal_shower_img_12q.npy or data/cal_shower_img_25q.npy.

Writes
    d=12   outputs/model_d12.npz, loss_d12.npz
    d=25   outputs/model_d25.npz, loss_d25.npz
"""
import argparse
import pathlib
import pickle
import sys
import time

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qfan import (                                                  # noqa: E402
    d12_config, d25_config, load_dataset, _resolve_subset_n, select_subset_indices,
    OnlineCountSketch, build_blocks, build_prefix_sketch_cache,
    EmpiricalGaussianCopulaCalibrator,
    correlation_error_summary, corr_nan_safe,
)
# The factorized loss is identical to the original to ~1e-17 on every code
# path (tests/test_factorized.py) but never builds the 2^(G n_q) joint
# record table, so it runs at any register size.
from qfan.born_factorized import (                                   # noqa: E402
    cf_mmd_loss_and_grads_factorized as cf_mmd_loss_and_grads)
from qfan.born import (                                             # noqa: E402
    BornBankSpec, StatevectorBornBank,
    draw_rff_frequencies, draw_joint_rff_frequencies,
    fit_all_blocks_born, sample_progressive_born, predict_future_born,
)
from qfan.mmd import median_sigma                                   # noqa: E402
from qfan.train import AdamOpt, _scheduled_lr, _grad_clip           # noqa: E402

# ---- protocol constants ----
# Records per block during training. The CF-MMD objective is exact in k, so
# this is not a Monte Carlo budget: it is the k whose sampling law we match.
K_SHOTS = 64
EPOCHS = 300
# Weight on the risk term that discourages the decoder from relying on
# feature directions the sampling noise cannot support.
GAMMA_RISK = 10.0
# b=2 throughout. Small blocks make the decoder's job easy (rho = p_f/b = 9,
# far above the rank floor) and push the modelling burden onto the sketch
# conditioning and the record sharing, which is where we want it.
BLOCK_SIZE = 2
# Pixels per detector layer, in image order. When set, blocks are cut per
# layer so record sharing never couples pixels from different layers.
# Physical layer widths of a generic geometry, read from the .json that the
# data preparation writes beside the data file. Recorded in the results for
# reporting only; blocks are always contiguous, of --block-size pixels.
PHYS_LAYER_WIDTHS = None
# --seed for generic geometries: changes the train/test split, the circuit's
# initialisation and the sketch hashing together. None keeps the default.
SEED_OVERRIDE = None
# Weight of the ridge decoder's shot-noise penalty (the library supports others;
# 1 is the noise-aware ridge used throughout).
NOISE_WEIGHT = 1.0
# --learn-k: each block learns its own number of measurement records k_b,
# trained by the same loss as the circuit (log k_b by Adam; its gradient is a
# central difference of the loss, the circuit gradients stay exact). At
# generation each block averages round(k_b) records. Off: k = K_SHOTS everywhere.
LEARN_K = False
LR_K = 0.05
K_BOUNDS = (2.0, 1024.0)
K_FD = 0.05
# Random Fourier frequencies used to evaluate the characteristic-function
# discrepancy. More frequencies means a tighter distributional match at
# linear cost. 1024 was where the validation loss stopped improving.
N_FREQ = 1024
BATCH = 256
VAL_BATCH = 512
LR = 0.1
LR_ENC = 0.1
LR_RHO = 0.05
# Record shares start high, meaning pixels in a block initially share almost
# all their records and are strongly correlated. Training pulls them down.
RHO_INIT = 0.9
# Half the frequencies carry a phase on the conditioning prefix, so the loss
# matches the joint law of block and prefix rather than the block marginal
# alone. Set this to 0 and the blocks decouple.
COND_FRAC = 0.5
# Multi-bandwidth kernel. A single bandwidth matches structure at one scale
# only, and calorimeter marginals span two orders of magnitude in intensity.
SIGMA_MULTS = (0.5, 1.0, 2.0, 4.0)
# The CF gradient can spike early when the ridge solve is poorly conditioned.
GRAD_CLIP = 5.0
LR_WARMUP = 5
LR_FLOOR_FRAC = 0.1
EMA_BETA = 0.9
PR_TAIL_FRAC = 0.3
# image size, set by --d before anything below is used
D_PIXELS = 25
_PER_D = {12: dict(config=d12_config, angle_dim=8, depth=2, tag="d12"),
          25: dict(config=d25_config, angle_dim=16, depth=3, tag="d25")}


def _cfg_for(d):
    return _PER_D[d]


def register_geometry(data_path, n_qubits, depth, angle_dim, tag):
    """Add a geometry beyond the two paper configurations. The algorithm is
    unchanged; only the data, register width and circuit shape differ."""
    def _make():
        cfg = d25_config()
        cfg.run_tag = tag
        cfg.data.data_path = str(data_path)
        cfg.bank.n_qubits = int(n_qubits)
        cfg.bank.q_depth = int(depth)
        if SEED_OVERRIDE is not None:
            cfg.data.seed = int(SEED_OVERRIDE)
        return cfg
    _PER_D[tag] = dict(config=_make, angle_dim=int(angle_dim),
                       depth=int(depth), tag=tag)
    return tag


TAG = _PER_D[D_PIXELS]["tag"]
CKPT = PROJECT_ROOT / "outputs" / f"ckpt_{TAG}.pkl"


# ----------------------------------------------------------------------
# (ii) deterministic per-pixel monotone quantile transport
# ----------------------------------------------------------------------
class MonotoneMarginalMap:
    """T_j = Q_train,j o F_ref,j  per pixel, piecewise-linear, monotone.

    Deterministic: injects no randomness. Coordinate-wise: injects no
    cross-pixel information; Spearman correlations invariant up to ties.
    Fitted train-side from a model reference sample (like k and rho)."""

    def fit(self, Y_ref_model: np.ndarray, Y_train: np.ndarray):
        self.d = Y_ref_model.shape[1]
        self.ref_sorted = [np.sort(Y_ref_model[:, j]) for j in range(self.d)]
        self.tr_sorted = [np.sort(Y_train[:, j]) for j in range(self.d)]
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        out = np.empty_like(Y, dtype=np.float64)
        for j in range(self.d):
            ref = self.ref_sorted[j]
            n = ref.size
            # mid-rank of each value within the model reference
            lo = np.searchsorted(ref, Y[:, j], side="left")
            hi = np.searchsorted(ref, Y[:, j], side="right")
            u = (0.5 * (lo + hi)) / n
            u = np.clip(u, 0.5 / n, 1.0 - 0.5 / n)
            out[:, j] = np.quantile(self.tr_sorted[j], u,
                                    method="linear")
        return out


def spearman_corr(Y: np.ndarray) -> np.ndarray:
    """Rank (Spearman) correlation matrix, ties by average rank."""
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(Y[:, j]) for j in range(Y.shape[1])])
    return corr_nan_safe(R)


def build_problem(d_pixels=None):
    sel = _PER_D[d_pixels if d_pixels is not None else D_PIXELS]
    cfg = sel['config']()
    data_dir = PROJECT_ROOT / "data"
    # an absolute path from register_geometry overrides the data directory
    X_all = load_dataset(str(data_dir / cfg.data.data_path))
    N_total = X_all.shape[0]
    subset_n = _resolve_subset_n(N_total, subset_n=cfg.data.subset_n,
                                 subset_frac=cfg.data.subset_frac)
    if cfg.data.use_subset and cfg.data.subset_apply == "before_split":
        idx = select_subset_indices(X_all, subset_n=subset_n,
                                    mode=cfg.data.subset_mode,
                                    seed=cfg.data.subset_seed)
        X_all = X_all[idx]
    tr_idx, te_idx = train_test_split(np.arange(len(X_all)),
                                      test_size=cfg.data.test_size,
                                      random_state=42)
    Y_tr, Y_te = X_all[tr_idx], X_all[te_idx]
    d = Y_tr.shape[1]
    _key = d_pixels if d_pixels is not None else D_PIXELS
    # the two paper configurations are keyed by their image size, so check the
    # file agrees; a generic geometry takes d from the file itself
    if isinstance(_key, int):
        assert d == _key, f"dataset has d={d} but --d selected {_key}"
    blocks = build_blocks(d, BLOCK_SIZE)
    sketcher = OnlineCountSketch(
        sketch_dim=cfg.sketch.sketch_dim,
        max_dim=max(cfg.sketch.max_dim_sketch, d),
        use_mixer=cfg.sketch.use_mixer, seed=cfg.data.seed,
        nonlinearity=cfg.sketch.nonlinearity, len_norm=cfg.sketch.len_norm)
    sketch_cache = build_prefix_sketch_cache(Y_tr, blocks, sketcher)
    spec = BornBankSpec(n_qubits=cfg.bank.n_qubits, depth=sel['depth'],
                        angle_dim=sel['angle_dim'],
                        include_y=True, gen_shots_k=K_SHOTS)
    bank = StatevectorBornBank(cfg.sketch.sketch_dim, spec,
                               seed=cfg.data.seed + 13)
    return cfg, Y_tr, Y_te, d, blocks, sketcher, sketch_cache, spec, bank


def setup_train(bank, sketch_cache, Y_tr, blocks, seed):
    """Pre-loop setup: fixed seeds, so the same draws on every run."""
    prng = np.random.default_rng(seed + 999)
    n_train = Y_tr.shape[0]
    # Validation may take at most a fifth of the training rows. Capping only at
    # VAL_BATCH let a small training set be swallowed whole, leaving an empty
    # training pool and a ZeroDivisionError three calls later.
    val_n = min(VAL_BATCH, n_train // 5)
    if n_train - val_n < 2:
        raise ValueError(f"only {n_train} training rows: too few to train on")
    val_idx = prng.choice(n_train, size=val_n, replace=False)
    train_mask = np.ones(n_train, dtype=bool)
    train_mask[val_idx] = False
    train_pool = np.where(train_mask)[0]

    sigmas_pb, Omega_pb, Omega_c_pb = [], [], []
    for bi, (start, bsz) in enumerate(blocks):
        Yb = Y_tr[val_idx, start:start + bsz]
        s_star = median_sigma(Yb, Yb, cap=256, seed=seed)
        sigs = [mf * s_star for mf in SIGMA_MULTS]
        sigmas_pb.append(sigs)
        d_c = start
        if d_c > 0:
            Xc = Y_tr[val_idx, :start]
            sc_star = median_sigma(Xc, Xc, cap=256, seed=seed)
            sigs_c = [mf * sc_star for mf in SIGMA_MULTS]
            Oy, Oc = draw_joint_rff_frequencies(
                sigs, bsz, sigs_c, d_c, N_FREQ, cond_frac=COND_FRAC,
                seed=seed + 31 * bi)
        else:
            Oy = draw_rff_frequencies(sigs, bsz, N_FREQ, seed=seed + 31 * bi)
            Oc = None
        Omega_pb.append(Oy)
        Omega_c_pb.append(Oc)
    return prng, val_idx, train_pool, sigmas_pb, Omega_pb, Omega_c_pb


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def total_val(bank, th, blocks, sketch_cache, val_idx, Y_tr,
              Omega_pb, Omega_c_pb, ridge_alpha, r_logit_vec, logk_vec=None):
    v = 0.0
    for bi, (start, bsz) in enumerate(blocks):
        S_v = sketch_cache[bi, val_idx].astype(np.float64, copy=False)
        Y_v = Y_tr[val_idx, start:start + bsz].astype(np.float64, copy=False)
        Xc = Y_tr[val_idx, :start] if Omega_c_pb[bi] is not None else None
        v += cf_mmd_loss_and_grads(
            bank, S_v, Y_v, th, Omega_pb[bi], ridge_alpha,
            k_shots=(K_SHOTS if logk_vec is None else float(np.exp(logk_vec[bi]))),
            gamma_risk=GAMMA_RISK, compute_grad=False,
            noise_weight=NOISE_WEIGHT,
            X_cond=Xc, Omega_cond=Omega_c_pb[bi],
            rho=float(sigmoid(r_logit_vec[bi])))[0]
    return v


def _run_one_chunk(args):
    """Train until the time budget expires. Returns True when training has
    completed, False when the chunk ended early and a re-run is needed."""
    global EPOCHS
    global D_PIXELS, TAG, CKPT, BLOCK_SIZE, PHYS_LAYER_WIDTHS, SEED_OVERRIDE
    global NOISE_WEIGHT, GAMMA_RISK, LEARN_K
    LEARN_K = bool(getattr(args, "learn_k", False))
    if getattr(args, "seed", None) is not None:
        SEED_OVERRIDE = int(args.seed)
    if getattr(args, "data", None):
        D_PIXELS = register_geometry(args.data, args.nq, args.depth,
                                     args.angle_dim, args.tag)
        BLOCK_SIZE = int(args.block_size)
        import json as _json
        import pathlib as _pl
        _side = _pl.Path(args.data).with_suffix(".json")
        if _side.exists():
            PHYS_LAYER_WIDTHS = _json.loads(_side.read_text()).get("layer_widths")
    else:
        D_PIXELS = int(args.d)
    TAG = _PER_D[D_PIXELS]["tag"]
    CKPT = PROJECT_ROOT / "outputs" / f"ckpt_{TAG}.pkl"
    EPOCHS = int(args.epochs)
    t_start = time.time()
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    (cfg, Y_tr, Y_te, d, blocks, sketcher, sketch_cache,
     spec, bank) = build_problem()
    seed = cfg.data.seed
    ridge_alpha = cfg.bank.ridge_alpha
    nb = len(blocks)
    p_theta = bank.n_var
    theta_init0 = bank.get_theta().copy()
    A_init0, b_init0 = bank.A.copy(), bank.b.copy()

    (prng, val_idx, train_pool, sigmas_pb, Omega_pb,
     Omega_c_pb) = setup_train(bank, sketch_cache, Y_tr, blocks, seed)

    n_enc = bank.L * bank.m + bank.L
    pr_T0 = int(np.floor((1.0 - PR_TAIL_FRAC) * EPOCHS))
    logit0 = float(np.log(RHO_INIT / (1.0 - RHO_INIT)))

    if CKPT.exists():
        with open(CKPT, "rb") as f:
            st = pickle.load(f)
        if np.size(st.get("r_logit", [])) != nb:
            print(f"[CKPT] {CKPT.name} holds {np.size(st.get('r_logit', []))} "
                  f"blocks but this run has {nb}. Ignoring it and starting "
                  f"fresh. Delete the file to silence this.")
            st = None
    else:
        st = None

    if st is not None:
        theta = st["theta"]
        bank.set_theta(theta)
        bank.A = st["A"]; bank.b = st["b"]
        r_logit = st["r_logit"]                     # (nb,) vector
        opt = AdamOpt(p_theta, lr=LR); opt.m, opt.v, opt.t = st["opt"]
        opt_enc = AdamOpt(n_enc, lr=LR_ENC)
        opt_enc.m, opt_enc.v, opt_enc.t = st["opt_enc"]
        opt_rho = AdamOpt(nb, lr=LR_RHO)
        opt_rho.m, opt_rho.v, opt_rho.t = st["opt_rho"]
        logk = np.asarray(st.get("logk", np.full(nb, np.log(K_SHOTS))), float)
        opt_k = AdamOpt(nb, lr=LR_K)
        if "opt_k" in st:
            opt_k.m, opt_k.v, opt_k.t = st["opt_k"]
        prng.bit_generator.state = st["prng_state"]
        ck_ep = int(st.get("epochs_total", EPOCHS))
        if ck_ep != EPOCHS:
            print(f"[WARN] --epochs={EPOCHS} conflicts with the "
                  f"checkpoint's locked total ({ck_ep}); using "
                  f"{ck_ep} to preserve the LR/PR protocol.")
            EPOCHS = ck_ep
        t0_epoch = st["epoch"]
        pr_buffer = st["pr_buffer"]
        hist = st["hist"]
        val_ema = st["val_ema"]
        print(f"[RESUME] epoch {t0_epoch}/{EPOCHS}  "
              f"rho_b={np.array2string(sigmoid(r_logit), precision=2)}",
              flush=True)
    else:
        theta = bank.get_theta().copy()
        r_logit = np.full(nb, logit0)               # per-block rho_b
        opt = AdamOpt(p_theta, lr=LR)
        opt_enc = AdamOpt(n_enc, lr=LR_ENC)
        opt_rho = AdamOpt(nb, lr=LR_RHO)
        logk = np.full(nb, np.log(K_SHOTS))          # per-block log k_b
        opt_k = AdamOpt(nb, lr=LR_K)
        t0_epoch = 0
        pr_buffer = []
        hist = dict(train_loss=[], val_loss=[], val_loss_ema=[],
                    grad_norm=[], step_norm=[], lr=[],
                    rho_trace=[], sigmas=sigmas_pb, pr_tail_start=pr_T0)
        val_ema = None
        print("=" * 78)
        print("TRAIN -- QFAN CF-MMD: per-block rho_b "
              "(chunked driver)")
        print(f"  p_theta={p_theta}  pf={bank.pf}  G={bank.G}  k={K_SHOTS}  "
              f"epochs={EPOCHS}  batch={BATCH}  n_freq={N_FREQ}  "
              f"n_rho={nb}")
        print(f"  lr={LR}  lr_enc={LR_ENC}  lr_rho={LR_RHO}  "
              f"gamma_risk={GAMMA_RISK}", flush=True)

    # ------------------- training loop -------------------
    t = t0_epoch
    while t < EPOCHS and (time.time() - t_start) < args.budget_s:
        batch_n = min(BATCH, train_pool.size)
        idx = prng.choice(train_pool, size=batch_n, replace=False)

        g_sum = np.zeros(p_theta)
        gA_sum = np.zeros_like(bank.A)
        gb_sum = np.zeros_like(bank.b)
        grho_vec = np.zeros(nb)
        gk_vec = np.zeros(nb)
        rho_now = sigmoid(r_logit)
        train_loss_sum = 0.0
        for bi, (start, bsz) in enumerate(blocks):
            S = sketch_cache[bi, idx].astype(np.float64, copy=False)
            Yb = Y_tr[idx, start:start + bsz].astype(np.float64, copy=False)
            Xc = Y_tr[idx, :start] if Omega_c_pb[bi] is not None else None
            k_b = float(np.exp(logk[bi])) if LEARN_K else K_SHOTS
            loss_b, gth, gA, gb, _, grho = cf_mmd_loss_and_grads(
                bank, S, Yb, theta, Omega_pb[bi], ridge_alpha,
                k_shots=k_b, gamma_risk=GAMMA_RISK, noise_weight=NOISE_WEIGHT,
                compute_grad=True, compute_encoding_grad=True,
                X_cond=Xc, Omega_cond=Omega_c_pb[bi],
                rho=float(rho_now[bi]))
            if LEARN_K:
                lk = []
                for sgn in (+1.0, -1.0):
                    lk.append(cf_mmd_loss_and_grads(
                        bank, S, Yb, theta, Omega_pb[bi], ridge_alpha,
                        k_shots=k_b * float(np.exp(sgn * K_FD)), gamma_risk=GAMMA_RISK,
                        noise_weight=NOISE_WEIGHT, compute_grad=False,
                        X_cond=Xc, Omega_cond=Omega_c_pb[bi],
                        rho=float(rho_now[bi]))[0])
                gk_vec[bi] = (lk[0] - lk[1]) / (2.0 * K_FD)
            g_sum += gth
            train_loss_sum += loss_b
            if grho is not None:
                grho_vec[bi] = grho
            gA_sum += gA
            gb_sum += gb

        sched = _scheduled_lr(t, EPOCHS, base_lr=1.0, warmup=LR_WARMUP,
                              decay="cosine", floor_frac=LR_FLOOR_FRAC)
        opt.lr = LR * sched
        step_vec = opt.step(_grad_clip(g_sum, GRAD_CLIP))
        theta = theta - step_vec
        bank.set_theta(theta)
        opt_enc.lr = LR_ENC * sched
        g_enc = _grad_clip(np.concatenate([gA_sum.ravel(), gb_sum]),
                           GRAD_CLIP)
        step_enc = opt_enc.step(g_enc)
        bank.A -= step_enc[:bank.L * bank.m].reshape(bank.L, bank.m)
        bank.b -= step_enc[bank.L * bank.m:]
        opt_rho.lr = LR_RHO * sched
        g_logit = grho_vec * rho_now * (1.0 - rho_now)   # chain rule
        r_logit = r_logit - opt_rho.step(g_logit)
        if LEARN_K:
            opt_k.lr = LR_K * sched
            logk = np.clip(logk - opt_k.step(gk_vec), *np.log(K_BOUNDS))

        if t >= pr_T0:
            pr_buffer.append(theta.copy())

        vl = total_val(bank, theta, blocks, sketch_cache, val_idx, Y_tr,
                       Omega_pb, Omega_c_pb, ridge_alpha, r_logit,
                       logk if LEARN_K else None)
        val_ema = vl if val_ema is None else \
            EMA_BETA * val_ema + (1 - EMA_BETA) * vl
        hist["train_loss"].append(float(train_loss_sum))
        hist["val_loss"].append(float(vl))
        hist["val_loss_ema"].append(float(val_ema))
        hist["grad_norm"].append(float(np.linalg.norm(g_sum)))
        hist["step_norm"].append(float(np.linalg.norm(step_vec)))
        hist["lr"].append(float(opt.lr))
        hist["rho_trace"].append(sigmoid(r_logit).copy())
        if (t + 1) % max(1, EPOCHS // 20) == 0 or t < 3:
            rr = sigmoid(r_logit)
            print(f"  epoch {t+1:3d}/{EPOCHS}  train={train_loss_sum:.6f}  "
                  f"val={vl:.6f}  ema={val_ema:.6f}  "
                  f"||g||={hist['grad_norm'][-1]:.4f}  lr={opt.lr:.4f}  "
                  f"rho_b[min/med/max]={rr.min():.2f}/"
                  f"{np.median(rr):.2f}/{rr.max():.2f}"
                  + (f"  k_b[min/med/max]={np.exp(logk).min():.0f}/"
                     f"{np.median(np.exp(logk)):.0f}/{np.exp(logk).max():.0f}"
                     if LEARN_K else ""), flush=True)
        t += 1

    if t < EPOCHS:
        with open(CKPT, "wb") as f:
            pickle.dump(dict(
                theta=theta, A=bank.A.copy(), b=bank.b.copy(),
                r_logit=r_logit.copy(),
                opt=(opt.m, opt.v, opt.t),
                opt_enc=(opt_enc.m, opt_enc.v, opt_enc.t),
                opt_rho=(opt_rho.m, opt_rho.v, opt_rho.t),
                logk=logk.copy(), opt_k=(opt_k.m, opt_k.v, opt_k.t),
                prng_state=prng.bit_generator.state,
                epoch=t, pr_buffer=pr_buffer, hist=hist,
                val_ema=val_ema, epochs_total=EPOCHS), f)
        print(f"[CHUNK DONE] epoch {t}/{EPOCHS} checkpointed "
              f"(wall {time.time()-t_start:.1f}s). Re-run to continue.",
              flush=True)
        return False

    # training complete: persist final state FIRST so any failure in
    # the evaluation/save phase resumes here instead of retraining.
    with open(CKPT, "wb") as f:
        pickle.dump(dict(
            theta=theta, A=bank.A.copy(), b=bank.b.copy(),
            r_logit=(r_logit.copy() if hasattr(r_logit, 'copy')
                     else r_logit),
            opt=(opt.m, opt.v, opt.t),
            opt_enc=(opt_enc.m, opt_enc.v, opt_enc.t),
            opt_rho=(opt_rho.m, opt_rho.v, opt_rho.t),
            logk=logk.copy(), opt_k=(opt_k.m, opt_k.v, opt_k.t),
            prng_state=prng.bit_generator.state,
            epoch=t, pr_buffer=pr_buffer, hist=hist,
            val_ema=val_ema, epochs_total=EPOCHS), f)
    # -------------------- eval + certificates --------------------
    theta_pr = (np.mean(np.stack(pr_buffer, axis=0), axis=0)
                if len(pr_buffer) > 1 else theta)
    bank.set_theta(theta_pr)
    vl_pr = total_val(bank, theta_pr, blocks, sketch_cache, val_idx, Y_tr,
                      Omega_pb, Omega_c_pb, ridge_alpha, r_logit, logk if LEARN_K else None)
    RHO_B = sigmoid(r_logit)                        # (nb,)
    hist["theta_final"] = theta_pr.copy()
    hist["val_loss_pr"] = float(vl_pr)
    hist["rho_final"] = RHO_B.copy()
    print("-" * 78)
    print(f"[DONE] val first/last: {hist['val_loss'][0]:.6f} -> "
          f"{hist['val_loss'][-1]:.6f}   PR-avg: {vl_pr:.6f}")
    with np.printoptions(precision=3, suppress=True):
        print(f"[MODEL] learned rho_b = {RHO_B}")
    cost = np.mean([1 + (bsz - 1) * (1 - RHO_B[bi])
                    for bi, (s, bsz) in enumerate(blocks)])
    print(f"[MODEL] mean record cost x{cost:.2f}", flush=True)

    out_dir = PROJECT_ROOT / "outputs"
    np.savez_compressed(
        out_dir / f"loss_{TAG}.npz",
        **{k: np.asarray(v) for k, v in hist.items()
           if isinstance(v, list) and k != "sigmas"},
        theta_init=theta_init0, theta_final=hist["theta_final"],
        val_loss_pr=hist["val_loss_pr"], rho_final=RHO_B)

    K_GEN = (np.maximum(1, np.rint(np.exp(logk))).astype(int) if LEARN_K else K_SHOTS)
    _ICPT = bool(getattr(args, 'decoder_intercept', False))
    rng = np.random.default_rng(seed)
    models = fit_all_blocks_born(bank, sketch_cache, Y_tr, blocks,
                                 ridge_alpha, k_shots=K_GEN, intercept=_ICPT, noise_weight=NOISE_WEIGHT)

    def metrics_of(Y_ref, Y_gen):
        w1 = float(np.mean([wasserstein_distance(Y_ref[:, j], Y_gen[:, j])
                            for j in range(d)]))
        cs = correlation_error_summary(
            corr_nan_safe(Y_ref), corr_nan_safe(Y_gen), blocks)
        sp = correlation_error_summary(
            spearman_corr(Y_ref), spearman_corr(Y_gen), blocks)
        cs = dict(w1=w1, **cs)
        cs["sp_offdiag"] = sp["corr_mae_offdiag"]
        cs["sp_within"] = sp["corr_mae_within"]
        cs["sp_cross"] = sp["corr_mae_cross"]
        return cs

    def fmt(tag, m):
        return (f"  {tag:34s} W1={m['w1']:.5f}  "
                f"off={m['corr_mae_offdiag']:.4f}  "
                f"win={m['corr_mae_within']:.4f}  "
                f"crs={m['corr_mae_cross']:.4f}  |  "
                f"Sp off={m['sp_offdiag']:.4f}  "
                f"win={m['sp_within']:.4f}  crs={m['sp_cross']:.4f}")

    print("=" * 78)
    print(f"GENERATION + CERTIFICATES (k={K_GEN}, per-block rho_b)",
          flush=True)
    # raw Born sampler
    Y_gen = sample_progressive_born(bank, models, d, blocks, sketcher,
                                    n_samples=len(Y_te), rng_np=rng,
                                    k=K_GEN, record_share=RHO_B)
    m_raw = metrics_of(Y_te, Y_gen)

    # (ii) train-side reference for the marginal map (deterministic)
    Y_ref_gen = sample_progressive_born(bank, models, d, blocks, sketcher,
                                        n_samples=len(Y_tr),
                                        rng_np=np.random.default_rng(999),
                                        k=K_GEN, record_share=RHO_B)
    T = MonotoneMarginalMap().fit(Y_ref_gen, Y_tr)
    Y_genT = T.transform(Y_gen)
    m_head = metrics_of(Y_te, Y_genT)

    prng2 = np.random.default_rng(seed + 777)
    Y_mean = sample_progressive_born(bank, models, d, blocks, sketcher,
                                     n_samples=len(Y_te), rng_np=prng2,
                                     mode="mean")
    m_c1 = metrics_of(Y_te, Y_mean)

    # C2 classical surrogate -- raw AND with its OWN marginal map
    Y_cls = sample_progressive_born(bank, models, d, blocks, sketcher,
                                    n_samples=len(Y_te), rng_np=prng2,
                                    k=K_GEN, mode="classical")
    m_c2 = metrics_of(Y_te, Y_cls)
    Y_cls_ref = sample_progressive_born(bank, models, d, blocks, sketcher,
                                        n_samples=len(Y_tr),
                                        rng_np=np.random.default_rng(998),
                                        k=K_GEN, mode="classical")
    T_cls = MonotoneMarginalMap().fit(Y_cls_ref, Y_tr)
    Y_clsT = T_cls.transform(Y_cls)
    m_c2T = metrics_of(Y_te, Y_clsT)

    bank_scr = StatevectorBornBank(cfg.sketch.sketch_dim, spec,
                                   seed=seed + 13)
    bank_scr.set_theta(prng2.normal(0, np.pi / 4, size=bank.n_var))
    bank_scr.A = A_init0 * prng2.normal(1.0, 0.5, size=A_init0.shape)
    bank_scr.b = b_init0 + 0.05 * prng2.normal(size=b_init0.shape)
    models_scr = fit_all_blocks_born(bank_scr, sketch_cache, Y_tr, blocks,
                                     ridge_alpha, k_shots=K_GEN, intercept=_ICPT, noise_weight=NOISE_WEIGHT)
    Y_scr = sample_progressive_born(bank_scr, models_scr, d, blocks,
                                    sketcher, n_samples=len(Y_te),
                                    rng_np=prng2, k=K_GEN,
                                    record_share=RHO_B)
    m_c3 = metrics_of(Y_te, Y_scr)

    bank_ini = StatevectorBornBank(cfg.sketch.sketch_dim, spec,
                                   seed=seed + 13)
    models_ini = fit_all_blocks_born(bank_ini, sketch_cache, Y_tr, blocks,
                                     ridge_alpha, k_shots=K_GEN, intercept=_ICPT, noise_weight=NOISE_WEIGHT)
    Y_ini = sample_progressive_born(bank_ini, models_ini, d, blocks,
                                    sketcher, n_samples=len(Y_te),
                                    rng_np=prng2, k=K_GEN,
                                    record_share=RHO_B)
    m_c4 = metrics_of(Y_te, Y_ini)

    cop = EmpiricalGaussianCopulaCalibrator(
        shrink=cfg.copula.shrink, eps=cfg.copula.eps).fit(Y_ref_gen, Y_tr)
    Y_cop = cop.transform(Y_gen)
    m_c5 = metrics_of(Y_te, Y_cop)

    k_obs = d // 2
    Y_pred = predict_future_born(bank, models, d, blocks, sketcher,
                                 Y_te[:, :k_obs])
    mse_mean = float(((Y_pred[:, k_obs:] - Y_te[:, k_obs:]) ** 2).mean())
    w1_floor = float(np.mean([wasserstein_distance(Y_tr[:, j], Y_te[:, j])
                              for j in range(d)]))

    print("=" * 78)
    print("[RESULTS]  Pearson corr MAE | Spearman corr MAE "
          "(monotone-invariant)")
    print(f"  {'W1 train-vs-test floor':34s} W1={w1_floor:.5f}")
    print(fmt(" HEADLINE (Born + marg. map)", m_head))
    print(fmt(" raw Born (no map)", m_raw))
    print(fmt("C1 Born noise removed (mean)", m_c1))
    print(fmt("C2  classical surrogate raw", m_c2))
    print(fmt("C2T classical + own marg. map", m_c2T))
    print(fmt("C3 angles scrambled (+refit)", m_c3))
    print(fmt("C4 untrained bank (+refit)", m_c4))
    print(fmt("C5 + copula (diagnostic)", m_c5))
    print(f"  pred mean MSE (k_obs={k_obs}): {mse_mean:.6g}")
    print("=" * 78)

    np.savez_compressed(
        out_dir / f"model_{TAG}.npz",
        Y_tr=Y_tr, Y_te=Y_te, Y_gen=Y_gen, Y_genT=Y_genT,
        Y_mean=Y_mean, Y_cls=Y_cls, Y_clsT=Y_clsT,
        Y_scr=Y_scr, Y_ini=Y_ini, Y_cop=Y_cop, Y_pred=Y_pred,
        C_test=corr_nan_safe(Y_te), C_gen=corr_nan_safe(Y_genT),
        theta_init=theta_init0, theta_final=hist["theta_final"],
        A_final=bank.A, b_final=bank.b, rho_final=RHO_B,
        w1_floor=w1_floor, mse_mean=mse_mean, k_obs=k_obs,
        meta=np.array([dict(
            k_shots=K_SHOTS, k_gen=K_GEN, block_size=BLOCK_SIZE,
            blocks=[(int(a), int(n)) for a, n in blocks],
            layer_widths=PHYS_LAYER_WIDTHS,
            decoder_intercept=_ICPT,
            seed=int(cfg.data.seed),
            noise_weight=NOISE_WEIGHT,
            data_path=str(getattr(cfg.data, 'data_path', '')),
            record_share=RHO_B, epochs=EPOCHS, gamma_risk=GAMMA_RISK,
            depth=spec.depth,
            headline=m_head, raw=m_raw, c1_mean=m_c1,
            c2_classical=m_c2, c2_classicalT=m_c2T,
            c3_scrambled=m_c3, c4_untrained=m_c4, c5_copula=m_c5,
            w1_floor=w1_floor, mse_mean=mse_mean,
            val_loss_first=hist["val_loss"][0],
            val_loss_last=hist["val_loss"][-1],
            val_loss_pr=hist["val_loss_pr"])], dtype=object))
    print(f"[SAVE] {out_dir / f'model_{TAG}.npz'}")
    print("[TRAINING COMPLETE]")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Train QFAN at a given image size.")
    ap.add_argument("--d", type=int, choices=[12, 25],
                    help="one of the two paper configurations")
    g = ap.add_argument_group(
        "generic geometry",
        "train on any (N, d) dataset, for instance one built by "
        "ds1/prepare_ds1.py. Use instead of --d.")
    g.add_argument("--data", help="path to an (N, d) .npy file")
    g.add_argument("--nq", type=int, default=3, help="register width")
    g.add_argument("--block-size", type=int, default=2, help="pixels per block")
    g.add_argument("--depth", type=int, default=3, help="circuit layers")
    g.add_argument("--angle-dim", type=int, default=None,
                   help="encoding angles per layer, default max(16, 2 n_q)")
    g.add_argument("--tag", default=None, help="name for outputs")
    g.add_argument("--learn-k", action="store_true",
                   help="each block learns its own number of measurement records; "
                        "outputs get _lk")
    g.add_argument("--seed", type=int, default=None,
                   help="seed for the split, the initialisation and the sketch; "
                        "outputs get a _s<seed> suffix")
    g.add_argument("--decoder-intercept", action="store_true",
                   help="give each block decoder an intercept so the generated "
                        "mean matches the data (recommended for Dataset 1)")
    ap.add_argument("--loop", action="store_true",
                    help="keep running chunks until training completes")
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="wall-clock ceiling for --loop")
    ap.add_argument("--budget-s", type=float, default=600.0,
                    help="seconds per chunk")
    ap.add_argument("--epochs", type=int, default=300,
                    help="total training epochs; fixed at run start and "
                         "stored in the checkpoint")
    args = ap.parse_args()
    if args.data:
        # resolve now: a relative path would otherwise be joined onto the
        # data directory inside build_problem and point at data/data/...
        import pathlib as _pl
        args.data = str(_pl.Path(args.data).expanduser().resolve())
        if args.angle_dim is None:
            args.angle_dim = max(16, 2 * args.nq)
        if args.tag is None:
            import pathlib as _pl
            args.tag = (f"{_pl.Path(args.data).stem}_nq{args.nq}_b{args.block_size}"
                        + ("_lk" if args.learn_k else "")
                        + ("" if args.seed is None else f"_s{args.seed}"))
    elif args.d is None:
        ap.error("give either --d 12/25 or --data PATH")

    if not args.loop:
        _run_one_chunk(args)
        return

    # --loop: keep starting fresh chunks until training reports completion.
    # Each chunk resumes from the checkpoint the previous one wrote, so this
    # is exactly equivalent to re-running the command by hand, just without
    # the typing.
    t_wall = time.time()
    n = 0
    while True:
        n += 1
        if _run_one_chunk(args):
            print(f"[LOOP] finished after {n} chunk(s), "
                  f"{(time.time()-t_wall)/60:.1f} min total.")
            return
        elapsed_h = (time.time() - t_wall) / 3600.0
        if elapsed_h > args.max_hours:
            print(f"[LOOP] stopped at the {args.max_hours} h ceiling after "
                  f"{n} chunk(s). Re-run to continue.")
            return
        print(f"[LOOP] chunk {n} done, {elapsed_h:.2f} h elapsed, "
              f"continuing.", flush=True)


if __name__ == "__main__":
    main()
