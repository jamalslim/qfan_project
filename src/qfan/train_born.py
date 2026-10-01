"""
Exact-gradient training of QFAN .

Single entry point: train_theta_born_cf -- exact characteristic-function
MMD training of the k-shot Born pushforward (conditional/joint matching by
default), co-training theta and the re-uploading encoding (A, b) with the
same optimizer stack as  (Adam + grad clip + warmup + cosine decay +
Polyak-Ruppert tail averaging + held-out validation monitor).

Consequence: minimizing this loss forces the circuit to match the data's
conditional dispersion and within-block correlation with its own
measurement statistics. There is no residual bootstrap and no copula
downstream to absorb what the circuit fails to learn.
"""

from typing import Optional

import numpy as np

from .born import (StatevectorBornBank,
                   cf_mmd_loss_and_grads, draw_rff_frequencies,
                   draw_joint_rff_frequencies)
from .mmd import median_sigma
from .train import AdamOpt, _scheduled_lr, _grad_clip


def train_theta_born_cf(bank: StatevectorBornBank,
                        sketch_cache: np.ndarray,
                        Y_train: np.ndarray,
                        blocks,
                        ridge_alpha: float,
                        k_shots: int = 64,
                        epochs: int = 150,
                        batch_size: int = 256,
                        val_batch_size: int = 512,
                        lr: float = 0.1,
                        lr_encoding: Optional[float] = None,
                        train_encoding: bool = True,
                        beta1: float = 0.9, beta2: float = 0.999,
                        eps: float = 1e-8,
                        grad_clip: float = 5.0,
                        lr_warmup: int = 5,
                        lr_decay: str = "cosine",
                        lr_floor_frac: float = 0.1,
                        ema_beta: float = 0.9,
                        pr_tail_frac: float = 0.3,
                        gamma_risk: float = 10.0,
                        train_rho: bool = False,
                        rho_init: float = 0.9,
                        lr_rho: float = 0.05,
                        sigma_mults=(0.5, 1.0, 2.0, 4.0),
                        n_freq: int = 768,
                        cond_frac: float = 0.5,
                        joint_matching: bool = True,
                        seed: int = 7,
                        verbose: bool = True) -> dict:
    """
    Exact characteristic-function MMD training of the k-shot-averaged Born
    pushforward (see qfan.born header for the CF identity). Co-trains
    theta and, by default, the re-uploading encoding (A, b). All gradients
    exact (T10/T11/T13); the RFF frequency set is drawn once at init and
    fixed, so the objective is deterministic and stationary.

    joint_matching=True (default) matches the JOINT law of (teacher-forced
    prefix, block) rather than the block marginal -- this is what trains
    the cross-block (anti)correlations into the conditional response; see
    cf_mmd_loss_and_grads docstring.

    train_rho=True (block size 2) additionally LEARNS the record-partition
    fraction rho -- the model's intra-block noise correlation -- by exact
    gradient descent on the same objective (logit-parameterized to stay in
    (0,1)). No post-training calibration of any kind remains: the model
    trains on the data and generates.
    """
    prng = np.random.default_rng(seed + 999)
    theta = bank.get_theta().copy()
    p_theta = theta.size
    nb = len(blocks)
    n_train = Y_train.shape[0]

    val_n = min(int(val_batch_size), n_train)
    val_idx = prng.choice(n_train, size=val_n, replace=False)
    train_mask = np.ones(n_train, dtype=bool)
    train_mask[val_idx] = False
    train_pool = np.where(train_mask)[0]

    sigmas_per_block, Omega_per_block, Omega_cond_per_block = [], [], []
    for bi, (start, bsz) in enumerate(blocks):
        Yb = Y_train[val_idx, start:start + bsz]
        s_star = median_sigma(Yb, Yb, cap=256, seed=seed)
        sigs = [mfac * s_star for mfac in sigma_mults]
        sigmas_per_block.append(sigs)
        d_c = start if joint_matching else 0
        if d_c > 0:
            Xc = Y_train[val_idx, :start]
            sc_star = median_sigma(Xc, Xc, cap=256, seed=seed)
            sigs_c = [mfac * sc_star for mfac in sigma_mults]
            Oy, Oc = draw_joint_rff_frequencies(
                sigs, bsz, sigs_c, d_c, n_freq, cond_frac=cond_frac,
                seed=seed + 31 * bi)
        else:
            Oy = draw_rff_frequencies(sigs, bsz, n_freq, seed=seed + 31 * bi)
            Oc = None
        Omega_per_block.append(Oy)
        Omega_cond_per_block.append(Oc)

    if verbose:
        print("=" * 78)
        print("TRAIN -- QFAN CF-MMD: exact k-shot pushforward matching")
        print(f"  p_theta={p_theta}  pf={bank.pf}  G={bank.G}  k={k_shots}  "
              f"epochs={epochs}  batch={batch_size}  n_freq={n_freq}")
        print(f"  lr={lr}  lr_enc={lr_encoding}  train_encoding={train_encoding}  "
              f"gamma_risk={gamma_risk}")
        print("=" * 78)

    opt = AdamOpt(p_theta, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    r_logit = float(np.log(rho_init / (1.0 - rho_init)))
    opt_rho = AdamOpt(1, lr=lr_rho, beta1=beta1, beta2=beta2, eps=eps)
    lr_enc = float(lr_encoding) if lr_encoding is not None else 0.25 * lr
    n_enc = bank.L * bank.m + bank.L
    opt_enc = AdamOpt(n_enc, lr=lr_enc, beta1=beta1, beta2=beta2, eps=eps)
    pr_T0 = int(np.floor((1.0 - pr_tail_frac) * epochs))
    pr_buffer = []
    hist = dict(train_loss=[], val_loss=[], val_loss_ema=[],
                grad_norm=[], step_norm=[], lr=[],
                sigmas=sigmas_per_block, pr_tail_start=pr_T0)
    val_ema = None

    def total_val(th):
        v = 0.0
        for bi in range(nb):
            start, bsz = blocks[bi]
            S_v = sketch_cache[bi, val_idx].astype(np.float64, copy=False)
            Y_v = Y_train[val_idx, start:start + bsz].astype(np.float64,
                                                             copy=False)
            Xc = (Y_train[val_idx, :start]
                  if Omega_cond_per_block[bi] is not None else None)
            v += cf_mmd_loss_and_grads(
                bank, S_v, Y_v, th, Omega_per_block[bi], ridge_alpha,
                k_shots=k_shots, gamma_risk=gamma_risk,
                compute_grad=False,
                X_cond=Xc, Omega_cond=Omega_cond_per_block[bi],
                rho=(1.0 / (1.0 + np.exp(-r_logit)) if train_rho
                     else None))[0]
        return v

    for t in range(epochs):
        batch_n = min(int(batch_size), train_pool.size)
        idx = prng.choice(train_pool, size=batch_n, replace=False)

        g_sum = np.zeros(p_theta)
        gA_sum = np.zeros_like(bank.A)
        gb_sum = np.zeros_like(bank.b)
        grho_sum = 0.0
        rho_now = 1.0 / (1.0 + np.exp(-r_logit))
        train_loss_sum = 0.0
        for bi in range(nb):
            start, bsz = blocks[bi]
            S = sketch_cache[bi, idx].astype(np.float64, copy=False)
            Yb = Y_train[idx, start:start + bsz].astype(np.float64,
                                                        copy=False)
            Xc = (Y_train[idx, :start]
                  if Omega_cond_per_block[bi] is not None else None)
            loss_b, gth, gA, gb, _, grho = cf_mmd_loss_and_grads(
                bank, S, Yb, theta, Omega_per_block[bi], ridge_alpha,
                k_shots=k_shots, gamma_risk=gamma_risk,
                compute_grad=True, compute_encoding_grad=train_encoding,
                X_cond=Xc, Omega_cond=Omega_cond_per_block[bi],
                rho=(rho_now if train_rho else None))
            g_sum += gth
            train_loss_sum += loss_b
            if train_rho and grho is not None:
                grho_sum += grho
            if train_encoding:
                gA_sum += gA
                gb_sum += gb

        sched = _scheduled_lr(t, epochs, base_lr=1.0, warmup=lr_warmup,
                              decay=lr_decay, floor_frac=lr_floor_frac)
        opt.lr = lr * sched
        step_vec = opt.step(_grad_clip(g_sum, grad_clip))
        theta = theta - step_vec
        bank.set_theta(theta)
        if train_encoding:
            opt_enc.lr = lr_enc * sched
            g_enc = _grad_clip(np.concatenate([gA_sum.ravel(), gb_sum]),
                               grad_clip)
            step_enc = opt_enc.step(g_enc)
            bank.A -= step_enc[:bank.L * bank.m].reshape(bank.L, bank.m)
            bank.b -= step_enc[bank.L * bank.m:]
        if train_rho:
            opt_rho.lr = lr_rho * sched
            g_logit = grho_sum * rho_now * (1.0 - rho_now)
            r_logit -= float(opt_rho.step(np.array([g_logit]))[0])

        if t >= pr_T0:
            pr_buffer.append(theta.copy())

        vl = total_val(theta)
        val_ema = vl if val_ema is None else \
            ema_beta * val_ema + (1 - ema_beta) * vl
        hist["train_loss"].append(float(train_loss_sum))
        hist["val_loss"].append(float(vl))
        hist["val_loss_ema"].append(float(val_ema))
        hist["grad_norm"].append(float(np.linalg.norm(g_sum)))
        hist["step_norm"].append(float(np.linalg.norm(step_vec)))
        hist["lr"].append(float(opt.lr))
        if verbose and ((t + 1) % max(1, epochs // 20) == 0 or t < 3):
            extra = (f"  rho={1.0/(1.0+np.exp(-r_logit)):.3f}"
                     if train_rho else "")
            print(f"  epoch {t+1:3d}/{epochs}  train={train_loss_sum:.6f}  "
                  f"val={vl:.6f}  ema={val_ema:.6f}  "
                  f"||g||={hist['grad_norm'][-1]:.4f}  lr={opt.lr:.4f}"
                  + extra)

    theta_pr = (np.mean(np.stack(pr_buffer, axis=0), axis=0)
                if len(pr_buffer) > 1 else theta)
    bank.set_theta(theta_pr)
    vl_pr = total_val(theta_pr)
    hist["theta_final"] = theta_pr.copy()
    hist["theta_last_iter"] = theta.copy()
    hist["val_loss_pr"] = float(vl_pr)
    hist["A_final"] = bank.A.copy()
    hist["b_final"] = bank.b.copy()
    hist["rho_final"] = (float(1.0 / (1.0 + np.exp(-r_logit)))
                         if train_rho else 1.0)
    if verbose:
        print("-" * 78)
        print(f"[DONE] val first/last/min: {hist['val_loss'][0]:.6f} -> "
              f"{hist['val_loss'][-1]:.6f} (min {min(hist['val_loss']):.6f})"
              f"   PR-avg: {vl_pr:.6f}")
    return hist
