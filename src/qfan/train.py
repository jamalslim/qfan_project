"""
EXACT parameter-shift training for QFAN -- the core novelty of v3.

Jamal Slim
jamal.slim@desy.de




Mathematical foundation - please ask in case of confusion
=======================
Loss:
    L(theta) = sum_{beta=1}^B sum_{sigma in Sigma_beta}
                  MMD^2_sigma( F_beta(theta) W_beta(theta), Y_beta )

where W_beta = A_beta F_beta^T Y_beta is the closed-form ridge solution and
A_beta = (F_beta^T F_beta + alpha I)^{-1}. The Sigma_beta bandwidth family
is FIXED at initialization (see mmd.multi_kernel_mmd2_and_grad_Yhat header).

Full analytic chain rule
========================
With G_k = dF/d theta_k computed by parameter-shift (quantum.py), and
R_beta = Y_beta - F_beta W_beta :

    dW_beta / d theta_k  =  A_beta [ G_k^T R_beta  -  F_beta^T G_k W_beta ]      (Eq. A)
    dY_hat / d theta_k   =  G_k W_beta  +  F_beta (dW_beta / d theta_k)          (Eq. B)

    dL / d theta_k
        =  < dL/dY_hat . W^T ,  G_k >_F                                          (direct)
        +  < F^T . dL/dY_hat , dW/dtheta_k >_F                                   (ridge-implicit)

The analytic MMD gradient dL/dY_hat is in mmd.multi_kernel_mmd2_and_grad_Yhat.
Both terms are verified numerically to relative error ~1e-8 against central
finite differences.

Optional Gauss-Newton preconditioning
=====================================
Using the same J_Y_k = dY_hat / d theta_k already computed above,
    H_GN = sum_k J_Y_k . J_Y_k^T    (p_theta x p_theta, trivial to invert)
    P = (H_GN + damp * trace(H_GN)/p_theta * I)^{-1}
    theta <- theta - eta * P @ grad

This is distinct from the quantum natural gradient (Stokes 2020, Gacon 2021):
QFI is geometry-intrinsic, whereas H_GN is TASK-adapted to the QFAN readout.
For p_theta in {12, 18} this is a 144 or 324 entry matrix with no extra
circuits beyond what param-shift already computes.

Other non-quantum optimization hygiene
======================================
  - Adam (Kingma & Ba 2015): standard moving-average of g and g^2.
  - Linear LR warmup for the first `lr_warmup` steps.
  - Cosine LR decay from post-warmup to lr * lr_floor_frac.
  - EMA tracking of val_loss (monitor smoothing, does NOT affect training).
  - Polyak-Ruppert averaging of the last pr_tail_frac fraction of iterates.
"""

from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .mmd import multi_kernel_mmd2, multi_kernel_mmd2_and_grad_Yhat, median_sigma
from .ridge import ridge_fit, ridge_inverse_matrix
from .quantum import (
    TrainableShotPauliBank,
    parameter_shift_features_and_jacobian,
    compute_features_chunked,
)


# =========================================================================
# Exact gradient of L w.r.t. theta
# =========================================================================

def exact_loss_and_grad_theta(bank: TrainableShotPauliBank,
                              S: np.ndarray,
                              Y_true: np.ndarray,
                              theta: np.ndarray,
                              sigmas,
                              ridge_alpha: float,
                              return_jacobian_Y: bool = False,
                              chunk_size: int = 256):
    """See module docstring for the derivation. Returns (loss, grad, J_Y)."""
    S = np.asarray(S, np.float64); Y_true = np.asarray(Y_true, np.float64)
    theta = np.asarray(theta, np.float64).ravel()

    F, J = parameter_shift_features_and_jacobian(
        bank, S, theta, chunk_size=chunk_size)
    p_theta, n, p_f = J.shape
    b_out = Y_true.shape[1]

    A  = ridge_inverse_matrix(F, ridge_alpha)
    W  = A @ (F.T @ Y_true)
    Yhat = F @ W
    R    = Y_true - Yhat

    loss, dL_dYhat = multi_kernel_mmd2_and_grad_Yhat(Yhat, Y_true, sigmas)

    aYWt = dL_dYhat @ W.T
    FtaY = F.T @ dL_dYhat

    grad = np.zeros(p_theta, dtype=np.float64)
    J_Y = np.zeros((p_theta, n, b_out), dtype=np.float64) if return_jacobian_Y else None

    for k in range(p_theta):
        Gk    = J[k]
        GkTR  = Gk.T @ R
        FtGk  = F.T @ Gk
        FtGkW = FtGk @ W
        dW_k  = A @ (GkTR - FtGkW)
        grad[k] = float(np.sum(aYWt * Gk)) + float(np.sum(FtaY * dW_k))
        if return_jacobian_Y:
            J_Y[k] = Gk @ W + F @ dW_k

    return float(loss), grad, J_Y


# =========================================================================
# Adam + grad-clip + warmup + cosine decay + GN preconditioner
# =========================================================================

class AdamOpt:
    def __init__(self, n_params: int, lr: float = 0.05,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr); self.b1 = float(beta1); self.b2 = float(beta2); self.eps = float(eps)
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)
        self.t = 0

    def step(self, g: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.v = self.b2 * self.v + (1.0 - self.b2) * (g * g)
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _scheduled_lr(step: int, total_steps: int,
                  base_lr: float, warmup: int,
                  decay: str = "cosine", floor_frac: float = 0.1) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    if decay == "cosine" and total_steps > warmup:
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        progress = min(1.0, max(0.0, progress))
        cos_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return base_lr * (floor_frac + (1.0 - floor_frac) * cos_factor)
    return base_lr


def _grad_clip(g: np.ndarray, clip: float) -> np.ndarray:
    n = np.linalg.norm(g)
    if n > clip:
        return g * (clip / n)
    return g


def _gauss_newton_preconditioner(J_Y_flat: np.ndarray, damp: float = 1e-2) -> np.ndarray:
    p_theta = J_Y_flat.shape[0]
    H = J_Y_flat @ J_Y_flat.T
    dd = damp * max(1e-12, np.trace(H) / max(1, p_theta))
    H += dd * np.eye(p_theta)
    try:
        c, low = cho_factor(H, lower=True, check_finite=False)
        P = cho_solve((c, low), np.eye(p_theta), check_finite=False)
    except Exception:
        P = np.linalg.pinv(H)
    return P


# =========================================================================
# Top-level training loops
# =========================================================================

def train_theta_exact(bank: TrainableShotPauliBank,
                      sketch_cache: np.ndarray,
                      Y_train: np.ndarray,
                      blocks,
                      ridge_alpha: float,
                      epochs: int = 60,
                      batch_size: int = 128,
                      val_batch_size: int = 512,
                      lr: float = 0.05,
                      beta1: float = 0.9,
                      beta2: float = 0.999,
                      eps: float = 1e-8,
                      grad_clip: float = 5.0,
                      lr_warmup: int = 5,
                      lr_decay: str = "cosine",
                      lr_floor_frac: float = 0.1,
                      ema_beta: float = 0.9,
                      pr_tail_frac: float = 0.5,
                      use_precond: bool = False,
                      precond_damp: float = 1e-2,
                      precond_every: int = 3,
                      all_blocks: bool = True,
                      block_subset: Optional[int] = None,
                      sigma_mults=(0.5, 1.0, 2.0),
                      feature_chunk_size: int = 256,
                      seed: int = 7,
                      verbose: bool = True) -> dict:
    """
    EXACT parameter-shift training. See module docstring for derivation.

    Returns a `hist` dict with:
        train_loss, val_loss, val_loss_ema,
        grad_norm, step_norm, lr,
        sigmas,          # per-block bandwidth families (fixed at init)
        theta_final,     # Polyak-Ruppert-averaged (the deployed theta)
        theta_last_iter, # last SGD iterate
        val_loss_pr,     # val loss evaluated at theta_final
        pr_tail_start,
    """
    prng = np.random.default_rng(seed + 999)
    theta = bank.get_theta().copy()
    p_theta = theta.size
    nb = len(blocks)
    n_train = Y_train.shape[0]

    val_n = min(int(val_batch_size), n_train)
    val_idx = prng.choice(n_train, size=val_n, replace=False)
    train_mask = np.ones(n_train, dtype=bool); train_mask[val_idx] = False
    train_pool = np.where(train_mask)[0]

    sigmas_per_block = []
    for (start, bsz) in blocks:
        Y_blk_val = Y_train[val_idx, start:start + bsz]
        sig_star = median_sigma(Y_blk_val, Y_blk_val, cap=256, seed=seed)
        sigmas_per_block.append([m * sig_star for m in sigma_mults])

    if verbose:
        print("=" * 78)
        print("TRAIN THETA  --  EXACT param-shift + multi-kernel MMD + Adam + PR avg")
        print(f"  p_theta={p_theta}  epochs={epochs}  batch={batch_size}  val={val_n}")
        print(f"  lr={lr}  warmup={lr_warmup}  decay={lr_decay} (floor={lr_floor_frac})")
        print(f"  ema_beta={ema_beta}  pr_tail_frac={pr_tail_frac}")
        print(f"  precond={use_precond}  damp={precond_damp}  every={precond_every}")
        print(f"  all_blocks={all_blocks}  block_subset={block_subset}  clip={grad_clip}")
        print(f"  sigma_mults={sigma_mults}")
        for bi, sigs in enumerate(sigmas_per_block):
            print(f"    block {bi:2d}  sigmas = {[f'{s:.4f}' for s in sigs]}")
        print("=" * 78)

    opt = AdamOpt(p_theta, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    pr_T0 = int(np.floor((1.0 - pr_tail_frac) * epochs))
    pr_buffer = []

    P_cache = np.eye(p_theta)
    P_age = 0

    hist = dict(
        train_loss=[], val_loss=[], val_loss_ema=[],
        grad_norm=[], step_norm=[], lr=[],
        theta_snapshots=[], sigmas=sigmas_per_block,
        pr_tail_start=pr_T0,
    )
    val_ema = None

    for t in range(epochs):
        if all_blocks:
            block_indices = list(range(nb))
        elif block_subset is not None and block_subset < nb:
            block_indices = prng.choice(nb, size=int(block_subset), replace=False).tolist()
        else:
            block_indices = [int(prng.integers(0, nb))]

        batch_n = min(int(batch_size), train_pool.size)
        train_idx = prng.choice(train_pool, size=batch_n, replace=False)

        g_sum = np.zeros(p_theta, dtype=np.float64)
        train_loss_sum = 0.0
        J_Y_stack = []

        for bi in block_indices:
            start, bsz = blocks[bi]
            S = sketch_cache[bi, train_idx].astype(np.float64, copy=False)
            Y_true = Y_train[train_idx, start:start + bsz].astype(np.float64, copy=False)
            sigmas = sigmas_per_block[bi]

            loss_b, grad_b, J_Y_b = exact_loss_and_grad_theta(
                bank, S, Y_true, theta, sigmas, ridge_alpha,
                return_jacobian_Y=use_precond,
                chunk_size=feature_chunk_size,
            )
            g_sum += grad_b
            train_loss_sum += loss_b
            if use_precond and J_Y_b is not None:
                J_Y_stack.append(J_Y_b.reshape(p_theta, -1))

        if use_precond and len(J_Y_stack) > 0 and (P_age % precond_every == 0):
            JY_flat = np.concatenate(J_Y_stack, axis=1)
            P_cache = _gauss_newton_preconditioner(JY_flat, damp=precond_damp)
        P_age += 1

        g_precond = (P_cache @ g_sum) if use_precond else g_sum
        g_precond = _grad_clip(g_precond, grad_clip)

        opt.lr = _scheduled_lr(
            t, epochs, base_lr=lr, warmup=lr_warmup,
            decay=lr_decay, floor_frac=lr_floor_frac,
        )
        step_vec = opt.step(g_precond)
        theta = theta - step_vec
        bank.set_theta(theta)

        grad_norm = float(np.linalg.norm(g_sum))
        step_norm = float(np.linalg.norm(step_vec))

        if t >= pr_T0:
            pr_buffer.append(theta.copy())

        val_loss_sum = 0.0
        for bi in range(nb):
            start, bsz = blocks[bi]
            S_v = sketch_cache[bi, val_idx].astype(np.float64, copy=False)
            Y_v = Y_train[val_idx, start:start + bsz].astype(np.float64, copy=False)
            F_v = compute_features_chunked(
                bank, S_v, theta_override=theta, chunk_size=feature_chunk_size)
            W_v, _ = ridge_fit(F_v, Y_v, alpha=ridge_alpha)
            Yhat_v = F_v @ W_v
            val_loss_sum += multi_kernel_mmd2(Yhat_v, Y_v, sigmas_per_block[bi])

        if val_ema is None:
            val_ema = val_loss_sum
        else:
            val_ema = ema_beta * val_ema + (1.0 - ema_beta) * val_loss_sum

        hist["train_loss"].append(float(train_loss_sum))
        hist["val_loss"].append(float(val_loss_sum))
        hist["val_loss_ema"].append(float(val_ema))
        hist["grad_norm"].append(grad_norm)
        hist["step_norm"].append(step_norm)
        hist["lr"].append(float(opt.lr))
        if (t + 1) % max(1, epochs // 10) == 0 or t < 3:
            hist["theta_snapshots"].append((t, theta.copy()))

        if verbose:
            print(f"  epoch {t+1:3d}/{epochs}  train={train_loss_sum:.6f}  "
                  f"val={val_loss_sum:.6f}  ema={val_ema:.6f}  "
                  f"||g||={grad_norm:.4f}  ||step||={step_norm:.4f}  "
                  f"lr={opt.lr:.4f}")

    theta_pr = np.mean(np.stack(pr_buffer, axis=0), axis=0) if len(pr_buffer) > 1 else theta
    bank.set_theta(theta_pr)

    vl_pr = 0.0
    for bi in range(nb):
        start, bsz = blocks[bi]
        S_v = sketch_cache[bi, val_idx].astype(np.float64, copy=False)
        Y_v = Y_train[val_idx, start:start + bsz].astype(np.float64, copy=False)
        F_v = compute_features_chunked(
            bank, S_v, theta_override=theta_pr, chunk_size=feature_chunk_size)
        W_v, _ = ridge_fit(F_v, Y_v, alpha=ridge_alpha)
        Yhat_v = F_v @ W_v
        vl_pr += multi_kernel_mmd2(Yhat_v, Y_v, sigmas_per_block[bi])

    hist["theta_final"]     = theta_pr.copy()
    hist["theta_last_iter"] = theta.copy()
    hist["val_loss_pr"]     = float(vl_pr)

    if verbose:
        print("-" * 78)
        print(f"[DONE] val_loss first/last/min:  "
              f"{hist['val_loss'][0]:.6f} -> {hist['val_loss'][-1]:.6f}  "
              f"(min {min(hist['val_loss']):.6f})")
        print(f"[DONE] val_loss (PR-avg final):  {vl_pr:.6f}")
    return hist


def train_theta_spsa(bank: TrainableShotPauliBank,
                     sketch_cache: np.ndarray,
                     Y_train: np.ndarray,
                     blocks,
                     ridge_alpha: float,
                     steps: int = 120,
                     batch_size: int = 128,
                     a0: float = 0.08, c0: float = 0.12,
                     alpha: float = 0.602, gamma: float = 0.101,
                     sigma_mults=(0.5, 1.0, 2.0),
                     val_batch_size: int = 512,
                     feature_chunk_size: int = 256,
                     seed: int = 7, verbose: bool = True) -> dict:
    """Paper-style SPSA baseline, identical MMD + val monitor for fair ablation."""
    prng = np.random.default_rng(seed + 999)
    theta = bank.get_theta().copy()
    p_theta = theta.size
    nb = len(blocks)
    n_train = Y_train.shape[0]

    val_n = min(int(val_batch_size), n_train)
    val_idx = prng.choice(n_train, size=val_n, replace=False)
    train_mask = np.ones(n_train, dtype=bool); train_mask[val_idx] = False
    train_pool = np.where(train_mask)[0]

    sigmas_per_block = []
    for (start, bsz) in blocks:
        Y_blk_val = Y_train[val_idx, start:start + bsz]
        sig_star = median_sigma(Y_blk_val, Y_blk_val, cap=256, seed=seed)
        sigmas_per_block.append([m * sig_star for m in sigma_mults])

    if verbose:
        print("=" * 78)
        print(f"TRAIN THETA -- SPSA baseline  p_theta={p_theta}  steps={steps}")
        print(f"  a0={a0}  c0={c0}  alpha={alpha}  gamma={gamma}  batch={batch_size}")
        print("=" * 78)

    hist = dict(train_loss=[], val_loss=[], val_loss_ema=[],
                grad_norm=[], step_norm=[], lr=[], sigmas=sigmas_per_block)
    val_ema = None

    for t in range(steps):
        a_t = a0 / ((t + 1.0) ** alpha)
        c_t = c0 / ((t + 1.0) ** gamma)
        bi = int(prng.integers(0, nb))
        start, bsz = blocks[bi]
        batch_n = min(int(batch_size), train_pool.size)
        tr_idx = prng.choice(train_pool, size=batch_n, replace=False)

        S = sketch_cache[bi, tr_idx].astype(np.float64, copy=False)
        Y_b = Y_train[tr_idx, start:start + bsz].astype(np.float64, copy=False)
        sigmas = sigmas_per_block[bi]

        Delta = prng.choice([-1.0, 1.0], size=p_theta).astype(np.float64)
        theta_p = theta + c_t * Delta
        theta_m = theta - c_t * Delta

        Fp = compute_features_chunked(
            bank, S, theta_override=theta_p, chunk_size=feature_chunk_size)
        Wp, _ = ridge_fit(Fp, Y_b, alpha=ridge_alpha)
        loss_p = multi_kernel_mmd2(Fp @ Wp, Y_b, sigmas)

        Fm = compute_features_chunked(
            bank, S, theta_override=theta_m, chunk_size=feature_chunk_size)
        Wm, _ = ridge_fit(Fm, Y_b, alpha=ridge_alpha)
        loss_m = multi_kernel_mmd2(Fm @ Wm, Y_b, sigmas)

        ghat = ((loss_p - loss_m) / (2.0 * c_t)) * Delta
        step_vec = a_t * ghat
        theta = theta - step_vec
        bank.set_theta(theta)

        vl = 0.0
        for bi2 in range(nb):
            s2, b2 = blocks[bi2]
            Sv = sketch_cache[bi2, val_idx].astype(np.float64, copy=False)
            Yv = Y_train[val_idx, s2:s2 + b2].astype(np.float64, copy=False)
            Fv = compute_features_chunked(
                bank, Sv, theta_override=theta, chunk_size=feature_chunk_size)
            Wv, _ = ridge_fit(Fv, Yv, alpha=ridge_alpha)
            vl += multi_kernel_mmd2(Fv @ Wv, Yv, sigmas_per_block[bi2])

        if val_ema is None:
            val_ema = vl
        else:
            val_ema = 0.9 * val_ema + 0.1 * vl

        hist["train_loss"].append(float(0.5 * (loss_p + loss_m)))
        hist["val_loss"].append(float(vl))
        hist["val_loss_ema"].append(float(val_ema))
        hist["grad_norm"].append(float(np.linalg.norm(ghat)))
        hist["step_norm"].append(float(np.linalg.norm(step_vec)))
        hist["lr"].append(float(a_t))

        if verbose and ((t + 1) % max(1, steps // 10) == 0 or t < 3):
            print(f"  step {t+1:3d}/{steps}  l+={loss_p:.5f}  l-={loss_m:.5f}  "
                  f"val={vl:.5f}  a={a_t:.4g}  c={c_t:.4g}")

    hist["theta_final"]     = theta.copy()
    hist["theta_last_iter"] = theta.copy()
    hist["val_loss_pr"]     = float(hist["val_loss"][-1])
    return hist
