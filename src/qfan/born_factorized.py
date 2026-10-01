"""
Factorized CF-MMD loss and exact gradients.

The original cf_mmd_loss_and_grads enumerates every joint measurement record
across the G tensor-product settings, an atom table of size 2^(G n_q). That is
512 atoms at n_q=3 but 10^9 at n_q=10 and 10^18 at n_q=20, and it is the only
thing standing between the architecture and larger registers.

It is unnecessary. The G settings are measured in separate circuit executions,
so the joint law of one record is the product of the per-setting laws,

    w_i(a_0, ..., a_{G-1}) = prod_g P_g[i, a_g],

and the decoded value is a sum over settings,

    y(a) = sum_g T[a_g] @ W_g .

The characteristic function therefore factorizes exactly,

    phi_i(omega) = prod_g  Phi_g[i, omega],
    Phi_g[i, omega] = sum_a P_g[i, a] exp(i T[a] @ W_g . omega / k),

which costs G 2^n_q instead of 2^(G n_q). Every gradient the original code
builds from the atom table reduces the same way, through leave-one-out
products L_g = prod_{h != g} Phi_h, so nothing here ever materializes a joint
record. See verify_factorized.py for the term-by-term check against the
original.
"""
from typing import Optional

import numpy as np

from .born import noise_aware_ridge_fit

PI = np.pi


def _per_setting_factors(bank, P, W, Om, kk):
    """Per-setting CF factors. Returns (Phi, e) with
         Phi[g] : (n, n_w)   the setting-g factor of the characteristic function
         e[g]   : (dim, n_w) exp(i y_g(a) . omega / k) for each outcome a
    """
    nf = bank.T.shape[1]
    Phi, e = [], []
    for g in range(bank.G):
        Yg = bank.T @ W[g * nf:(g + 1) * nf]            # (dim, b)
        eg = np.exp(1j * (Yg @ Om.T) / kk)              # (dim, n_w)
        e.append(eg)
        Phi.append(P[g] @ eg)                           # (n, n_w)
    return Phi, e


def _leave_one_out(Phi):
    """L[g] = prod_{h != g} Phi[h], computed without division so that a
    factor passing near zero cannot produce an inf."""
    G = len(Phi)
    L = []
    for g in range(G):
        acc = np.ones_like(Phi[0])
        for h in range(G):
            if h != g:
                acc = acc * Phi[h]
        L.append(acc)
    return L


def _accumulate_grads(bank, P, Om, kk, dcoef, Phi, e, D_atom, dL_dW):
    """Add one CF term's contribution to the atom-weight gradient D_atom
    (G, n, dim) and to the decoder gradient dL_dW (G nf, b).

    dcoef is dL/dphi for this term, in the convention that the loss changes by
    Re(sum dcoef * dphi). This replaces
        G_wi  = Re(dcoef @ Emat.T)             -> D_atom via the product rule
        dL_dW = f_atoms.T @ Re(tmp @ Omega)    -> per-setting contraction
    from the original, without the (n_atoms,) axis.
    """
    nf = bank.T.shape[1]
    L = _leave_one_out(Phi)
    for g in range(bank.G):
        Zg = dcoef * L[g]                               # (n, n_w)
        # d phi / d P_g[i, a] = L_g[i, w] e_g[a, w]
        D_atom[g] += np.real(Zg @ e[g].T)               # (n, dim)
        # decoder gradient, setting-g block of W
        Qg = Zg.T @ P[g]                                # (n_w, dim)
        Rg = (e[g].T * Qg) @ bank.T                     # (n_w, nf)
        dL_dW[g * nf:(g + 1) * nf] += np.real((1j / kk) * (Rg.T @ Om))


def cf_mmd_loss_and_grads_factorized(bank, S, Y, theta, Omega, ridge_alpha,
                                     k_shots: int = 1,
                                     gamma_risk: float = 0.0,
                                     compute_grad: bool = True,
                                     compute_encoding_grad: bool = False,
                                     X_cond: Optional[np.ndarray] = None,
                                     Omega_cond: Optional[np.ndarray] = None,
                                     rho: Optional[float] = None,
                                     noise_weight: float = 1.0):
    """Drop-in replacement for born.cf_mmd_loss_and_grads. Same signature,
    same return tuple (loss, grad_theta, gA, gb, W, grad_rho), same numbers
    to machine precision, without the joint atom table.

    noise_weight (lambda) weights the shot-noise penalty of the ridge decoder,
    W = (F^T F + lambda Sigma_bar / k + alpha I)^{-1} F^T Y: 1 is the noise-aware
    ridge, 0 the plain ridge of arXiv:2605.16044, Eq. (5). The gradients are
    exact for every lambda. With lambda != 1 the ridge no longer minimises the
    risk term, so gamma_risk must be 0."""
    if noise_weight != 1.0 and gamma_risk > 0.0:
        raise ValueError("noise_weight != 1 requires gamma_risk = 0")
    S = np.asarray(S, np.float64)
    Y = np.asarray(Y, np.float64)
    theta = np.asarray(theta, np.float64).ravel()
    n = S.shape[0]
    nw = Omega.shape[0]
    kk = float(k_shots)
    nf = bank.T.shape[1]
    b = Y.shape[1]

    # ---- probabilities, features, and the closed-form decoder -------------
    # Unchanged from the original: none of this touches the atom table.
    P = bank.setting_probs(S, theta)
    F = bank.expectation_features(P)
    Sigma_bar = bank.parity_covariance_sum(P)
    W, Minv = noise_aware_ridge_fit(F, Y, noise_weight * Sigma_bar, ridge_alpha, k=k_shots)

    if X_cond is not None and Omega_cond is not None:
        phase = np.exp(1j * (np.asarray(X_cond, np.float64) @ Omega_cond.T))
    else:
        phase = np.ones((n, nw))

    # ---- forward: characteristic function ---------------------------------
    # Record sharing applies to any block of two or more pixels. (The original
    # implementation handled only b == 2, so larger blocks silently trained as
    # if every record were shared, rho = 1, and rho never moved from its
    # initial value.) A one-pixel block has nothing to share and needs no split.
    use_split = (rho is not None and b >= 2)
    if use_split:
        # Pixel j averages the k_sh shared records plus its own k_ex private
        # ones. Independent groups of records multiply in the characteristic
        # function, so
        #     psi = A^(rho k) * prod_j M_j^((1 - rho) k)
        # with A the CF at the full frequency and M_j the CF at the frequency
        # restricted to pixel j. Every factor is an ordinary factorized CF,
        # just at a different omega. b = 2 reproduces the original exactly.
        alpha = float(rho) * kk
        beta = (1.0 - float(rho)) * kk
        eps_c = 1e-300
        OmA = Omega
        PhiA, eA = _per_setting_factors(bank, P, W, OmA, kk)
        Aw = np.prod(PhiA, axis=0)
        LA = np.log(Aw + eps_c)
        OmM, PhiM, eM, Mw, LM = [], [], [], [], []
        for j in range(b):
            mask = np.zeros(b)
            mask[j] = 1.0
            Om_j = Omega * mask
            Phi_j, e_j = _per_setting_factors(bank, P, W, Om_j, kk)
            M_j = np.prod(Phi_j, axis=0)
            OmM.append(Om_j); PhiM.append(Phi_j); eM.append(e_j)
            Mw.append(M_j); LM.append(np.log(M_j + eps_c))
        sumLM = np.sum(LM, axis=0)
        psi_i = np.exp(alpha * LA + beta * sumLM)
    else:
        Phi, e = _per_setting_factors(bank, P, W, Omega, kk)
        phi = np.prod(Phi, axis=0)
        phi_km1 = phi ** (k_shots - 1)
        psi_i = phi_km1 * phi

    psi = (phase * psi_i).mean(axis=0)
    psi_d = (phase * np.exp(1j * (Y @ Omega.T))).mean(axis=0)
    diff = psi - psi_d
    loss = float(np.mean(np.abs(diff) ** 2))

    E_res = F @ W - Y
    if gamma_risk > 0.0:
        risk = (float(np.sum(E_res * E_res))
                + float(np.sum(W * (Sigma_bar @ W))) / kk
                + float(ridge_alpha) * float(np.sum(W * W))) / n
        loss += gamma_risk * risk

    if not compute_grad and not compute_encoding_grad:
        return loss, None, None, None, W, None

    # ---- backward through the characteristic function ---------------------
    Qbase = (2.0 / (n * nw)) * (np.conj(diff)[None, :] * phase)
    D_atom = np.zeros((bank.G, n, bank.dim))
    dL_dW = np.zeros((bank.G * nf, b))
    grad_rho = None
    if use_split:
        Qp = Qbase * psi_i
        _accumulate_grads(bank, P, OmA, kk, alpha * Qp / (Aw + eps_c),
                          PhiA, eA, D_atom, dL_dW)
        for j in range(b):
            _accumulate_grads(bank, P, OmM[j], kk, beta * Qp / (Mw[j] + eps_c),
                              PhiM[j], eM[j], D_atom, dL_dW)
        # d/d rho of (rho k log A + (1 - rho) k sum_j log M_j)
        grad_rho = float(np.real((Qp * (kk * (LA - sumLM))).sum()))
    else:
        dphi = (k_shots * Qbase) * phi_km1
        _accumulate_grads(bank, P, Omega, kk, dphi, Phi, e, D_atom, dL_dW)

    C = Minv @ dL_dW
    R = Y - F @ W

    # ---- theta: parameter shift ------------------------------------------
    # The original contracts G_wi with d(atom weights)/d theta. By the product
    # rule that equals sum_g <D_atom[g], dP_g>, which needs only (G, n, dim).
    grad_theta = None
    if compute_grad:
        grad_theta = np.zeros(theta.size)
        shift = 0.5 * PI
        for j in range(theta.size):
            tp = theta.copy(); tp[j] += shift
            tm = theta.copy(); tm[j] -= shift
            dP = 0.5 * (bank.setting_probs(S, tp) - bank.setting_probs(S, tm))
            g = float(np.sum(D_atom * dP))
            Gk = bank.expectation_features(dP)
            dSig = bank.parity_covariance_sum(P, dP)
            dM = Gk.T @ F + F.T @ Gk + noise_weight * dSig / kk
            dW = Minv @ (Gk.T @ Y - dM @ W)
            g += float(np.sum(dL_dW * dW))
            if gamma_risk > 0.0:
                g += gamma_risk * (
                    2.0 * float(np.sum(E_res * (Gk @ W)))
                    + float(np.sum(W * (dSig @ W))) / kk) / n
            grad_theta[j] = g

    # ---- encoding (A, b) ---------------------------------------------------
    # D starts from the atom-weight gradient, then picks up the implicit
    # decoder terms exactly as in the original.
    gA = gb = None
    if compute_encoding_grad:
        D = D_atom.copy()
        U = 0.5 * (C @ W.T + W @ C.T)
        for g in range(bank.G):
            sl = slice(g * nf, (g + 1) * nf)
            Wg = W[sl]
            Ug = U[sl, sl]
            mu_g = P[g] @ bank.T
            coeff = R @ C.T - (F @ C) @ W.T
            D[g] += coeff[:, sl] @ bank.T.T
            q_quad = np.einsum("ac,cd,ad->a", bank.T, Ug, bank.T)
            lin = 2.0 * (mu_g @ Ug) @ bank.T.T
            D[g] += -noise_weight * (q_quad[None, :] - lin) / kk
            if gamma_risk > 0.0:
                G_risk = (2.0 / n) * (E_res @ W.T)
                D[g] += gamma_risk * (G_risk[:, sl] @ bank.T.T)
                WWt = Wg @ Wg.T
                q2 = np.einsum("ac,cd,ad->a", bank.T, WWt, bank.T)
                lin2 = 2.0 * (mu_g @ WWt) @ bank.T.T
                D[g] += gamma_risk * (q2[None, :] - lin2) / (kk * n)

        a_mat = bank.angles_from_sketch(S)
        dL_da = np.zeros((n, bank.L))
        for l in range(bank.L):
            acc = np.zeros((bank.G, n, bank.dim))
            for layer in range(bank.depth):
                Pp = bank.setting_probs(S, theta,
                                        angle_shift=(l, layer, +0.5 * PI))
                Pm = bank.setting_probs(S, theta,
                                        angle_shift=(l, layer, -0.5 * PI))
                acc += 0.5 * (Pp - Pm)
            dL_da[:, l] = np.einsum("gia,gia->i", D, PI * acc)
        sig_prime = a_mat * (1.0 - a_mat)
        gA = (dL_da * sig_prime).T @ S
        gb = (dL_da * sig_prime).sum(axis=0)

    return loss, grad_theta, gA, gb, W, grad_rho
