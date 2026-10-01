"""
QFAN circuit engine: the Born measurement distribution is the generative model.

MODEL
A shower image of d pixels is generated in consecutive blocks of b pixels.
For each block, a fixed-length count sketch S of the pixels generated so far is
encoded into rotation angles a(S), and one shared parameterised circuit
U(theta) on n_q qubits (RY/RZ layers with a CZ ring) prepares |psi(S, theta)>.
The state is measured k times in each of G = 3 tensor-product settings (all Z,
all X, all Y). From every measurement record the single-qubit and two-qubit
parities are read, p_f = 3 n_q (n_q + 1) / 2 features in all, and averaged over
the k records to give f_hat. The block is decoded linearly,

    y = f_hat W (+ b0),        clipped at zero,

and appended to the image, updating the sketch for the next block. The finite-k
average is the model output: the Born measurement outcomes are its only source
of randomness. Conditionally on S,

    E[y | S]   = F(S) W,               F(S) = Pauli expectations,
    Cov(y | S) = W^T Sigma(S, theta) W / k,

with Sigma the covariance of the parities under the Born rule, so training the
circuit shapes the conditional mean and the conditional dispersion together.
Within a block a trained fraction rho of the k records is shared between the
pixels, the rest drawn separately for each pixel, which makes the covariance of
two pixels (k_sh / k^2) w_i^T Sigma w_j: record sharing turns measurement noise
into inter-pixel correlation.

DECODER
W is fitted in closed form by the shot-noise-aware ridge (noise_aware_ridge_fit)

    W = (F^T F + Sigma_bar / k + alpha I)^{-1} F^T Y,

where Sigma_bar sums the parity covariances over the training batch. The
Sigma_bar / k term accounts for the fact that generation reads noisy k-shot
averages rather than exact expectations; without it the decoder amplifies shot
noise. An optional per-pixel intercept b0 restores the training mean, which the
shrinkage otherwise pulls toward zero.

TRAINING
The loss is a characteristic-function maximum mean discrepancy between each
block's model law and the data, evaluated at random frequencies
(cf_mmd_loss_and_grads, with the memory-efficient factorised form in
born_factorized.py). Its gradients are exact: parameter-shift rules for theta,
analytic derivatives for the encoding and the record shares, and implicit
differentiation through the closed-form decoder. All are verified against
finite differences in tests/test_born.py. The single-record atomic loss
(born_block_loss_and_grad) is kept as an exact reference for the tests.

GENERATION
sample_progressive_born generates free-running from the first block. Options:
return_latent returns the values before clipping.

SCOPE
Every trained parameter is a circuit parameter or part of the closed-form
decoder, and every random bit of a generated sample is a Born measurement
outcome. This module simulates the circuit exactly with a statevector; at the
register sizes used here that is cheap, and no computational advantage is
claimed.
"""

from dataclasses import dataclass
from math import pi
from typing import List, Optional, Tuple

import numpy as np

from .mmd import _sqdist
from .ridge import ridge_fit, ridge_inverse_matrix
from .utils import _stable_sigmoid


# =========================================================================
# Pure-numpy statevector engine for the (unchanged) QFAN circuit
# =========================================================================
#
# Bit convention: basis index i in [0, 2^nq); bit of qubit q is (i >> q) & 1.
# Verified against an independent dense-unitary (Kronecker) implementation
# and, when qiskit is installed, against qiskit's Statevector / aer counts
# (tests/test_born.py).

@dataclass
class BornBankSpec:
    n_qubits: int = 3
    depth: int = 2
    angle_dim: int = 8
    include_y: bool = True      # G=3 settings (Z, X, Y); False -> G=2 (paper)
    gen_shots_k: int = 1        # records averaged per setting at generation


class StatevectorBornBank:
    """
    Same circuit family, same parameter count, same sketch->angle projection
    as TrainableShotPauliBank (), executed on an exact statevector engine.

    Feature layout (per measurement setting g, in order Z, X[, Y]):
        [ P_q0, P_q1, P_q2, P_{q0 q1}, P_{q0 q2}, P_{q1 q2} ]   (parities)
    so pf = 6 * G. Expectations of these parities are exactly the Pauli
    expectations <Z_i>, <Z_i Z_j>, <X_i>, ... of paper Lemma 1 (plus the
    optional Y family).
    """

    def __init__(self, sketch_dim: int, spec: BornBankSpec, seed: int = 7):
        # n_q generalization: verified against an independent dense-unitary
        # construction (states to ~5e-16) and by finite differences on the
        # theta, rho and encoding gradients, for n_q = 2, 3, 4, 5.
        # Cost note: the joint-record atom table scales as 2^(G*n_q), i.e.
        # 32768 atoms at n_q=5 and 262144 at n_q=6; n_q >= 7 is impractical.
        if spec.n_qubits < 2:
            raise NotImplementedError("need n_qubits >= 2")
        self.spec = spec
        self.m = int(sketch_dim)
        self.rng = np.random.default_rng(seed)

        self.nq = int(spec.n_qubits)
        self.dim = 1 << self.nq
        self.depth = int(spec.depth)
        self.L = int(spec.angle_dim)
        self.G = 3 if spec.include_y else 2

        # identical init pattern to 's TrainableShotPauliBank
        self.A = (0.2 * self.rng.normal(size=(self.L, self.m))).astype(np.float64)
        self.b = (0.05 * self.rng.normal(size=(self.L,))).astype(np.float64)
        self.n_var = self.depth * self.nq * 2
        self.theta = self.rng.normal(0, pi / 4, size=(self.n_var,)).astype(np.float64)

        # index bookkeeping for vectorized gates
        self._I0 = []
        self._I1 = []
        for q in range(self.nq):
            i0 = np.array([i for i in range(self.dim) if not ((i >> q) & 1)],
                          dtype=np.int64)
            self._I0.append(i0)
            self._I1.append(i0 | (1 << q))
        # CZ ring masks
        self._cz_masks = []
        pairs = [(q, q + 1) for q in range(self.nq - 1)]
        if self.nq > 2:
            pairs.append((self.nq - 1, 0))
        for (i, j) in pairs:
            mask = np.array([bool(((n >> i) & 1) and ((n >> j) & 1))
                             for n in range(self.dim)])
            self._cz_masks.append(mask)

        # parity table: T[outcome, feature]  for features
        # [q0, q1, q2, (q0,q1), (q0,q2), (q1,q2)]
        singles = [(q,) for q in range(self.nq)]
        doubles = [(i, j) for i in range(self.nq) for j in range(i + 1, self.nq)]
        self.parity_supports = singles + doubles
        T = np.zeros((self.dim, len(self.parity_supports)), dtype=np.float64)
        for out in range(self.dim):
            for c, sup in enumerate(self.parity_supports):
                s = sum((out >> q) & 1 for q in sup)
                T[out, c] = 1.0 if (s % 2 == 0) else -1.0
        self.T = T                              # (8, 6)
        self.pf = T.shape[1] * self.G           # 12 or 18

        # The joint-record atom table has dim^G = 2^(G n_q) rows. It is only
        # needed by the original cf_mmd_loss_and_grads and the k=1 atomic loss,
        # and it grows tenfold per qubit (270 MB at n_q=6, ~14 GB at n_q=8).
        # It is therefore built lazily, on first access, so that the factorized
        # loss can run at any register size without ever allocating it.
        self.n_atoms = self.dim ** self.G
        self._f_atoms = None

        self.feature_names = [
            f"{ax}:{'.'.join(map(str, sup))}"
            for ax in ("Z", "X", "Y")[: self.G]
            for sup in self.parity_supports
        ]

    # ---------------- parameter access ----------------

    def feature_dim(self) -> int:
        return self.pf

    def get_theta(self) -> np.ndarray:
        return self.theta.copy()

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, np.float64).ravel()
        if theta.size != self.n_var:
            raise ValueError("theta dim mismatch")
        self.theta = theta.copy()

    # ---------------- gates (vectorized over batch) ----------------

    def _ry(self, psi, q, phi):
        phi = np.asarray(phi, np.float64)
        c = np.cos(0.5 * phi)[..., None]
        s = np.sin(0.5 * phi)[..., None]
        a0 = psi[:, self._I0[q]].copy()
        a1 = psi[:, self._I1[q]]
        psi[:, self._I0[q]] = c * a0 - s * a1
        psi[:, self._I1[q]] = s * a0 + c * a1

    def _rz(self, psi, q, phi):
        phi = np.asarray(phi, np.float64)
        e0 = np.exp(-0.5j * phi)[..., None]
        e1 = np.exp(+0.5j * phi)[..., None]
        psi[:, self._I0[q]] *= e0
        psi[:, self._I1[q]] *= e1

    def _h_all(self, psi):
        inv = 1.0 / np.sqrt(2.0)
        for q in range(self.nq):
            a0 = psi[:, self._I0[q]].copy()
            a1 = psi[:, self._I1[q]]
            psi[:, self._I0[q]] = inv * (a0 + a1)
            psi[:, self._I1[q]] = inv * (a0 - a1)

    def _sdg_all(self, psi):
        for q in range(self.nq):
            psi[:, self._I1[q]] *= -1j

    def _cz_ring(self, psi):
        for mask in self._cz_masks:
            psi[:, mask] *= -1.0

    # ---------------- circuit ----------------

    def angles_from_sketch(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, np.float64)
        if S.ndim == 1:
            S = S.reshape(1, -1)
        return _stable_sigmoid(S @ self.A.T + self.b[None, :])

    def state(self, S: np.ndarray, theta: Optional[np.ndarray] = None,
              angle_shift: Optional[Tuple[int, int, float]] = None) -> np.ndarray:
        """
        |psi(a(S), theta)> for each row of S. Returns (n, 2^nq) complex.

        angle_shift=(l, layer, delta): add `delta` to the ROTATION ANGLE of
        the single re-uploading gate that encodes a_l in layer `layer`
        (used for the per-occurrence parameter-shift rule on the trainable
        encoding; a_l appears once per layer, so d/d(pi a_l) is the sum of
        per-occurrence shift-rule terms).
        """
        th = self.theta if theta is None else np.asarray(theta, np.float64).ravel()
        a = self.angles_from_sketch(S)
        n = a.shape[0]
        psi = np.zeros((n, self.dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        t = 0
        for layer in range(self.depth):
            for k in range(self.L):                     # data re-uploading
                q = k % self.nq
                phi = pi * a[:, k]
                if angle_shift is not None and \
                        angle_shift[0] == k and angle_shift[1] == layer:
                    phi = phi + angle_shift[2]
                if (k % 2) == 0:
                    self._ry(psi, q, phi)
                else:
                    self._rz(psi, q, phi)
            for q in range(self.nq):                    # variational
                self._rz(psi, q, np.full(n, th[t])); t += 1
                self._ry(psi, q, np.full(n, th[t])); t += 1
            if self.nq >= 2:
                self._cz_ring(psi)
        return psi

    def setting_probs(self, S: np.ndarray,
                      theta: Optional[np.ndarray] = None,
                      angle_shift: Optional[Tuple[int, int, float]] = None
                      ) -> np.ndarray:
        """
        Born probabilities in each tensor-product setting.
        Returns P of shape (G, n, 2^nq); P[0]=Z basis, P[1]=X, [P[2]=Y].
        """
        psi = self.state(S, theta, angle_shift=angle_shift)
        n = psi.shape[0]
        P = np.zeros((self.G, n, self.dim), dtype=np.float64)
        P[0] = np.abs(psi) ** 2
        px = psi.copy()
        self._h_all(px)
        P[1] = np.abs(px) ** 2
        if self.G == 3:
            py = psi.copy()
            self._sdg_all(py)
            self._h_all(py)
            P[2] = np.abs(py) ** 2
        return P

    # ---------------- features / atoms / weights ----------------

    def expectation_features(self, P: np.ndarray) -> np.ndarray:
        """F (n, pf): exact Pauli expectations from probability tables.
        Linear in P, so the same map applied to dP gives dF."""
        return np.concatenate([P[g] @ self.T for g in range(self.G)], axis=1)

    @property
    def f_atoms(self) -> np.ndarray:
        """(dim^G, p_f) joint-record feature table, built on first use."""
        if self._f_atoms is None:
            grids = np.meshgrid(*[np.arange(self.dim)] * self.G, indexing="ij")
            outs = [g.ravel() for g in grids]
            self._f_atoms = np.concatenate([self.T[o] for o in outs], axis=1)
        return self._f_atoms

    def atom_weights(self, P: np.ndarray) -> np.ndarray:
        """w (n, 8^G): joint probability of each multi-setting record."""
        n = P.shape[1]
        w = P[0]
        for g in range(1, self.G):
            w = w[..., None] * P[g].reshape((n,) + (1,) * g + (self.dim,))
        return w.reshape(n, self.n_atoms)

    def atom_weight_derivative(self, P: np.ndarray, dP: np.ndarray) -> np.ndarray:
        """
        d w / d theta_k via the multi-setting product rule:
            d prod_g p_g = sum_g (prod_{h!=g} p_h) dp_g.
        P, dP: (G, n, 2^nq).  Returns (n, 8^G).
        """
        n = P.shape[1]
        total = np.zeros((n, self.n_atoms), dtype=np.float64)
        for g in range(self.G):
            w = dP[g] if g == 0 else P[0]
            for h in range(1, self.G):
                fac = dP[h] if h == g else P[h]
                w = w[..., None] * fac.reshape((n,) + (1,) * h + (self.dim,))
            total += w.reshape(n, self.n_atoms)
        return total

    def parity_covariance_sum(self, P: np.ndarray,
                              dP: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sigma_bar = sum_i Cov[f | S_i]  (pf x pf), the total single-record
        parity covariance over the batch. Exact from probability tables.

        Records of different settings are independent given the state, so
        the covariance is exactly block-diagonal over settings; within a
        setting,  Cov_g = E[T_a T_a^T] - mu_g mu_g^T  under p_g.

        If dP is given, returns d Sigma_bar / d theta_k instead (exact,
        using the same tables):  d(E2) - d(mu) mu^T - mu d(mu)^T, summed.
        """
        n = P.shape[1]
        nf = self.T.shape[1]
        if not hasattr(self, "_T_outer"):
            self._T_outer = np.einsum("ac,ad->acd", self.T, self.T)\
                              .reshape(self.dim, nf * nf)
        Sig = np.zeros((self.pf, self.pf), dtype=np.float64)
        for g in range(self.G):
            sl = slice(g * nf, (g + 1) * nf)
            mu = P[g] @ self.T                                   # (n, nf)
            if dP is None:
                E2 = (P[g] @ self._T_outer).reshape(n, nf, nf).sum(axis=0)
                Sig[sl, sl] = E2 - np.einsum("ic,id->cd", mu, mu)
            else:
                dmu = dP[g] @ self.T
                dE2 = (dP[g] @ self._T_outer).reshape(n, nf, nf).sum(axis=0)
                Sig[sl, sl] = (dE2 - np.einsum("ic,id->cd", dmu, mu)
                               - np.einsum("ic,id->cd", mu, dmu))
        return Sig

    def sample_records(self, P: np.ndarray, k: int,
                       rng: np.random.Generator) -> np.ndarray:
        """Draw k Born records per setting; returns outcome ints (G, n, k)."""
        n = P.shape[1]
        rec = np.zeros((self.G, n, k), dtype=np.int64)
        for g in range(self.G):
            cdf = np.cumsum(P[g], axis=1)
            cdf[:, -1] = 1.0
            u = rng.random((n, k))
            for i in range(n):
                rec[g, i] = np.searchsorted(cdf[i], u[i], side="right")
        return np.clip(rec, 0, self.dim - 1)

    def sample_record_features(self, P: np.ndarray, k: int,
                               rng: np.random.Generator) -> np.ndarray:
        """
        Draw k Born records per setting and return k-shot parity averages
        f_hat (n, pf). E[f_hat] equals expectation_features(P) exactly.
        (Simulator surrogate for hardware measurement; identical law.)
        """
        n = P.shape[1]
        feats = []
        for g in range(self.G):
            cdf = np.cumsum(P[g], axis=1)
            cdf[:, -1] = 1.0
            u = rng.random((n, k))
            idx = np.stack([np.searchsorted(cdf[i], u[i], side="right")
                            for i in range(n)])          # (n, k)
            idx = np.clip(idx, 0, self.dim - 1)
            feats.append(self.T[idx].mean(axis=1))        # (n, 6)
        return np.concatenate(feats, axis=1)


# =========================================================================
# Shot-noise-aware closed-form ridge decoder
# =========================================================================

def noise_aware_ridge_fit(F: np.ndarray, Y: np.ndarray,
                          Sigma_bar: np.ndarray, alpha: float,
                          k: int = 1):
    """
    The decoder: the closed form of the paper's Eq. (5), with the physically
    mandated metric. The decoder output is y = f_hat W where f_hat is
    a k-shot Born record, so the decoder's expected square loss is

        sum_i E || f_hat_i W - Y_i ||^2
          = ||F W - Y||_F^2  +  (1/k) tr( W^T Sigma_bar W ),

    with Sigma_bar = sum_i Cov[f | S_i]. Minimizing it (plus alpha||W||^2)
    stays closed-form:

        W = ( F^T F + Sigma_bar / k + alpha I )^{-1} F^T Y.

    Interpretation: Gauss-Markov-optimal linear read-out of a quantum
    sampler -- the decoder trusts each Pauli feature in proportion to its
    signal-to-shot-noise ratio. Plain LSQ ('s fit, Sigma_bar = 0) ignores
    the noise gain ||W|| it imposes on the Born randomness and overshoots
    the dispersion by an order of magnitude (ablation C6).
    """
    F = np.asarray(F, np.float64)
    Y = np.asarray(Y, np.float64)
    p = F.shape[1]
    M = F.T @ F + Sigma_bar / float(k) + float(alpha) * np.eye(p)
    Minv = np.linalg.inv(M)
    W = Minv @ (F.T @ Y)
    return W, Minv


# =========================================================================
# Exact loss + exact gradient (population MMD^2 of the Born pushforward)
# =========================================================================

def born_block_loss_and_grad(bank: StatevectorBornBank,
                             S: np.ndarray,
                             Y: np.ndarray,
                             theta: np.ndarray,
                             sigmas,
                             ridge_alpha: float,
                             compute_grad: bool = True,
                             include_ridge_term: bool = True,
                             clip_atoms: bool = True,
                             noise_aware: bool = True,
                             k_shots: int = 1,
                             gamma_risk: float = 0.0):
    """
    L(theta) = sum_sigma MMD^2_sigma( model law , empirical Y )   [exact]

    model law = (1/n) sum_i sum_atoms w_i(atom; theta) delta_{clip(f(atom) W)}

    clip_atoms=True applies the same max(0, .) used at generation time to
    the atom positions, so the trained law IS the generated law ( trained
    unclipped and clipped only at generation; that mismatch is removed
    here). The W-gradient uses the exact a.e. subgradient of the clip.

    gamma_risk > 0 adds the decoder's own irreducible risk,

        Risk(theta) = (1/n) min_W [ ||F W - Y||_F^2
                                    + (1/k) tr(W^T Sigma_bar W)
                                    + alpha ||W||_F^2 ],

    to the loss. This is the *quantum feature amplification* term: its
    theta-gradient (exact by the ENVELOPE THEOREM, since W is the argmin)
    rewards the circuit for producing measurement statistics that are
    simultaneously informative about the target (large signal) and
    low-uncertainty (small shot variance) -- i.e. it trains conditional
    response, which the aggregate per-block MMD alone does not reward.
    Envelope gradient (no d W / d theta needed for this term):

        d Risk / d theta_k = (2/n) < F W - Y , G_k W >_F
                             + (1/(n k)) tr( W^T dSigma_k W ).

    Gradient = probability term (multi-setting parameter shift, exact)
             + ridge-implicit term through W(theta) (exact,  Eq. A form)
             + envelope risk term (exact).
    Cost: (1 + 2 p_theta) statevector batch passes.
    """
    S = np.asarray(S, np.float64)
    Y = np.asarray(Y, np.float64)
    theta = np.asarray(theta, np.float64).ravel()
    n = S.shape[0]
    m = Y.shape[0]

    P = bank.setting_probs(S, theta)                    # (G, n, 8)
    F = bank.expectation_features(P)                    # (n, pf)
    if noise_aware:
        Sigma_bar = bank.parity_covariance_sum(P)
        W, Minv = noise_aware_ridge_fit(F, Y, Sigma_bar, ridge_alpha,
                                        k=k_shots)
    else:
        Minv = ridge_inverse_matrix(F, ridge_alpha)
        W = Minv @ (F.T @ Y)

    wbar = bank.atom_weights(P).mean(axis=0)            # (8^G,)
    y_raw = bank.f_atoms @ W                            # (8^G, b)
    if clip_atoms:
        clip_mask = (y_raw > 0.0).astype(np.float64)
        y_atoms = np.maximum(y_raw, 0.0)
    else:
        clip_mask = None
        y_atoms = y_raw

    Daa = _sqdist(y_atoms, y_atoms)
    Dad = _sqdist(y_atoms, Y)
    Ddd = _sqdist(Y, Y)

    loss = 0.0
    g_w = np.zeros_like(wbar)                           # dL/d wbar
    dL_dy = np.zeros_like(y_atoms)                      # dL/d y_atoms
    for sigma in sigmas:
        s2 = 2.0 * float(sigma) ** 2
        inv = 1.0 / float(sigma) ** 2
        Kaa = np.exp(-Daa / s2)
        Kad = np.exp(-Dad / s2)
        Kdd = np.exp(-Ddd / s2)
        Kw = Kaa @ wbar
        Kd1 = Kad.sum(axis=1) / m
        loss += float(wbar @ Kw - 2.0 * wbar @ Kd1 + Kdd.sum() / (m * m))
        if compute_grad:
            g_w += 2.0 * (Kw - Kd1)
            # d/dy_p of  sum w_p w_q k(y_p,y_q):  -2 w_p/s^2 sum_q w_q (y_p-y_q) K_pq
            wK = wbar[:, None] * (Kaa * wbar[None, :])          # w_p w_q K_pq
            dL_dy += -2.0 * inv * (wK.sum(axis=1, keepdims=True) * y_atoms
                                   - wK @ y_atoms)
            # cross term  -2/m sum_j w_p k(y_p, Y_j):
            #   d/dy_p = +2 w_p/(m s^2) sum_j (y_p - Y_j) K_pj
            wKd = wbar[:, None] * Kad
            dL_dy += (2.0 / m) * inv * (wKd.sum(axis=1, keepdims=True) * y_atoms
                                        - wKd @ Y)

    E = F @ W - Y                                       # mean residual
    if gamma_risk > 0.0:
        if not noise_aware:
            raise ValueError("gamma_risk requires noise_aware=True")
        risk = (float(np.sum(E * E))
                + float(np.sum(W * (Sigma_bar @ W))) / float(k_shots)
                + float(ridge_alpha) * float(np.sum(W * W))) / n
        loss = loss + gamma_risk * risk

    if not compute_grad:
        return float(loss), None, W, F

    if clip_atoms:
        dL_dy = dL_dy * clip_mask                       # subgradient of clip
    dL_dW = bank.f_atoms.T @ dL_dy                      # (pf, b)

    p_theta = theta.size
    grad = np.zeros(p_theta, dtype=np.float64)
    shift = 0.5 * pi
    for kk in range(p_theta):
        tp = theta.copy(); tp[kk] += shift
        tm = theta.copy(); tm[kk] -= shift
        Pp = bank.setting_probs(S, tp)
        Pm = bank.setting_probs(S, tm)
        dP = 0.5 * (Pp - Pm)                            # exact d P/d theta_k
        # (i) probability (Born-weight) term
        dwbar = bank.atom_weight_derivative(P, dP).mean(axis=0)
        gk_total = float(g_w @ dwbar)

        Gk = bank.expectation_features(dP)              # dF/d theta_k
        dSig = (bank.parity_covariance_sum(P, dP)
                if (noise_aware and (include_ridge_term or gamma_risk > 0))
                else None)

        # (ii) implicit gradient through the closed-form decoder:
        #   M W = F^T Y,  M = F^T F [+ Sigma_bar/k] + alpha I
        #   dW = M^{-1} ( dF^T Y - dM W )
        if include_ridge_term:
            dM = Gk.T @ F + F.T @ Gk
            if noise_aware:
                dM = dM + dSig / float(k_shots)
            dW = Minv @ (Gk.T @ Y - dM @ W)
            gk_total += float(np.sum(dL_dW * dW))

        # (iii) envelope gradient of the decoder risk (feature amplification)
        if gamma_risk > 0.0:
            d_risk = (2.0 * float(np.sum(E * (Gk @ W)))
                      + float(np.sum(W * (dSig @ W))) / float(k_shots)) / n
            gk_total += gamma_risk * d_risk

        grad[kk] = gk_total
    return float(loss), grad, W, F


# =========================================================================
# Exact characteristic-function MMD for k-shot-averaged Born pushforwards
# =========================================================================
#
# For generation with k > 1 records per setting (k-shot parity averages),
# the model conditional law is the k-fold convolution of the (rescaled)
# single-record atomic law. Its characteristic function is EXACTLY
#
#     E[ exp(i w . y_bar) | S_i ]  =  phi_i(w / k)^k,
#     phi_i(u) = sum_atoms  w_i(atom)  exp(i u . y_atom),
#
# so with a random-Fourier-feature kernel  kappa(x,y) = E_w cos(w.(x-y))
# (w drawn once at init from the multi-bandwidth Gaussian family and then
# FIXED -- a valid PSD kernel, deterministic objective), the population
# MMD^2 between the k-shot model law and the data is an exact, closed-form,
# differentiable function of the probability tables and W:
#
#     MMD^2 = (1/n_w) sum_w | psi(w) - psi_data(w) |^2,
#     psi(w) = (1/n) sum_i phi_i(w/k)^k,   psi_data(w) = (1/m) sum_j e^{i w.Y_j}.
#
# This makes the shot count k a TRAINABLE-THROUGH physical dispersion
# parameter: k = 1 is the maximally stochastic Born sampler, k -> inf
# collapses to the deterministic conditional mean. The decoder shrinkage
# (Sigma_bar / k) and the noise scale (1/k) stay mutually consistent.

def draw_rff_frequencies(sigmas, b_out: int, n_freq: int, seed: int = 0):
    """Frequencies for the multi-bandwidth RFF kernel: for each sigma,
    n_freq/len(sigmas) draws from N(0, sigma^-2 I)."""
    rng = np.random.default_rng(seed)
    per = max(1, n_freq // len(list(sigmas)))
    blocks_ = [rng.normal(0.0, 1.0 / float(s), size=(per, b_out))
               for s in sigmas]
    return np.concatenate(blocks_, axis=0)                    # (n_w, b)


def draw_joint_rff_frequencies(sigmas_y, b_out: int,
                               sigmas_c, d_c: int,
                               n_freq: int, cond_frac: float = 0.5,
                               seed: int = 0):
    """
    Frequencies for the JOINT (prefix, block) RFF kernel. A fraction
    (1 - cond_frac) of rows have zero conditioning frequency (pure marginal
    matching); the rest carry Gaussian frequencies in both prefix and block
    coordinates, with bandwidths drawn from the two sigma families.
    Returns (Omega_y (n_freq, b_out), Omega_c (n_freq, d_c)).
    """
    rng = np.random.default_rng(seed)
    sy = list(sigmas_y)
    sc = list(sigmas_c) if d_c > 0 else [1.0]
    Oy = np.zeros((n_freq, b_out))
    Oc = np.zeros((n_freq, max(d_c, 1)))
    n_cond = int(round(cond_frac * n_freq)) if d_c > 0 else 0
    for r in range(n_freq):
        s_y = sy[rng.integers(0, len(sy))]
        Oy[r] = rng.normal(0.0, 1.0 / float(s_y), size=b_out)
        if r < n_cond:
            s_c = sc[rng.integers(0, len(sc))]
            Oc[r] = rng.normal(0.0, 1.0 / float(s_c), size=max(d_c, 1))
    return Oy, (Oc[:, :d_c] if d_c > 0 else None)


def cf_mmd_loss_and_grads(bank: StatevectorBornBank,
                          S: np.ndarray, Y: np.ndarray,
                          theta: np.ndarray,
                          Omega: np.ndarray,
                          ridge_alpha: float,
                          k_shots: int = 64,
                          gamma_risk: float = 0.0,
                          compute_grad: bool = True,
                          compute_encoding_grad: bool = False,
                          X_cond: Optional[np.ndarray] = None,
                          Omega_cond: Optional[np.ndarray] = None,
                          rho: Optional[float] = None):
    """
    Exact loss and exact gradients (theta, and optionally the encoding
    (A, b)) of

        L = MMD^2_RFF( k-shot Born pushforward , data )  +  gamma * Risk,

    with the shot-noise-aware decoder W = (F^T F + Sigma_bar/k + a I)^-1 F^T Y
    refit inside (its implicit gradient is included exactly).
    All statements FD-verified in tests/test_born.py (T10, T11, T13).

    CONDITIONAL (joint) matching -- the cross-block correlation fix
    ----------------------------------------------------------------
    Per-block marginal MMD is blind to cross-block dependence, which is why
    marginal-trained models miss the prefix<->block (anti)correlations. If
    X_cond (n, d_c) holds the TEACHER-FORCED PREFIX values of each sample
    and Omega_cond (n_w, d_c) the matching frequency block, the loss becomes
    the RFF-MMD between the JOINT laws

        ( X_prefix ,  y_block^model )   vs   ( X_prefix ,  Y_block^data ),

    sharing the same conditioning marginal, i.e. a direct match of the
    model's CONDITIONAL law to the data's. In CF space this costs one
    per-sample phase factor:

        psi(w)   = (1/n) sum_i e^{i w_c . X_i}  phi_i(w_y / k)^k
        psi_d(w) = (1/n) sum_i e^{i w_c . X_i}  e^{i w_y . Y_i} .

    Rows of Omega_cond that are zero reduce to plain marginal matching, so
    a mix of zero and nonzero conditioning rows trains marginals and
    conditionals simultaneously. Cross-block structure in generation is
    then carried by the trained conditional response (autoregressively),
    with no copula and no classical randomness.
    """
    S = np.asarray(S, np.float64)
    Y = np.asarray(Y, np.float64)
    theta = np.asarray(theta, np.float64).ravel()
    n = S.shape[0]
    m = Y.shape[0]
    nw = Omega.shape[0]
    kk = float(k_shots)
    nf = bank.T.shape[1]

    P = bank.setting_probs(S, theta)
    F = bank.expectation_features(P)
    Sigma_bar = bank.parity_covariance_sum(P)
    W, Minv = noise_aware_ridge_fit(F, Y, Sigma_bar, ridge_alpha, k=k_shots)

    Wt = bank.atom_weights(P)                           # (n, n_atoms)
    y_atoms = bank.f_atoms @ W                          # (n_atoms, b)

    Emat = np.exp(1j * (y_atoms @ Omega.T) / kk)        # (n_atoms, n_w)
    if X_cond is not None and Omega_cond is not None:
        phase = np.exp(1j * (np.asarray(X_cond, np.float64) @ Omega_cond.T))
    else:
        phase = np.ones((n, nw))                        # marginal matching

    use_split = (rho is not None and Y.shape[1] == 2)
    if use_split:
        alpha = float(rho) * kk
        beta = (1.0 - float(rho)) * kk
        E1 = np.exp(1j * np.outer(y_atoms[:, 0], Omega[:, 0]) / kk)
        E2 = np.exp(1j * np.outer(y_atoms[:, 1], Omega[:, 1]) / kk)
        Aw = Wt @ Emat                                  # shared factor
        Bw = Wt @ E1
        Cw = Wt @ E2
        eps_c = 1e-300
        LA = np.log(Aw + eps_c)
        LB = np.log(Bw + eps_c)
        LC = np.log(Cw + eps_c)
        psi_i = np.exp(alpha * LA + beta * (LB + LC))   # (n, n_w)
        psi = (phase * psi_i).mean(axis=0)
    else:
        phi = Wt @ Emat                                 # (n, n_w) complex
        phi_km1 = phi ** (k_shots - 1)
        psi_i = phi_km1 * phi
        psi = (phase * psi_i).mean(axis=0)              # (n_w,)
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

    # ---- dL/dw_i(atom) and dL/dy_atom ----
    Qbase = (2.0 / (n * nw)) * (np.conj(diff)[None, :] * phase)  # dL/dpsi_i
    grad_rho = None
    if use_split:
        Qp = Qbase * psi_i                              # (n, n_w)
        dA = alpha * Qp / (Aw + eps_c)
        dB = beta * Qp / (Bw + eps_c)
        dC = beta * Qp / (Cw + eps_c)
        G_wi = np.real(dA @ Emat.T + dB @ E1.T + dC @ E2.T)
        # atom-position gradient, factor by factor
        dL_dy = np.zeros_like(y_atoms)
        for dX, EX, cols in ((dA, Emat, (0, 1)), (dB, E1, (0,)),
                             (dC, E2, (1,))):
            c_aw = dX.T @ Wt                            # (n_w, n_atoms)
            tmp = (c_aw.T * EX) * 1j / kk               # (n_atoms, n_w)
            if cols == (0, 1):
                dL_dy += np.real(tmp @ Omega)
            else:
                dL_dy[:, cols[0]] += np.real(tmp @ Omega[:, cols[0]])
        # exact d loss / d rho
        grad_rho = float(np.real(
            (Qp * (kk * (LA - LB - LC))).sum()))
    else:
        dphi = (k_shots * Qbase) * phi_km1
        G_wi = np.real(dphi @ Emat.T)                   # (n, n_atoms)
        c_aw = dphi.T @ Wt                              # (n_w, n_atoms)
        tmp = (c_aw.T * Emat) * 1j / kk                 # (n_atoms, n_w)
        dL_dy = np.real(tmp @ Omega)                    # (n_atoms, b)

    dL_dW = bank.f_atoms.T @ dL_dy                      # (pf, b)
    C = Minv @ dL_dW
    R = Y - F @ W

    grad_theta = None
    if compute_grad:
        p_theta = theta.size
        grad_theta = np.zeros(p_theta)
        shift = 0.5 * pi
        for kkk in range(p_theta):
            tp = theta.copy(); tp[kkk] += shift
            tm = theta.copy(); tm[kkk] -= shift
            dP = 0.5 * (bank.setting_probs(S, tp) - bank.setting_probs(S, tm))
            dw = bank.atom_weight_derivative(P, dP)     # (n, n_atoms)
            g = float(np.sum(G_wi * dw))
            Gk = bank.expectation_features(dP)
            dSig = bank.parity_covariance_sum(P, dP)
            dM = Gk.T @ F + F.T @ Gk + dSig / kk
            dW = Minv @ (Gk.T @ Y - dM @ W)
            g += float(np.sum(dL_dW * dW))
            if gamma_risk > 0.0:
                g += gamma_risk * (
                    2.0 * float(np.sum(E_res * (Gk @ W)))
                    + float(np.sum(W * (dSig @ W))) / kk) / n
            grad_theta[kkk] = g

    gA = gb = None
    if compute_encoding_grad:
        # per-sample probability-table gradient D[g,i,a]
        D = np.zeros_like(P)
        if bank.G == 3:
            Gw3 = G_wi.reshape(n, bank.dim, bank.dim, bank.dim)
            D[0] += np.einsum("iabc,ib,ic->ia", Gw3, P[1], P[2])
            D[1] += np.einsum("iabc,ia,ic->ib", Gw3, P[0], P[2])
            D[2] += np.einsum("iabc,ia,ib->ic", Gw3, P[0], P[1])
        else:
            Gw2 = G_wi.reshape(n, bank.dim, bank.dim)
            D[0] += np.einsum("iab,ib->ia", Gw2, P[1])
            D[1] += np.einsum("iab,ia->ib", Gw2, P[0])

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
            D[g] += -(q_quad[None, :] - lin) / kk
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
                                        angle_shift=(l, layer, +0.5 * pi))
                Pm = bank.setting_probs(S, theta,
                                        angle_shift=(l, layer, -0.5 * pi))
                acc += 0.5 * (Pp - Pm)
            dL_da[:, l] = np.einsum("gia,gia->i", D, pi * acc)
        sig_prime = a_mat * (1.0 - a_mat)
        gA = (dL_da * sig_prime).T @ S
        gb = (dL_da * sig_prime).sum(axis=0)

    return loss, grad_theta, gA, gb, W, grad_rho

# =========================================================================
# Hardware-faithful (sampled) estimators
# =========================================================================

def sampled_mmd_and_grad_theta(bank: StatevectorBornBank,
                               S: np.ndarray, Y: np.ndarray,
                               theta: np.ndarray, sigmas,
                               W: np.ndarray,
                               n_model_samples: int,
                               rng: np.random.Generator,
                               k: int = 1):
    """
    Unbiased sampled counterpart of the probability term of the exact
    gradient (what one would run on hardware, where probability tables are
    unavailable). For each parameter and each measurement setting g the
    product rule requires records drawn with theta_k shifted ONLY in the
    circuit executions of setting g:

      grad_k = sum_g 1/2 [ E_{записи: g at theta_k^+} D - E_{g at theta_k^-} D ]

    with D(y) = 2( mean_j k(y, y_model_j) - mean_j k(y, Y_j) ).
    Provided for completeness / hardware runs; the exact path above is used
    for the simulator experiments. NOTE: cost is 2 * p_theta * G sample sets.
    """
    theta = np.asarray(theta, np.float64).ravel()
    idx = rng.integers(0, S.shape[0], size=n_model_samples)
    Sb = S[idx]

    def gen_samples(th_per_setting):
        # th_per_setting: list of G theta vectors (one per setting)
        feats = []
        for g in range(bank.G):
            Pg = bank.setting_probs(Sb, th_per_setting[g])[g]
            cdf = np.cumsum(Pg, axis=1); cdf[:, -1] = 1.0
            u = rng.random((Sb.shape[0], k))
            ii = np.stack([np.searchsorted(cdf[i], u[i], side="right")
                           for i in range(Sb.shape[0])])
            feats.append(bank.T[np.clip(ii, 0, bank.dim - 1)].mean(axis=1))
        return np.concatenate(feats, axis=1) @ W

    y_base = gen_samples([theta] * bank.G)

    def D_of(y_probe):
        val = 0.0
        for sigma in sigmas:
            s2 = 2.0 * float(sigma) ** 2
            val += 2.0 * (np.exp(-_sqdist(y_probe, y_base) / s2).mean()
                          - np.exp(-_sqdist(y_probe, Y) / s2).mean())
        return val

    p_theta = theta.size
    grad = np.zeros(p_theta)
    loss = 0.0
    for sigma in sigmas:
        s2 = 2.0 * float(sigma) ** 2
        loss += (np.exp(-_sqdist(y_base, y_base) / s2).mean()
                 - 2.0 * np.exp(-_sqdist(y_base, Y) / s2).mean()
                 + np.exp(-_sqdist(Y, Y) / s2).mean())
    for kk in range(p_theta):
        for g in range(bank.G):
            ths_p = [theta.copy() for _ in range(bank.G)]
            ths_m = [theta.copy() for _ in range(bank.G)]
            ths_p[g][kk] += 0.5 * pi
            ths_m[g][kk] -= 0.5 * pi
            grad[kk] += 0.5 * (D_of(gen_samples(ths_p)) - D_of(gen_samples(ths_m)))
    return float(loss), grad


# =========================================================================
# Block model & generation (residual-gate slot = Born record noise)
# =========================================================================

@dataclass
class BornBlockModel:
    W: np.ndarray
    start: int
    bsz: int
    # Optional intercept, one value per pixel. The Pauli features have no
    # constant term, and the noise-aware correction shrinks W toward zero, so
    # without an intercept the decoded mean is biased low wherever the signal
    # is weak relative to the shot noise. None reproduces the original decoder.
    b0: np.ndarray = None


def _with_intercept(Y, model):
    """Add the block intercept if the model has one; untouched otherwise."""
    b = getattr(model, "b0", None)
    return Y if b is None else Y + b[None, :]


def fit_all_blocks_born(bank: StatevectorBornBank,
                        sketch_cache: np.ndarray,
                        Y_train: np.ndarray,
                        blocks,
                        ridge_alpha: float,
                        noise_aware: bool = True,
                        k_shots: int = 1,
                        intercept: bool = False,
                        noise_weight: float = 1.0) -> List[BornBlockModel]:
    """Closed-form (shot-noise-aware) ridge calibration at bank.theta.

    noise_weight (lambda) weights the shot-noise penalty, as in training: the
    decoders must be fitted with the lambda the circuit was trained with.

    intercept=True adds, per pixel, the gap between the data mean and the
    decoder's expected output, so the generated mean matches the data. The
    weights themselves are unchanged."""
    models = []
    k_arr = np.broadcast_to(np.asarray(k_shots), (len(blocks),))   # scalar or per block
    for bi, (start, bsz) in enumerate(blocks):
        S = sketch_cache[bi].astype(np.float64, copy=False)
        Yb = Y_train[:, start:start + bsz].astype(np.float64, copy=False)
        P = bank.setting_probs(S)
        F = bank.expectation_features(P)
        if noise_aware:
            Sigma_bar = bank.parity_covariance_sum(P)
            W, _ = noise_aware_ridge_fit(F, Yb, noise_weight * Sigma_bar, ridge_alpha,
                                         k=k_arr[bi])
        else:
            W, _ = ridge_fit(F, Yb, alpha=ridge_alpha)
        b0 = (Yb.mean(axis=0) - (F @ W).mean(axis=0)) if intercept else None
        models.append(BornBlockModel(W=W, start=start, bsz=bsz, b0=b0))
    return models


def sample_progressive_born(bank: StatevectorBornBank,
                            models: List[BornBlockModel],
                            d: int, blocks,
                            sketcher,
                            n_samples: int,
                            rng_np: np.random.Generator,
                            k: Optional[int] = None,
                            clip_nonnegative: bool = True,
                            mode: str = "born",
                            record_share: float = 1.0,
                            theta_override: Optional[np.ndarray] = None,
                            return_latent: bool = False):
    """
    Free-running autoregressive generation.

    record_share (rho, "born" mode, block size > 1): fraction of the k
    records per setting SHARED between the pixels of a block; the rest are
    pixel-exclusive draws from the same state. rho controls the intra-block
    noise correlation continuously (rho = 1: fully shared records -> noise
    correlation pinned at the decoder-column angle, the 2x2 "tiling"
    artifact; rho = 0: independent records -> noise-decorrelated pixels).
    Physically it is only bookkeeping of which shots enter which pixel's
    parity average; record cost per setting is k*(1 + (bsz-1)*(1-rho)).
    Calibrated train-side (like k_gen).

    mode:
      "born"        y = f_hat W, f_hat = k-shot Born record parities  
      "mean"        y = F W (deterministic conditional mean; ablation:
                    'shot noise removed' == k -> infinity)
      "classical"   ablation: each parity feature sampled INDEPENDENTLY
                    +-1 with matched mean. Destroys (a) intra-record parity
                    products (P_ij = P_i P_j per record) and (b) the
                    entanglement-induced feature covariance, while keeping
                    E[f_hat] and hence the conditional mean E[y|S] exact.
    """
    k = bank.spec.gen_shots_k if k is None else k
    k_arr = np.asarray(k)
    if k_arr.ndim == 0:
        k_arr = np.full(len(blocks), int(k_arr))
    assert k_arr.shape == (len(blocks),), \
        f"k must be scalar or len(blocks) array, got {k_arr.shape}"
    out = np.zeros((n_samples, d), dtype=np.float64)
    # return_latent=True also returns each value before clipping at zero.
    # The chain itself is unchanged: later blocks still see clipped values.
    latent = np.zeros_like(out) if return_latent else None
    # record_share may be a scalar (one shared rho) or an array of
    # per-block rho_b values of length len(blocks) ().
    rs_arr = np.asarray(record_share, dtype=np.float64)
    if rs_arr.ndim == 0:
        rs_arr = np.full(len(blocks), float(rs_arr))
    assert rs_arr.shape == (len(blocks),), \
        f"record_share must be scalar or len(blocks) array, got {rs_arr.shape}"
    Sraw, cur_len = sketcher.init_state(n_samples)
    for bi, (start, bsz) in enumerate(blocks):
        rs = float(rs_arr[bi])
        kk = int(k_arr[bi])
        Sprefix = sketcher.mixed(Sraw, cur_len)
        P = bank.setting_probs(Sprefix, theta_override)
        if mode == "born" and (rs < 1.0 and bsz > 1):
            k_sh = int(round(rs * kk))
            k_ex = kk - k_sh
            ktot = k_sh + bsz * k_ex
            rec = bank.sample_records(P, ktot, rng_np)
            W = models[bi].W
            cols = []
            for j in range(bsz):
                sl = np.concatenate([np.arange(k_sh),
                                     k_sh + j * k_ex + np.arange(k_ex)])
                f_j = np.concatenate(
                    [bank.T[rec[g][:, sl]].mean(axis=1)
                     for g in range(bank.G)], axis=1)
                cols.append(f_j @ W[:, j])
            Yblk = _with_intercept(np.stack(cols, axis=1), models[bi])
            if return_latent:
                latent[:, start:start + bsz] = Yblk
            if clip_nonnegative:
                Yblk = np.maximum(Yblk, 0.0)
            out[:, start:start + bsz] = Yblk
            cur_len = sketcher.update_inplace(Sraw, cur_len, Yblk)
            continue
        if mode == "born":
            f_hat = bank.sample_record_features(P, kk, rng_np)
        elif mode == "mean":
            f_hat = bank.expectation_features(P)
        elif mode == "classical":
            F = bank.expectation_features(P)
            pr_plus = 0.5 * (1.0 + np.clip(F, -1.0, 1.0))
            f_hat = np.where(
                rng_np.random((n_samples * kk, bank.pf)).reshape(kk, n_samples, bank.pf)
                < pr_plus[None], 1.0, -1.0).mean(axis=0)
        else:
            raise ValueError(f"unknown mode={mode}")
        Yblk = _with_intercept(f_hat @ models[bi].W, models[bi])
        if return_latent:
            latent[:, start:start + bsz] = Yblk
        if clip_nonnegative:
            Yblk = np.maximum(Yblk, 0.0)
        out[:, start:start + bsz] = Yblk
        cur_len = sketcher.update_inplace(Sraw, cur_len, Yblk)
    return (out, latent) if return_latent else out


def predict_future_born(bank: StatevectorBornBank,
                        models: List[BornBlockModel],
                        d: int, blocks, sketcher,
                        Y_partial: np.ndarray,
                        clip_nonnegative: bool = True) -> np.ndarray:
    """Mean-only prediction of unseen dims (uses E[y|S] = F W; no sampling)."""
    Y_partial = np.asarray(Y_partial, np.float64)
    n, kobs = Y_partial.shape
    out = np.zeros((n, d), dtype=np.float64)
    out[:, :kobs] = Y_partial
    Sraw, cur_len = sketcher.init_state(n)
    for bi, (start, bsz) in enumerate(blocks):
        end = start + bsz
        if end <= kobs:
            cur_len = sketcher.update_inplace(Sraw, cur_len, Y_partial[:, start:end])
            continue
        Sprefix = sketcher.mixed(Sraw, cur_len)
        P = bank.setting_probs(Sprefix)
        F = bank.expectation_features(P)
        block = _with_intercept(F @ models[bi].W, models[bi])
        if clip_nonnegative:
            block = np.maximum(block, 0.0)
        if start < kobs < end:
            block[:, : kobs - start] = Y_partial[:, start:kobs]
        out[:, start:end] = block
        cur_len = sketcher.update_inplace(Sraw, cur_len, block)
    out[:, :kobs] = Y_partial
    return out


def rollout_refine_born(bank: StatevectorBornBank,
                        models: List[BornBlockModel],
                        Y_train: np.ndarray, blocks, sketcher,
                        ridge_alpha: float,
                        epochs: int = 6, max_rollout_ratio: float = 0.85,
                        monitor_n: int = 512, seed: int = 2025,
                        clip_nonnegative: bool = True,
                        k_shots: int = 1,
                        verbose: bool = True) -> List[BornBlockModel]:
    """DAgger-lite rollout refinement of the ridge calibration (unchanged
    from  in structure; only the residual source differs)."""
    if epochs <= 0:
        return models
    from .sketch import build_prefix_sketch_cache
    from .metrics import correlation_error_summary
    from .utils import corr_nan_safe

    Y_train = np.asarray(Y_train, np.float64)
    n, d = Y_train.shape
    prng = np.random.default_rng(seed)
    if verbose:
        print("=" * 78)
        print(f"ROLLOUT REFINEMENT (Born)  epochs={epochs}  "
              f"max_ratio={max_rollout_ratio}")
        print("=" * 78)
    for ep in range(epochs):
        rr = float(max_rollout_ratio) * float(ep + 1) / float(epochs)
        Y_roll = sample_progressive_born(
            bank, models, d, blocks, sketcher, n_samples=n, rng_np=prng,
            k=k_shots, clip_nonnegative=clip_nonnegative)
        Y_cond = (1.0 - rr) * Y_train + rr * Y_roll
        if clip_nonnegative:
            Y_cond = np.maximum(Y_cond, 0.0)
        cache = build_prefix_sketch_cache(Y_cond, blocks, sketcher)
        models = fit_all_blocks_born(bank, cache, Y_train, blocks,
                                     ridge_alpha, k_shots=k_shots)
        mon_n = min(int(monitor_n), n)
        idx = prng.choice(n, size=mon_n, replace=False)
        Y_mon = sample_progressive_born(
            bank, models, d, blocks, sketcher, n_samples=mon_n, rng_np=prng,
            k=k_shots, clip_nonnegative=clip_nonnegative)
        s = correlation_error_summary(
            corr_nan_safe(Y_train[idx]), corr_nan_safe(Y_mon), blocks)
        if verbose:
            print(f"  refine {ep+1:2d}/{epochs}  rr={rr:.3f}  "
                  f"offdiag={s['corr_mae_offdiag']:.5f}  "
                  f"within={s['corr_mae_within']:.5f}  "
                  f"cross={s['corr_mae_cross']:.5f}")
    return models
