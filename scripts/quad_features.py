"""
Quadratic record features.

Augments the parity expectations with exact record-level second moments, so the
decoder sees products of parities as well as the parities themselves. This
widens the feature space without touching the circuit, which matters because
the decoder rank, not the circuit, is what limits block size.

Provides ExtFeatures, imported by scripts/evaluate.py. """

import pathlib
import sys

import numpy as np
from scipy.stats import wasserstein_distance

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qfan import correlation_error_summary, corr_nan_safe               # noqa
from qfan.born import (fit_all_blocks_born, sample_progressive_born,     # noqa
                       noise_aware_ridge_fit, BornBlockModel)
from train import (build_problem, MonotoneMarginalMap,     # noqa
                                 spearman_corr, K_SHOTS)
from calibration import corr_only_R, apply_R                   # noqa

GEN_BATCH = 1200      # memory-bounded generation batches


# ----------------------------------------------------------------------
# extended feature machinery
# ----------------------------------------------------------------------
class ExtFeatures:
    def __init__(self, bank):
        self.bank = bank
        self.G = bank.G
        T = bank.T                                   # (8, 6)
        sup = [set(s) for s in bank.parity_supports]
        full = set(range(bank.spec.n_qubits))
        # find two columns whose supports partition {0,1,2} -> triple parity
        pair = None
        for a in range(len(sup)):
            for b in range(len(sup)):
                if a != b and (sup[a] | sup[b]) == full and not (sup[a] & sup[b]):
                    pair = (a, b)
                    break
            if pair:
                break
        t3 = T[:, pair[0]] * T[:, pair[1]]
        self.Text = np.concatenate([T, t3[:, None]], axis=1)   # (8, 7)
        self.nf = self.Text.shape[1]                            # 7
        self.pairs = [(g, gp) for g in range(self.G)
                      for gp in range(g + 1, self.G)]
        self.pf_ext = self.G * self.nf + len(self.pairs) * self.nf ** 2
        # second-moment table: E[Text_a Text_b] = p @ TT (8, 49)
        self.TT = np.einsum("xa,xb->xab", self.Text, self.Text)\
                    .reshape(self.Text.shape[0], self.nf * self.nf)

    def mu_M(self, P):
        """per-group means (G, n, nf) and second moments (G, n, nf, nf)."""
        n = P.shape[1]
        mu = np.stack([P[g] @ self.Text for g in range(self.G)])
        M = np.stack([(P[g] @ self.TT).reshape(n, self.nf, self.nf)
                      for g in range(self.G)])
        return mu, M

    def expectation(self, P):
        """E[f_ext] (n, pf_ext): [a-features; c-features mu_i mu_j]."""
        mu, _ = self.mu_M(P)
        n = P.shape[1]
        parts = [mu[g] for g in range(self.G)]
        for (g, gp) in self.pairs:
            parts.append(np.einsum("ni,nj->nij", mu[g], mu[gp])
                         .reshape(n, -1))
        return np.concatenate(parts, axis=1)

    def sigma_bar(self, P):
        """sum_i Cov_1[f_ext | S_i]  (pf_ext, pf_ext), exact."""
        mu, M = self.mu_M(P)
        C = mu.shape[1]
        nf, G = self.nf, self.G
        pe = self.pf_ext
        Sig = np.zeros((pe, pe))
        D = M - np.einsum("gni,gnj->gnij", mu, mu)     # (G, n, nf, nf)
        off = {g: g * nf for g in range(G)}
        offc = {}
        base = G * nf
        for (g, gp) in self.pairs:
            offc[(g, gp)] = base
            base += nf * nf
        # a-a (within group)
        for g in range(G):
            Sig[off[g]:off[g]+nf, off[g]:off[g]+nf] += D[g].sum(0)
        # a-c (a in one of the pair's groups)
        for (g, gp) in self.pairs:
            oc = offc[(g, gp)]
            # Cov(a_i^g, c_{jl}) = D_ij^g mu_l^{gp}
            B = np.einsum("nij,nl->nijl", D[g], mu[gp]).sum(0)\
                  .reshape(nf, nf * nf)
            Sig[off[g]:off[g]+nf, oc:oc+nf*nf] += B
            Sig[oc:oc+nf*nf, off[g]:off[g]+nf] += B.T
            # Cov(a_j^{gp}, c_{il}) with c indexed (i from g, l from gp):
            # = mu_i^g D_jl^{gp}
            B2 = np.einsum("ni,njl->njil", mu[g], D[gp]).sum(0)\
                   .reshape(nf, nf * nf)
            Sig[off[gp]:off[gp]+nf, oc:oc+nf*nf] += B2
            Sig[oc:oc+nf*nf, off[gp]:off[gp]+nf] += B2.T
        # c-c same pair: M_il^g M_jm^{gp} - mu_i mu_l mu_j mu_m
        for (g, gp) in self.pairs:
            oc = offc[(g, gp)]
            A = np.einsum("nil,njm->nijlm", M[g], M[gp]).sum(0)
            Bm = np.einsum("ni,nl,nj,nm->nijlm", mu[g], mu[g],
                           mu[gp], mu[gp]).sum(0)
            Sig[oc:oc+nf*nf, oc:oc+nf*nf] += (A - Bm)\
                .reshape(nf*nf, nf*nf)
        # c-c sharing one group
        for i1, (g, gp) in enumerate(self.pairs):
            for (h, hp) in self.pairs[i1+1:]:
                shared = ({g, gp} & {h, hp})
                if not shared:
                    continue
                s = shared.pop()
                o1, o2 = offc[(g, gp)], offc[(h, hp)]
                u1 = gp if s == g else g       # non-shared group of pair 1
                u2 = hp if s == h else h       # non-shared group of pair 2
                # Cov = D_(s-indices) * mu(u1 index) * mu(u2 index)
                # c1 index (i from g, j from gp): shared-idx position
                #   depends on whether s is first or second in the pair.
                def block(Dn, mu1, mu2, s_first_1, s_first_2):
                    # returns (nf,nf,nf,nf) with axes (i1a, i1b, i2a, i2b)
                    # where the pair's own index order is (first, second)
                    E = np.einsum("nab,nc,nd->nabcd", Dn, mu1, mu2).sum(0)
                    # E axes: (shared1, shared2, u1, u2) -> reorder into
                    # (c1 first, c1 second, c2 first, c2 second)
                    if s_first_1 and s_first_2:
                        return E.transpose(0, 2, 1, 3)
                    if s_first_1 and not s_first_2:
                        return E.transpose(0, 2, 3, 1)
                    if not s_first_1 and s_first_2:
                        return E.transpose(2, 0, 1, 3)
                    return E.transpose(2, 0, 3, 1)
                Bc = block(D[s], mu[u1], mu[u2], s == g, s == h)\
                    .reshape(nf*nf, nf*nf)
                Sig[o1:o1+nf*nf, o2:o2+nf*nf] += Bc
                Sig[o2:o2+nf*nf, o1:o1+nf*nf] += Bc.T
        return Sig

    def record_features(self, rec_T, sl):
        """rec_T: list over g of (n, ktot, nf) Text-values; sl: record
        index subset for this pixel. Returns (n, pf_ext)."""
        A = [rt[:, sl] for rt in rec_T]                 # (n, |sl|, nf)
        ksl = len(sl)
        parts = [a.mean(axis=1) for a in A]
        for (g, gp) in self.pairs:
            parts.append(np.einsum("nri,nrj->nij", A[g], A[gp])
                         .reshape(A[g].shape[0], -1) / ksl)
        return np.concatenate(parts, axis=1)


def mc_selfcheck(ext, bank, S, k=64, n_mc=160000, seed=0):
    """Hard verification of analytic mean and covariance vs Monte-Carlo."""
    P = bank.setting_probs(S[:1])
    rng = np.random.default_rng(seed)
    F = ext.expectation(P)[0]
    Sig1 = ext.sigma_bar(P)                     # single sample -> Cov_1
    rec = bank.sample_records(P, n_mc, rng)     # (G, 1, n_mc)
    rec_T = [ext.Text[rec[g]] for g in range(bank.G)]
    # split MC records into n_mc/k groups of k to form k-shot features
    m = n_mc // k
    feats = np.stack([ext.record_features(rec_T, np.arange(i*k, (i+1)*k))[0]
                      for i in range(m)])
    err_mu = np.abs(feats.mean(0) - F).max()
    C_mc = np.cov(feats, rowvar=False) * k      # -> single-record scale
    scale = max(np.abs(Sig1).max(), 1e-9)
    err_C = np.abs(C_mc - Sig1).max() / scale
    print(f"[MC CHECK] m={n_mc//k} k-shot samples:  "
          f"max|mean err|={err_mu:.4f} (tol .02)   "
          f"max rel|cov err|={err_C:.4f} (tol .10; MC-noise floor "
          f"~{2.8/np.sqrt(n_mc//k):.3f})")
    assert err_mu < 0.02 and err_C < 0.10, "analytic moments WRONG"
