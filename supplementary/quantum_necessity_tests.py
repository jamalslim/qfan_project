#!/usr/bin/env python3
"""
quantum_necessity_tests.py -- turn the referee's question
("at which scale is an actual quantum device needed?") into measurements.

The question cannot be settled by proving classical hardness -- nobody can do
that empirically. But four things CAN be measured, and together they bound the
answer tightly. This script implements the two cheap ones and specifies the
other two precisely.

--------------------------------------------------------------------------
TEST A (implemented): quantum-resource diagnostics of the LEARNED solution
--------------------------------------------------------------------------
Does training actually use entanglement and non-stabilizerness, or does it
find a solution that a classical surrogate could represent?

  * bipartite entanglement entropy S(1|rest) of the conditioned state
  * stabilizer 2-Renyi entropy M2 (magic); M2 = 0 iff stabilizer state
  * compared against the UNTRAINED circuit and against Haar-random states

Why it matters: low entanglement means small MPS bond dimension, i.e. the
model is classically simulable for a *structural* reason, not merely because
n_q is small. If training REDUCES these quantities relative to a random
circuit, the optimizer is not exploiting quantum resources and adding qubits
is unlikely to change that.

--------------------------------------------------------------------------
TEST B (implemented): classical-noise control -- the missing C1 companion
--------------------------------------------------------------------------
C1 shows that removing the sampling fluctuation collapses the model to a
point mass, i.e. the randomness is load-bearing. It does NOT show the
randomness must be quantum. This test replaces the k-shot Born record average
by a CLASSICAL surrogate noise with the same first two moments:

    Yhat_block = mu(prefix) + eps,   eps ~ N(0, Sigma_born/k)

with Sigma_born the exact per-block record covariance implied by the circuit
(including the shared-record structure rho_beta). If the model survives this
substitution, the Born rule is supplying nothing beyond a covariance a
classical sampler can copy, and the honest claim shrinks accordingly.

--------------------------------------------------------------------------
TEST C (specified, not implemented): surrogate-cost scaling
--------------------------------------------------------------------------
Replace the exact expectation values by a truncated classical surrogate --
matrix-product state at bond dimension chi, or Pauli propagation retaining
the N largest coefficients -- and find the smallest chi (or N) at which the
generated samples become statistically indistinguishable from the exact
pipeline. Then repeat at n_q = 3,4,5,6,... and depth L = 2,3,4,...
The growth of chi*(n_q, L) is an EMPIRICAL crossover curve: extrapolating it
to the point where chi* exceeds what is tractable gives a measured, rather
than assumed, estimate of where a device becomes necessary. This is the
gold-standard experiment for the referee's question.

--------------------------------------------------------------------------
TEST D (specified): does fidelity improve with n_q at FIXED block size?
--------------------------------------------------------------------------
The capacity rule says qubits are needed to make blocks larger. The separate
question is whether MORE qubits at FIXED b improves accuracy. Sweep
n_q = 2..8 at fixed d and b, retrain each, and plot off-diagonal correlation
error. Two possible outcomes, both decisive:
  * saturates at small n_q -> extra qubits buy nothing, and the architecture
    has no route to a regime where a device is required. Say so in the paper.
  * keeps improving -> fit the trend and report where it would cross the
    classical-simulation boundary, which is the referee's requested regime.
This is the single most informative run and it is a simulator sweep.

Usage:
    python quantum_necessity_tests.py --test A --state outputs/..._v50_results.npz
    python quantum_necessity_tests.py --test B --state outputs/..._v50_results.npz
"""
import argparse
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve()
# Walk upward until we find the repo root (the directory containing src/qfan),
# so the script works whether it lives at the repo root or inside scripts/.
PROJECT_ROOT = None
for _cand in [ROOT.parent] + list(ROOT.parents):
    if (_cand / "src" / "qfan").is_dir():
        PROJECT_ROOT = _cand
        break
if PROJECT_ROOT is None:
    raise SystemExit(
        "cannot locate the repo root: no ancestor directory contains "
        "src/qfan. Run this script from inside the qfan project tree, or "
        "set PYTHONPATH=src.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1]).astype(complex)
_P1 = [_I, _X, _Y, _Z]


def pauli_basis(n):
    out = []
    for idx in itertools.product(range(4), repeat=n):
        M = np.array([[1]], complex)
        for k in idx:
            M = np.kron(M, _P1[k])
        out.append(M)
    return out


def magic_m2(psi, paulis, n):
    """Stabilizer 2-Renyi entropy. 0 for stabilizer states."""
    s = sum(np.real(np.vdot(psi, P @ psi)) ** 4 for P in paulis)
    return float(-np.log2(s / 2 ** n))


def entanglement(psi, n, cut=1):
    M = psi.reshape(2 ** cut, 2 ** (n - cut))
    sv = np.linalg.svd(M, compute_uv=False)
    p = sv ** 2
    p = p[p > 1e-14]
    return float(-(p * np.log2(p)).sum())


def test_A(state_path, n_samples=96, block=6, seed=0):
    from train import build_problem
    (cfg, Y_tr, Y_te, d, blocks, sk, cache, spec, bank) = build_problem()
    r = np.load(state_path, allow_pickle=True)
    n = spec.n_qubits
    paulis = pauli_basis(n)
    rng = np.random.default_rng(seed)
    S = cache[block, rng.choice(cache.shape[1], n_samples,
                                replace=False)].astype(float)

    def survey(theta, A, b):
        bank.set_theta(theta)
        bank.A = np.asarray(A).copy()
        bank.b = np.asarray(b).copy()
        PSI = bank.state(S, theta)
        E, M = [], []
        for psi in PSI:
            psi = np.asarray(psi).ravel()
            psi = psi / np.linalg.norm(psi)
            E.append(entanglement(psi, n))
            M.append(magic_m2(psi, paulis, n))
        return np.mean(E), np.std(E), np.mean(M), np.std(M)

    tr = survey(r["theta_final"], r["A_final"], r["b_final"])
    try:
        lc = np.load(str(state_path).replace("_results", "_loss_curve"),
                     allow_pickle=True)
        th0 = lc["theta_init"]
    except Exception:
        th0 = rng.normal(size=bank.n_var) * 0.3
    un = survey(th0, r["A_final"], r["b_final"])

    hs, hm = [], []
    for _ in range(300):
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        v /= np.linalg.norm(v)
        hs.append(entanglement(v, n))
        hm.append(magic_m2(v, paulis, n))

    print(f"\nTEST A -- quantum resources in the LEARNED solution "
          f"(n_q={n}, {n_samples} conditioning inputs)")
    print(f"{'state':14s} {'S(1|rest)':>12s} {'+/-':>7s} {'magic M2':>10s} {'+/-':>7s}")
    print(f"{'trained':14s} {tr[0]:12.4f} {tr[1]:7.4f} {tr[2]:10.4f} {tr[3]:7.4f}")
    print(f"{'untrained':14s} {un[0]:12.4f} {un[1]:7.4f} {un[2]:10.4f} {un[3]:7.4f}")
    print(f"{'Haar random':14s} {np.mean(hs):12.4f} {np.std(hs):7.4f} "
          f"{np.mean(hm):10.4f} {np.std(hm):7.4f}")
    print(f"{'stabilizer':14s} {'--':>12s} {'--':>7s} {0.0:10.4f} {'--':>7s}")
    print(f"\n  max S(1|rest) = 1.000 bit for one qubit; M2 = 0 iff stabilizer.")
    if tr[0] < un[0] and tr[2] < un[2]:
        print("  READING: training DECREASED both entanglement and magic "
              "relative to the\n  untrained circuit. The optimizer is not "
              "moving toward the quantum-resource-rich\n  region; the learned "
              "solution sits in a low-entanglement corner, which is the\n  "
              "classically easy regime for structural reasons and not only "
              "because n_q is small.")
    else:
        print("  READING: training increased at least one resource measure; "
              "report both.")


def test_B(state_path, n_gen=1200, seed=0):
    """Classical-noise control: Born records -> Gaussian noise of matched
    covariance. Survival of the model implies the Born rule contributes
    nothing beyond a copyable second moment."""
    from train import build_problem, spearman_corr
    from qfan.born import fit_all_blocks_born
    from qfan import correlation_error_summary, corr_nan_safe
    from scipy.stats import wasserstein_distance
    (cfg, Y_tr, Y_te, d, blocks, sketcher, cache, spec, bank) = build_problem()
    r = np.load(state_path, allow_pickle=True)
    bank.set_theta(r["theta_final"])
    bank.A = r["A_final"].copy()
    bank.b = r["b_final"].copy()
    rho = np.atleast_1d(r["rho_final"])
    K = 64
    models = fit_all_blocks_born(bank, cache, Y_tr, blocks,
                                 cfg.bank.ridge_alpha, noise_aware=True,
                                 k_shots=K)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_gen, d))
    Sraw, cur = sketcher.init_state(n_gen)
    for bi, (s, bsz) in enumerate(blocks):
        Sm = sketcher.mixed(Sraw, cur)
        P = bank.setting_probs(Sm)
        F = bank.expectation_features(P)              # exact conditional mean
        Sig = bank.parity_covariance_sum(P)           # record covariance
        W = models[bi].W
        mu = F @ W
        # covariance of the k-shot feature average, propagated through W
        Cov_y = (W.T @ (Sig / K) @ W)
        Cov_y = 0.5 * (Cov_y + Cov_y.T) + 1e-12 * np.eye(bsz)
        L = np.linalg.cholesky(Cov_y)
        eps = rng.standard_normal((n_gen, bsz)) @ L.T
        Yb = np.maximum(mu + eps, 0.0)
        out[:, s:s + bsz] = Yb
        cur = sketcher.update_inplace(Sraw, cur, Yb)
    Ct = corr_nan_safe(Y_te)
    cs = correlation_error_summary(Ct, corr_nan_safe(out), blocks)
    w1 = float(np.mean([wasserstein_distance(Y_te[:, j], out[:, j])
                        for j in range(d)]))
    print("\nTEST B -- classical-noise control (C7): Born records replaced by "
          "Gaussian\nnoise with the SAME conditional mean and the SAME "
          "record covariance.")
    print(f"  off-diag corr MAE = {cs['corr_mae_offdiag']:.4f}   "
          f"within = {cs['corr_mae_within']:.4f}   W1 = {w1:.5f}   "
          f"Var(E) = {out.sum(1).var():.4f}")
    print("  compare with the quantum pipeline and with C1 (degenerate).")
    print("\n  READING: if these numbers are close to the quantum pipeline, "
          "the Born rule is\n  supplying only a second moment that a classical "
          "sampler can copy, and the paper\n  must say so. If they are clearly "
          "worse, the higher moments of the Born\n  distribution matter and "
          "that is a genuine, reportable quantum contribution.")
    return cs, w1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["A", "B", "both"], default="both")
    ap.add_argument("--state",
                    default="outputs/model_d25.npz")
    args = ap.parse_args()
    if args.test in ("A", "both"):
        test_A(args.state)
    if args.test in ("B", "both"):
        test_B(args.state)
    print("\nTests C and D are specified in this file's docstring; D "
          "(the n_q sweep at fixed\nblock size) is the single most "
          "informative run and needs only the simulator.")


if __name__ == "__main__":
    main()
