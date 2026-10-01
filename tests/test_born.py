"""
Test suite for the QFAN circuit engine, gradients and decoder.

Run:  python tests/test_born.py

Checks (all must pass before any experiment is believed):
  T1  statevector engine vs INDEPENDENT dense-unitary (Kronecker) build
  T2  norm preservation, probability normalization (all settings)
  T3  parameter-shift dP/dtheta_k vs central finite differences
  T4  full exact loss gradient (probability + ridge-implicit terms)
      vs central finite differences of L(theta) with ridge refit inside
  T5  E[sampled record parities] -> expectation features (LLN, statistical)
  T6  intra-record parity algebra: P_ij = P_i * P_j per record (structural)
  T7  expectation features == Aer shot-bank Pauli expectations
      (only if qiskit+aer installed; skipped otherwise)
  T8  exact atomic MMD == sampled MMD (LLN, statistical)
  T10 CF-MMD exact dL/dtheta vs FD (k-shot pushforward)
  T11 CF-MMD exact encoding gradient (A, b) vs FD
  T12 CF identity phi(w/k)^k vs brute-force sampled k-shot generation
  T13 conditional (joint prefix-block) CF-MMD gradients vs FD
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qfan.born import (BornBankSpec, StatevectorBornBank,          # noqa: E402
                       born_block_loss_and_grad)
from qfan.mmd import median_sigma, mmd2_rbf                        # noqa: E402
from qfan.ridge import ridge_fit                                   # noqa: E402
from qfan.utils import _stable_sigmoid                             # noqa: E402

PASS = []


def check(name, ok, detail=""):
    PASS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------- T1
def dense_state(bank, S_row, theta):
    """Independent implementation: sequential 8x8 dense unitaries."""
    from math import pi
    I2 = np.eye(2, dtype=np.complex128)

    def op1(U, q, nq):
        mats = [I2] * nq
        mats[q] = U
        # qubit q is bit q => index = sum b_q 2^q; kron with qubit (nq-1) leftmost
        out = mats[nq - 1]
        for qq in range(nq - 2, -1, -1):
            out = np.kron(out, mats[qq])
        return out

    def ry(phi):
        c, s = np.cos(phi / 2), np.sin(phi / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    def rz(phi):
        return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])

    nq, dim = bank.nq, bank.dim
    a = _stable_sigmoid(S_row @ bank.A.T + bank.b)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    t = 0
    for _layer in range(bank.depth):
        for k in range(bank.L):
            q = k % nq
            U = ry(pi * a[k]) if k % 2 == 0 else rz(pi * a[k])
            psi = op1(U, q, nq) @ psi
        for q in range(nq):
            psi = op1(rz(theta[t]), q, nq) @ psi; t += 1
            psi = op1(ry(theta[t]), q, nq) @ psi; t += 1
        pairs = [(q, q + 1) for q in range(nq - 1)] + ([(nq - 1, 0)] if nq > 2 else [])
        for (i, j) in pairs:
            CZ = np.eye(dim, dtype=np.complex128)
            for idx in range(dim):
                if ((idx >> i) & 1) and ((idx >> j) & 1):
                    CZ[idx, idx] = -1.0
            psi = CZ @ psi
    return psi


def t1_t2():
    rng = np.random.default_rng(0)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    S = rng.normal(size=(5, 32))
    theta = rng.normal(size=bank.n_var)
    psi_fast = bank.state(S, theta)
    err = max(np.abs(psi_fast[i] - dense_state(bank, S[i], theta)).max()
              for i in range(5))
    check("T1 engine vs dense-unitary", err < 1e-12, f"max|dpsi|={err:.2e}")
    P = bank.setting_probs(S, theta)
    nrm = np.abs(P.sum(axis=2) - 1.0).max()
    check("T2 probability normalization (Z,X,Y)", nrm < 1e-12, f"max|1-sum|={nrm:.2e}")


# ---------------------------------------------------------------- T3
def t3():
    rng = np.random.default_rng(1)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    S = rng.normal(size=(4, 32))
    theta = rng.normal(size=bank.n_var)
    eps = 1e-6
    worst = 0.0
    for k in range(bank.n_var):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += 0.5 * np.pi
        tm[k] -= 0.5 * np.pi
        dP_shift = 0.5 * (bank.setting_probs(S, tp) - bank.setting_probs(S, tm))
        tp2, tm2 = theta.copy(), theta.copy()
        tp2[k] += eps
        tm2[k] -= eps
        dP_fd = (bank.setting_probs(S, tp2) - bank.setting_probs(S, tm2)) / (2 * eps)
        worst = max(worst, np.abs(dP_shift - dP_fd).max())
    check("T3 parameter-shift dP vs FD", worst < 1e-8, f"max err={worst:.2e}")


# ---------------------------------------------------------------- T4
def t4():
    rng = np.random.default_rng(2)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    n, b = 24, 4
    S = rng.normal(size=(n, 32))
    Y = np.abs(rng.normal(0.05, 0.04, size=(n, b)))
    theta = rng.normal(size=bank.n_var)
    sig = median_sigma(Y, Y)
    sigmas = [0.5 * sig, sig, 2.0 * sig]
    alpha = 1e-2

    for clip, gamma in ((False, 0.0), (True, 0.0), (True, 10.0)):
        kw = dict(clip_atoms=clip, gamma_risk=gamma)
        L0, grad, _, _ = born_block_loss_and_grad(
            bank, S, Y, theta, sigmas, alpha, compute_grad=True,
            include_ridge_term=True, **kw)

        eps = 1e-6
        fd = np.zeros_like(grad)
        for k in range(bank.n_var):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += eps
            tm[k] -= eps
            Lp, _, _, _ = born_block_loss_and_grad(
                bank, S, Y, tp, sigmas, alpha, compute_grad=False, **kw)
            Lm, _, _, _ = born_block_loss_and_grad(
                bank, S, Y, tm, sigmas, alpha, compute_grad=False, **kw)
            fd[k] = (Lp - Lm) / (2 * eps)
        rel = np.linalg.norm(grad - fd) / max(1e-30, np.linalg.norm(fd))
        tol = 1e-6 if not clip else 1e-5   # clip: a.e.-subgradient, generic pts
        check(f"T4 exact dL/dtheta vs FD (clip={clip}, gamma={gamma})",
              rel < tol, f"rel err={rel:.2e}  ||g||={np.linalg.norm(grad):.3e}")


# ---------------------------------------------------------------- T5, T6
def t5_t6():
    rng = np.random.default_rng(3)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    S = rng.normal(size=(3, 32))
    P = bank.setting_probs(S)
    F = bank.expectation_features(P)
    N = 100_000
    acc = bank.sample_record_features(P, N, rng)   # N-shot parity average
    err = np.abs(acc - F).max()
    check("T5 E[record parities] -> Pauli expectations", err < 0.015,
          f"max dev={err:.2e} at N={N} (4-sigma tol)")

    # structural: for the Z-record features [P0,P1,P2,P01,P02,P12],
    # each atom must satisfy P01 = P0*P1 etc.
    fa = bank.f_atoms
    ok = True
    for g in range(bank.G):
        base = 6 * g
        ok &= np.allclose(fa[:, base + 3], fa[:, base + 0] * fa[:, base + 1])
        ok &= np.allclose(fa[:, base + 4], fa[:, base + 0] * fa[:, base + 2])
        ok &= np.allclose(fa[:, base + 5], fa[:, base + 1] * fa[:, base + 2])
    check("T6 intra-record parity algebra P_ij = P_i P_j", ok)


# ---------------------------------------------------------------- T7
def t7():
    try:
        from qiskit.quantum_info import Statevector          # noqa: F401
        from qfan.quantum import ShotBankSpec, TrainableShotPauliBank
    except Exception:
        print("  [SKIP] T7 qiskit not installed (run on a machine with "
              "qiskit+aer to cross-check against the Aer shot bank)")
        return
    rng = np.random.default_rng(4)
    spec3 = BornBankSpec(include_y=False)          # the reference bank measures Z and X only
    bank = StatevectorBornBank(32, spec3, seed=11)
    ref = TrainableShotPauliBank(32, ShotBankSpec(shots=200_000), seed=11)
    S = rng.normal(size=(2, 32))
    theta = rng.normal(size=bank.n_var)
    F4 = bank.expectation_features(bank.setting_probs(S, theta))
    F3 = ref.features(S, theta_override=theta)
    # map the reference Pauli order onto the engine's feature layout
    m = {}
    for j, p in enumerate(ref.paulis):
        axes = {c for c in p if c != 'I'}
        sup = tuple(i for i, c in enumerate(p) if c != 'I')
        ax = axes.pop()
        base = {"Z": 0, "X": 6}[ax]
        singles = [(q,) for q in range(3)]
        doubles = [(0, 1), (0, 2), (1, 2)]
        order = singles + doubles
        m[j] = base + order.index(sup)
    F3m = np.zeros_like(F4)
    for j, jj in m.items():
        F3m[:, jj] = F3[:, j]
    err = np.abs(F4 - F3m).max()
    check("T7 exact expectations vs Aer shot bank", err < 2e-2,
          f"max dev={err:.2e} (shot noise at 2e5 shots)")


# ---------------------------------------------------------------- T8
def t8():
    rng = np.random.default_rng(5)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    n = 64
    S = rng.normal(size=(n, 32))
    Y = np.abs(rng.normal(0.05, 0.04, size=(n, 6)))
    W, _ = ridge_fit(
        bank.expectation_features(bank.setting_probs(S)), Y, alpha=1e-2)
    sig = median_sigma(Y, Y)

    # exact atomic law
    P = bank.setting_probs(S)
    wbar = bank.atom_weights(P).mean(axis=0)
    y_atoms = bank.f_atoms @ W
    from qfan.mmd import _sqdist
    s2 = 2 * sig ** 2
    L_exact = float(
        wbar @ np.exp(-_sqdist(y_atoms, y_atoms) / s2) @ wbar
        - 2 * wbar @ np.exp(-_sqdist(y_atoms, Y) / s2).mean(axis=1)
        + np.exp(-_sqdist(Y, Y) / s2).mean())

    reps = 60
    Ls = []
    for _ in range(reps):
        f_hat = bank.sample_record_features(P, 1, rng)
        Ls.append(mmd2_rbf(f_hat @ W, Y, sig))
    # sampled biased V-stat has a +E k(y,y)/n_model bias vs population; use
    # large effective sample by pooling
    pool = np.vstack([bank.sample_record_features(P, 1, rng) @ W
                      for _ in range(40)])
    L_pool = mmd2_rbf(pool, Y, sig)
    dev = abs(L_pool - L_exact)
    check("T8 exact atomic MMD vs sampled MMD", dev < 0.02,
          f"exact={L_exact:.5f}  sampled(pooled)={L_pool:.5f}  |d|={dev:.3e}")


# ---------------------------------------------------------------- T10/T11
def t10_t11():
    from qfan.born import cf_mmd_loss_and_grads, draw_rff_frequencies
    rng = np.random.default_rng(7)
    bank = StatevectorBornBank(16, BornBankSpec(include_y=True), seed=3)
    n, b, kshots = 18, 4, 16
    S = rng.normal(size=(n, 16))
    Y = np.abs(rng.normal(0.05, 0.04, size=(n, b)))
    theta = rng.normal(size=bank.n_var)
    sig = median_sigma(Y, Y)
    Om = draw_rff_frequencies([0.5 * sig, sig, 2 * sig], b, 90, seed=1)
    alpha, gamma = 1e-2, 10.0

    L0, gth, gA, gb, _, _ = cf_mmd_loss_and_grads(
        bank, S, Y, theta, Om, alpha, k_shots=kshots, gamma_risk=gamma,
        compute_grad=True, compute_encoding_grad=True)

    eps = 1e-6
    fd = np.zeros_like(gth)
    for k in range(bank.n_var):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps; tm[k] -= eps
        Lp = cf_mmd_loss_and_grads(bank, S, Y, tp, Om, alpha, k_shots=kshots,
                                   gamma_risk=gamma, compute_grad=False)[0]
        Lm = cf_mmd_loss_and_grads(bank, S, Y, tm, Om, alpha, k_shots=kshots,
                                   gamma_risk=gamma, compute_grad=False)[0]
        fd[k] = (Lp - Lm) / (2 * eps)
    rel = np.linalg.norm(gth - fd) / max(1e-30, np.linalg.norm(fd))
    check("T10 CF-MMD exact dL/dtheta vs FD (k=16)", rel < 1e-5,
          f"rel err={rel:.2e}")

    worst = 0.0
    for (l, c) in [(0, 0), (3, 7), (6, 12)]:
        old = bank.A[l, c]
        bank.A[l, c] = old + eps
        Lp = cf_mmd_loss_and_grads(bank, S, Y, theta, Om, alpha,
                                   k_shots=kshots, gamma_risk=gamma,
                                   compute_grad=False)[0]
        bank.A[l, c] = old - eps
        Lm = cf_mmd_loss_and_grads(bank, S, Y, theta, Om, alpha,
                                   k_shots=kshots, gamma_risk=gamma,
                                   compute_grad=False)[0]
        bank.A[l, c] = old
        fdv = (Lp - Lm) / (2 * eps)
        worst = max(worst, abs(gA[l, c] - fdv) / max(1e-12, abs(fdv)))
    check("T11 CF-MMD exact encoding grad vs FD (k=16)", worst < 1e-5,
          f"worst rel err={worst:.2e}")

    # CF law consistency: exact CF-MMD vs sampled k-shot generation MMD
    from qfan.ridge import ridge_fit as _rf  # noqa: F401
    P = bank.setting_probs(S, theta)
    Sig = bank.parity_covariance_sum(P)
    from qfan.born import noise_aware_ridge_fit
    W, _ = noise_aware_ridge_fit(bank.expectation_features(P), Y, Sig,
                                 alpha, k=kshots)
    pool = np.vstack([bank.sample_record_features(P, kshots, rng) @ W
                      for _ in range(300)])
    psi_mod = np.exp(1j * (pool @ Om.T)).mean(axis=0)
    Wt = bank.atom_weights(P)
    Em = np.exp(1j * ((bank.f_atoms @ W) @ Om.T) / kshots)
    psi_cf = ((Wt @ Em) ** kshots).mean(axis=0)
    dev = np.abs(psi_cf - psi_mod).max()
    check("T12 CF identity phi(w/k)^k vs sampled k-shot law", dev < 0.02,
          f"max |dpsi|={dev:.3e} (sampling tol)")


# ---------------------------------------------------------------- T13
def t13():
    from qfan.born import cf_mmd_loss_and_grads, draw_joint_rff_frequencies
    rng = np.random.default_rng(9)
    bank = StatevectorBornBank(16, BornBankSpec(include_y=True), seed=3)
    n, b, d_c, kshots = 16, 3, 5, 32
    S = rng.normal(size=(n, 16))
    Y = np.abs(rng.normal(0.05, 0.04, size=(n, b)))
    Xc = np.abs(rng.normal(0.05, 0.04, size=(n, d_c)))
    theta = rng.normal(size=bank.n_var)
    sig = median_sigma(Y, Y)
    Oy, Oc = draw_joint_rff_frequencies([0.5 * sig, sig], b,
                                        [sig], d_c, 80, seed=2)
    alpha, gamma = 1e-2, 10.0
    kw = dict(k_shots=kshots, gamma_risk=gamma, X_cond=Xc, Omega_cond=Oc)

    L0, gth, gA, gb, _, _ = cf_mmd_loss_and_grads(
        bank, S, Y, theta, Oy, alpha, compute_grad=True,
        compute_encoding_grad=True, **kw)

    eps = 1e-6
    fd = np.zeros_like(gth)
    for k in range(bank.n_var):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps; tm[k] -= eps
        Lp = cf_mmd_loss_and_grads(bank, S, Y, tp, Oy, alpha,
                                   compute_grad=False, **kw)[0]
        Lm = cf_mmd_loss_and_grads(bank, S, Y, tm, Oy, alpha,
                                   compute_grad=False, **kw)[0]
        fd[k] = (Lp - Lm) / (2 * eps)
    rel = np.linalg.norm(gth - fd) / max(1e-30, np.linalg.norm(fd))
    ok1 = rel < 1e-5

    worst = 0.0
    for (l, c) in [(1, 2), (4, 9)]:
        old = bank.A[l, c]
        bank.A[l, c] = old + eps
        Lp = cf_mmd_loss_and_grads(bank, S, Y, theta, Oy, alpha,
                                   compute_grad=False, **kw)[0]
        bank.A[l, c] = old - eps
        Lm = cf_mmd_loss_and_grads(bank, S, Y, theta, Oy, alpha,
                                   compute_grad=False, **kw)[0]
        bank.A[l, c] = old
        fdv = (Lp - Lm) / (2 * eps)
        worst = max(worst, abs(gA[l, c] - fdv) / max(1e-12, abs(fdv)))
    check("T13 conditional (joint) CF-MMD grads vs FD", ok1 and worst < 1e-5,
          f"theta rel={rel:.2e}  encoding worst rel={worst:.2e}")


# ---------------------------------------------------------------- T14
def t14():
    from qfan.born import cf_mmd_loss_and_grads, draw_rff_frequencies
    rng = np.random.default_rng(12)
    bank = StatevectorBornBank(16, BornBankSpec(include_y=True), seed=3)
    n, kshots, rho = 16, 32, 0.7
    S = rng.normal(size=(n, 16))
    Y = np.abs(rng.normal(0.05, 0.04, size=(n, 2)))
    theta = rng.normal(size=bank.n_var)
    sig = median_sigma(Y, Y)
    Om = draw_rff_frequencies([0.5 * sig, sig, 2 * sig], 2, 90, seed=4)
    kw = dict(k_shots=kshots, gamma_risk=10.0, rho=rho)

    L0, gth, _, _, _, grho = cf_mmd_loss_and_grads(
        bank, S, Y, theta, Om, 1e-2, compute_grad=True, **kw)
    eps = 1e-6
    fd = np.zeros_like(gth)
    for k in range(bank.n_var):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps; tm[k] -= eps
        Lp = cf_mmd_loss_and_grads(bank, S, Y, tp, Om, 1e-2,
                                   compute_grad=False, **kw)[0]
        Lm = cf_mmd_loss_and_grads(bank, S, Y, tm, Om, 1e-2,
                                   compute_grad=False, **kw)[0]
        fd[k] = (Lp - Lm) / (2 * eps)
    rel = np.linalg.norm(gth - fd) / max(1e-30, np.linalg.norm(fd))
    kw_p = dict(kw); kw_p["rho"] = rho + eps
    kw_m = dict(kw); kw_m["rho"] = rho - eps
    Lp = cf_mmd_loss_and_grads(bank, S, Y, theta, Om, 1e-2,
                               compute_grad=False, **kw_p)[0]
    Lm = cf_mmd_loss_and_grads(bank, S, Y, theta, Om, 1e-2,
                               compute_grad=False, **kw_m)[0]
    fd_rho = (Lp - Lm) / (2 * eps)
    rel_r = abs(grho - fd_rho) / max(1e-12, abs(fd_rho))
    check("T14 split-record CF: theta & rho grads vs FD",
          rel < 1e-5 and rel_r < 1e-5,
          f"theta rel={rel:.2e}  rho rel={rel_r:.2e}  rho grad={grho:+.4f}")


# ---------------------------------------------------------------- T15
def t15():
    """Definitive qiskit-construction test: the record DISTRIBUTIONS of the
    QiskitCircuitFactory circuits (via noiseless Aer) must match the exact
    engine's Born probabilities in every measurement setting. This is the
    test that separates 'device noise' from 'construction bug' for the
    hardware deployment path."""
    try:
        from qfan.ibm_born import QiskitCircuitFactory
        from qiskit_aer import AerSimulator
    except Exception:
        print("  [SKIP] T15 qiskit-aer not installed (REQUIRED before any "
              "hardware run: validates the deployment circuits)")
        return
    rng = np.random.default_rng(21)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    theta = rng.normal(size=bank.n_var)
    S = rng.normal(size=(3, 32))
    P = bank.setting_probs(S, theta)
    fac = QiskitCircuitFactory(bank)
    backend = AerSimulator(seed_simulator=5)
    angles = bank.angles_from_sketch(S)
    shots = 40000
    worst_tv = 0.0
    for g, setting in enumerate(("Z", "X", "Y")):
        circs = [fac.build(angles[i], theta, setting) for i in range(3)]
        res = backend.run(circs, shots=shots).result()
        for i in range(3):
            counts = res.get_counts(i)
            emp = np.zeros(bank.dim)
            for bs, cnum in counts.items():
                emp[int(bs.replace(" ", ""), 2)] = cnum / shots
            tv = 0.5 * np.abs(emp - P[g, i]).sum()
            worst_tv = max(worst_tv, tv)
    # TV sampling floor at 40k shots over 8 outcomes ~ 0.5*sum|noise| ~ 0.006
    check("T15 qiskit circuits vs exact engine (all settings)",
          worst_tv < 0.02, f"worst TV distance={worst_tv:.4f} "
          f"(sampling floor ~0.006; >0.02 means CONSTRUCTION BUG)")


# ---------------------------------------------------------------- T16
def t16():
    """Regression test for the seeded-simulator bug: successive executor
    calls MUST produce independent records. A fixed seed_simulator makes
    every aer job reuse the same random stream, silently correlating the
    Born noise across autoregressive blocks (observed as a washed-out,
    over-positive correlation matrix on aer while local stays correct)."""
    rng = np.random.default_rng(31)
    bank = StatevectorBornBank(32, BornBankSpec(include_y=True), seed=3)
    theta = rng.normal(size=bank.n_var)
    S = rng.normal(size=(64, 32))

    F_exact = bank.expectation_features(bank.setting_probs(S, theta))

    def indep_check(executor, name):
        r1 = executor.run(S, theta, 16)
        r2 = executor.run(S, theta, 16)
        # identical streams would give identical records
        frac_equal = float((np.asarray(r1) == np.asarray(r2)).mean())
        # SHOT-NOISE correlation across the two calls: subtract the exact
        # conditional means so signal variation does not leak into the test
        f1 = np.concatenate([bank.T[np.asarray(r1[g])].mean(1)
                             for g in range(bank.G)], axis=1)
        f2 = np.concatenate([bank.T[np.asarray(r2[g])].mean(1)
                             for g in range(bank.G)], axis=1)
        n1 = f1 - F_exact
        n2 = f2 - F_exact
        cc = float(np.mean(np.abs(
            (n1 * n2).mean(0)
            / (n1.std(0) * n2.std(0) + 1e-12))))
        # independent noise: cc ~ 1/sqrt(64 samples) ~ 0.12 expected scatter
        check(f"T16 call-to-call record independence ({name})",
              frac_equal < 0.6 and cc < 0.3,
              f"frac identical={frac_equal:.3f}  |noise corr|={cc:.3f} "
              f"(independent ~0.1; seeded-stream bug ~1.0)")

    from qfan.ibm_born import LocalRecordExecutor
    indep_check(LocalRecordExecutor(bank, seed=0), "local")
    try:
        from qfan.ibm_born import AerRecordExecutor
        indep_check(AerRecordExecutor(bank, seed=0), "aer")
    except Exception:
        print("  [SKIP] T16-aer qiskit-aer not installed")


if __name__ == "__main__":
    print("QFAN test suite")
    print("-" * 78)
    t1_t2()
    t3()
    t4()
    t5_t6()
    t7()
    t8()
    t10_t11()
    t13()
    t14()
    t15()
    t16()
    print("-" * 78)
    print(f"{sum(PASS)}/{len(PASS)} checks passed"
          + ("" if all(PASS) else "  <<< FAILURES PRESENT"))
    sys.exit(0 if all(PASS) else 1)
