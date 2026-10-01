#!/usr/bin/env python3
"""
sweep_nq.py -- TEST D: does accuracy improve with qubit count at FIXED block
size?

This is the decisive experiment for the referee's question "at which scale is
an actual quantum device needed?". The capacity rule says qubits are needed to
make blocks bigger. This sweep asks the separate and more important question:
holding the block size fixed, does adding qubits make the model better?

  * If the error SATURATES by n_q = 4 or 5, then extra qubits buy nothing in
    this architecture, there is no route to a regime where a device is
    required, and the paper should say so plainly and rest on the
    architectural contribution.
  * If the error KEEPS IMPROVING, fit the trend, extrapolate, and report the
    register size at which it would cross the classical-simulation boundary.
    That is the referee's requested regime, answered with a measurement.

Everything else is held fixed: data, split, seeds, block size, sketch, shots
per block, epochs, learning rates, and the loss. The only variable is n_q
(and therefore p_f = 3 n_q (n_q+1)/2 and the parameter count 2 L n_q).

RESUMABLE: each n_q is checkpointed separately, so run it repeatedly with a
time budget until every point reports COMPLETE.

    python sweep_nq.py --nq 2 3 4 5 6 --epochs 150 --budget-s 280
    python sweep_nq.py --report            # print the table when done

Cost note: training time grows with n_q because the exact gradient loops over
2 L n_q parameters and the state dimension is 2^n_q. n_q = 6 is roughly an
order of magnitude slower per epoch than n_q = 3; budget accordingly. Use
--epochs 150 for a trend (the headline runs used 300).
"""
import argparse
import json
import os
import pathlib
import pickle
import sys
import time

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

from qfan import correlation_error_summary, corr_nan_safe        # noqa: E402
from qfan.born import (BornBankSpec, StatevectorBornBank,        # noqa: E402
                       cf_mmd_loss_and_grads, fit_all_blocks_born,
                       sample_progressive_born)
from qfan.train import AdamOpt, _scheduled_lr, _grad_clip        # noqa: E402
# ----------------------------------------------------------------------
# Locate the d=25 QFAN training driver. Its module name differs between
# checkouts, so we discover it rather than hardcoding: any module in
# scripts/ that exposes both build_problem() and setup_train() qualifies.
# Override explicitly with  --driver <module_name>  if several match.
# ----------------------------------------------------------------------
import importlib                                              # noqa: E402

_DRIVER_CANDIDATES = ["run_d25_v50_chunked", "run_d25_v50", "run_d25_chunked",
                      "run_d25_v5", "run_d25"]


def load_driver(name=None):
    tried = []
    names = [name] if name else _DRIVER_CANDIDATES
    for n in names:
        try:
            m = importlib.import_module(n)
            if hasattr(m, "build_problem") and hasattr(m, "setup_train"):
                return m
            tried.append(f"{n} (no build_problem/setup_train)")
        except ImportError:
            tried.append(f"{n} (not found)")
    if name:
        raise SystemExit(f"--driver {name}: " + tried[0])
    # Fall back to scanning scripts/ -- by READING the source, never by
    # importing, because importing an arbitrary script executes its
    # module-level code (plots, argparse, training runs).
    cands = []
    for py in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        mod = py.stem
        if mod.startswith("_") or mod in ("sweep_nq",
                                          "quantum_necessity_tests"):
            continue
        try:
            txt = py.read_text(errors="ignore")
        except Exception:
            continue
        if "def build_problem" in txt and "def setup_train" in txt:
            cands.append(mod)
    # prefer the package's own trainer, then import ONLY the chosen one
    cands.sort(key=lambda n: (0 if n == "train" else 1, len(n)))
    found = []
    for mod in cands:
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            print(f"[driver] {mod} failed to import: "
                  f"{type(e).__name__}: {e}")
            continue
        if hasattr(m, "build_problem") and hasattr(m, "setup_train"):
            found.append((mod, m))
            break
    if len(found) == 1:
        print(f"[driver] auto-detected: {found[0][0]}")
        return found[0][1]
    if len(found) > 1:
        names_ = ", ".join(n for n, _ in found)
        # prefer the package's own trainer
        for n, m in found:
            if n == "train":
                print(f"[driver] auto-detected: {n}  (candidates: {names_})")
                return m
        print(f"[driver] several candidates: {names_}")
        print("[driver] using the first; override with --driver <name>")
        return found[0][1]
    raise SystemExit(
        "could not find the training driver. Looked for "
        + ", ".join(_DRIVER_CANDIDATES)
        + " and scanned scripts/ for a module exposing build_problem() and "
          "setup_train(). Pass it explicitly:  --driver <module_name>")


_DRV = load_driver(os.environ.get("QFAN_DRIVER"))
build_problem = _DRV.build_problem
setup_train = _DRV.setup_train
K_SHOTS = getattr(_DRV, "K_SHOTS", 64)
GAMMA_RISK = getattr(_DRV, "GAMMA_RISK", 10.0)
GRAD_CLIP = getattr(_DRV, "GRAD_CLIP", 5.0)
LR = getattr(_DRV, "LR", 0.1)
LR_ENC = getattr(_DRV, "LR_ENC", 0.1)
LR_RHO = getattr(_DRV, "LR_RHO", 0.05)
LR_WARMUP = getattr(_DRV, "LR_WARMUP", 5)
LR_FLOOR_FRAC = getattr(_DRV, "LR_FLOOR_FRAC", 0.1)
BATCH = getattr(_DRV, "BATCH", 256)

OUT = PROJECT_ROOT / "outputs"
MON_N = 600            # samples for the train-side free-running monitor




# ----------------------------------------------------------------------
# PREFLIGHT: the engine currently REFUSES n_q != 3.
#   src/qfan/born.py, StatevectorBornBank.__init__:
#       if spec.n_qubits != 3: raise NotImplementedError(...)
# The construction is written generically (parity table, CZ ring and index
# masks all loop over n_q), but simply deleting the guard is NOT sufficient:
# a naive lift produces wrong states, because the atom enumeration, f_atoms
# and the cached outer-product tables are built once for dim=8 and must be
# rebuilt for the new dimension. Generalizing is real work, and the code
# comment ("mechanical but must be re-verified") is right.
#
# This preflight is the re-verification. It rebuilds the circuit from an
# INDEPENDENT dense-unitary construction and compares state vectors. It is
# validated at n_q=3, where it agrees with the engine to 2e-16. Do not run
# the sweep until it passes at every n_q you intend to use.
# ----------------------------------------------------------------------
def _op(single, q, nq):
    """Single-qubit op embedded with qubit 0 = LEAST significant bit."""
    M = np.array([[1]], complex)
    for k in range(nq - 1, -1, -1):
        M = np.kron(M, single if k == q else np.eye(2))
    return M


def _RY(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                     [np.sin(t / 2), np.cos(t / 2)]], complex)


def _RZ(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], complex)


def _CZ(i, j, nq):
    dim = 1 << nq
    U = np.eye(dim, dtype=complex)
    for n in range(dim):
        if ((n >> i) & 1) and ((n >> j) & 1):
            U[n, n] = -1
    return U


def verify_engine(nq, depth=3, angle_dim=16, m=32, n_test=4, tol=1e-12):
    """Cross-check the engine's state() against an independent dense-unitary
    build. Returns max |dpsi|. Must be < tol before the sweep is meaningful."""
    from qfan.born import BornBankSpec, StatevectorBornBank
    spec = BornBankSpec(n_qubits=nq, depth=depth, angle_dim=angle_dim,
                        include_y=True, gen_shots_k=K_SHOTS)
    bank = StatevectorBornBank(m, spec, seed=3)          # raises if guarded
    rng = np.random.default_rng(0)
    th = 0.4 * rng.normal(size=bank.n_var)
    bank.set_theta(th)
    S = rng.normal(size=(n_test, m))
    a = bank.angles_from_sketch(S)
    psi_e = bank.state(S, th)
    pairs = [(q, q + 1) for q in range(nq - 1)] + ([(nq - 1, 0)] if nq > 2 else [])
    dev = 0.0
    for s in range(n_test):
        psi = np.zeros(1 << nq, complex)
        psi[0] = 1.0
        t = 0
        for _layer in range(depth):
            for k in range(angle_dim):
                q = k % nq
                phi = np.pi * a[s, k]
                psi = _op(_RY(phi) if k % 2 == 0 else _RZ(phi), q, nq) @ psi
            for q in range(nq):
                psi = _op(_RZ(th[t]), q, nq) @ psi; t += 1
                psi = _op(_RY(th[t]), q, nq) @ psi; t += 1
            for (i, j) in pairs:
                psi = _CZ(i, j, nq) @ psi
        dev = max(dev, float(np.abs(psi - psi_e[s]).max()))
    return dev


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def make_bank(cfg, nq, depth, angle_dim, seed):
    """Same construction as build_problem() but with n_q overridden."""
    spec = BornBankSpec(n_qubits=nq, depth=depth, angle_dim=angle_dim,
                        include_y=True, gen_shots_k=K_SHOTS)
    return spec, StatevectorBornBank(cfg.sketch.sketch_dim, spec,
                                     seed=seed + 13)


def monitor(bank, cache, Y_tr, blocks, sketcher, alpha, rho, d, seed):
    """Train-side free-running off-diagonal correlation error."""
    models = fit_all_blocks_born(bank, cache, Y_tr, blocks, alpha,
                                 noise_aware=True, k_shots=K_SHOTS)
    rng = np.random.default_rng(seed)
    Y = sample_progressive_born(bank, models, d, blocks, sketcher, MON_N,
                                rng, k=K_SHOTS, record_share=rho)
    cs = correlation_error_summary(corr_nan_safe(Y_tr), corr_nan_safe(Y),
                                   blocks)
    return cs["corr_mae_offdiag"], cs["corr_mae_within"]


def train_one(nq, epochs, budget_s, depth, angle_dim, verbose=True):
    t0 = time.time()
    (cfg, Y_tr, Y_te, d, blocks, sketcher, cache, _spec0,
     _bank0) = build_problem()
    seed = cfg.data.seed
    alpha = cfg.bank.ridge_alpha
    spec, bank = make_bank(cfg, nq, depth, angle_dim, seed)
    nb = len(blocks)
    ck = OUT / f"sweep_nq{nq}_ckpt.pkl"

    (prng, val_idx, train_pool, sigmas_pb, Omega_pb,
     Omega_c_pb) = setup_train(bank, cache, Y_tr, blocks, seed)
    p_theta = bank.n_var
    n_enc = bank.L * bank.m + bank.L

    if ck.exists():
        st = pickle.load(open(ck, "rb"))
        theta = st["theta"]; bank.set_theta(theta)
        bank.A = st["A"]; bank.b = st["b"]; r_logit = st["r_logit"]
        opt = AdamOpt(p_theta, lr=LR); opt.m, opt.v, opt.t = st["opt"]
        opt_e = AdamOpt(n_enc, lr=LR_ENC); opt_e.m, opt_e.v, opt_e.t = st["oe"]
        opt_r = AdamOpt(nb, lr=LR_RHO); opt_r.m, opt_r.v, opt_r.t = st["or"]
        prng.bit_generator.state = st["prng"]
        t = st["epoch"]; hist = st["hist"]
    else:
        theta = 0.3 * prng.normal(size=p_theta); bank.set_theta(theta)
        r_logit = logit(np.full(nb, 0.5))
        opt = AdamOpt(p_theta, lr=LR); opt_e = AdamOpt(n_enc, lr=LR_ENC)
        opt_r = AdamOpt(nb, lr=LR_RHO)
        t = 0; hist = dict(mon_off=[], mon_ep=[], loss=[])
        if verbose:
            print(f"[n_q={nq}] p_f={bank.pf}  theta={p_theta}  "
                  f"G={bank.G}  blocks={nb}", flush=True)

    while t < epochs and (time.time() - t0) < budget_s:
        idx = prng.choice(train_pool, size=min(BATCH, train_pool.size),
                          replace=False)
        rho_now = sigmoid(r_logit)
        g = np.zeros(p_theta); gA = np.zeros_like(bank.A)
        gb = np.zeros_like(bank.b); gr = np.zeros(nb); tot = 0.0
        for bi, (s, bsz) in enumerate(blocks):
            S = cache[bi, idx].astype(np.float64, copy=False)
            Yb = Y_tr[idx, s:s + bsz].astype(np.float64, copy=False)
            Xc = Y_tr[idx, :s] if Omega_c_pb[bi] is not None else None
            l, gt, ga, gbb, _, grho = cf_mmd_loss_and_grads(
                bank, S, Yb, theta, Omega_pb[bi], alpha, k_shots=K_SHOTS,
                gamma_risk=GAMMA_RISK, compute_grad=True,
                compute_encoding_grad=True, X_cond=Xc,
                Omega_cond=Omega_c_pb[bi], rho=float(rho_now[bi]))
            g += gt; gA += ga; gb += gbb; tot += l
            if grho is not None:
                gr[bi] += grho
        sch = _scheduled_lr(t, epochs, base_lr=1.0, warmup=LR_WARMUP,
                            decay="cosine", floor_frac=LR_FLOOR_FRAC)
        opt.lr = LR * sch
        theta = theta - opt.step(_grad_clip(g, GRAD_CLIP)); bank.set_theta(theta)
        opt_e.lr = LR_ENC * sch
        se = opt_e.step(_grad_clip(np.concatenate([gA.ravel(), gb]), GRAD_CLIP))
        bank.A -= se[:bank.L * bank.m].reshape(bank.L, bank.m)
        bank.b -= se[bank.L * bank.m:]
        opt_r.lr = LR_RHO * sch
        r_logit = r_logit - opt_r.step(gr * rho_now * (1 - rho_now))
        hist["loss"].append(tot); t += 1
        if t % 25 == 0 or t == epochs:
            off, win = monitor(bank, cache, Y_tr, blocks, sketcher, alpha,
                               sigmoid(r_logit), d, seed + 4242 + t)
            hist["mon_off"].append(off); hist["mon_ep"].append(t)
            if verbose:
                print(f"  [n_q={nq} ep {t:4d}/{epochs}] loss={tot:.4f} "
                      f"monitor off={off:.4f} within={win:.4f}   "
                      f"(PROVISIONAL: train-side monitor, n={MON_N}, "
                      f"not a result)", flush=True)
        pickle.dump(dict(theta=theta, A=bank.A, b=bank.b, r_logit=r_logit,
                         opt=(opt.m, opt.v, opt.t), oe=(opt_e.m, opt_e.v, opt_e.t),
                         orr=(opt_r.m, opt_r.v, opt_r.t),
                         **{"or": (opt_r.m, opt_r.v, opt_r.t)},
                         prng=prng.bit_generator.state, epoch=t, hist=hist),
                    open(ck, "wb"))

    if t < epochs:
        print(f"[n_q={nq}] CHUNK DONE at epoch {t}/{epochs}; re-run.")
        return None

    # final: TEST evaluation, seed-replicated
    rho = sigmoid(r_logit)
    models = fit_all_blocks_born(bank, cache, Y_tr, blocks, alpha,
                                 noise_aware=True, k_shots=K_SHOTS)
    Ct = corr_nan_safe(Y_te); offs = []
    for s_ in range(3):
        rng = np.random.default_rng(7000 + s_)
        Y = sample_progressive_born(bank, models, d, blocks, sketcher,
                                    len(Y_te), rng, k=K_SHOTS,
                                    record_share=rho)
        offs.append(correlation_error_summary(
            Ct, corr_nan_safe(Y), blocks)["corr_mae_offdiag"])
    mon = hist.get("mon_off", [])
    res = dict(nq=nq, p_f=int(bank.pf), n_theta=int(p_theta),
               off_mean=float(np.mean(offs)), off_std=float(np.std(offs)),
               epochs=epochs, epochs_run=int(t), statevector_dim=2 ** nq,
               monitor_last=float(mon[-1]) if mon else None,
               monitor_best=float(min(mon)) if mon else None,
               monitor_tail_slope=(float(np.polyfit(
                   np.arange(len(mon[-4:])), mon[-4:], 1)[0])
                   if len(mon) >= 4 else None))
    json.dump(res, open(OUT / f"sweep_nq{nq}_result.json", "w"), indent=1)
    print(f"[n_q={nq}] COMPLETE  off = {res['off_mean']:.4f} "
          f"+/- {res['off_std']:.4f}  (p_f={res['p_f']})")
    return res


def report(requested=None):
    """Print the sweep table. Refuses to draw a conclusion unless the sweep is
    complete, the points share an epoch budget, and each point looks converged.
    A partially finished sweep produced a confident-looking but meaningless
    verdict in an earlier version; these guards exist for that reason."""
    rows = []
    for f in sorted(OUT.glob("sweep_nq*_result.json")):
        rows.append(json.load(open(f)))
    if not rows:
        print("\nno COMPLETED sweep points yet.")
        if requested:
            partial = []
            for nq in requested:
                ck = OUT / f"sweep_nq{nq}_ckpt.pkl"
                if ck.exists():
                    try:
                        st = pickle.load(open(ck, "rb"))
                        partial.append(f"n_q={nq} at epoch {st['epoch']}")
                    except Exception:
                        pass
            if partial:
                print("in progress: " + "; ".join(partial))
                print("These are CHECKPOINTS, not results. Re-run with --loop "
                      "until every point reports COMPLETE.")
        return
    rows.sort(key=lambda r: r["nq"])
    done = {r["nq"] for r in rows}
    missing = sorted(set(requested) - done) if requested else []

    print(f"\n{'n_q':>4} {'p_f':>5} {'theta':>6} {'epochs':>7} "
          f"{'off-diag (test)':>20} {'2^n_q':>8}")
    for r in rows:
        print(f"{r['nq']:>4} {r['p_f']:>5} {r['n_theta']:>6} "
              f"{r.get('epochs','?'):>7} "
              f"{r['off_mean']:>11.4f} +/- {r['off_std']:.4f} "
              f"{r['statevector_dim']:>8}")

    # --- guards -------------------------------------------------------
    problems = []
    if missing:
        problems.append(f"INCOMPLETE: n_q {missing} have no result yet "
                        f"(checkpoints only).")
    eps = {r.get("epochs") for r in rows}
    if len(eps) > 1:
        problems.append(f"EPOCH MISMATCH: points were trained for different "
                        f"budgets {sorted(eps)}; the comparison confounds "
                        f"capacity with training budget.")
    for r in rows:
        er, ep = r.get("epochs_run"), r.get("epochs")
        if er is not None and ep is not None and er < ep:
            problems.append(f"n_q={r['nq']} stopped at epoch {er}/{ep}.")
        sl = r.get("monitor_tail_slope")
        if sl is not None and sl < -0.002:
            problems.append(f"n_q={r['nq']} monitor still improving "
                            f"(tail slope {sl:+.4f}/monitor step): "
                            f"undertrained, its value is an upper bound.")
    if problems:
        print("\n" + "!" * 68)
        for p_ in problems:
            print("  " + p_)
        print("  NO CONCLUSION IS DRAWN from an incomplete or inconsistent "
              "sweep.")
        print("!" * 68)
        return

    if len(rows) < 3:
        print("\nfewer than 3 complete points: no trend reading.")
        return
    y = np.array([r["off_mean"] for r in rows])
    sd = max(r["off_std"] for r in rows)
    d1 = np.diff(y)
    print(f"\n  successive changes: {np.round(d1, 4)}   "
          f"(seed noise ~ {sd:.4f})")
    if np.all(np.abs(d1[-2:]) < 2 * sd):
        print("  READING: the last two steps are within twice the seed noise "
              "-> SATURATED.\n  Extra qubits buy nothing at fixed block size; "
              "the architecture has no route\n  to a register size where "
              "classical simulation fails. Report this plainly.")
    else:
        print("  READING: still improving. Fit the trend and report the n_q at "
              "which it would\n  cross the classical-simulation boundary "
              "(state vector fails near n_q ~ 45).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--budget-s", type=float, default=280.0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--angle-dim", type=int, default=16)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--loop", action="store_true",
                    help="keep re-running chunks until every n_q is COMPLETE")
    ap.add_argument("--max-hours", type=float, default=24.0,
                    help="wall-clock ceiling for --loop")
    ap.add_argument("--driver", default=None,
                    help="training-driver module name in scripts/ "
                         "(auto-detected if omitted)")
    ap.add_argument("--verify", action="store_true",
                    help="preflight: cross-check the engine at each n_q")
    args = ap.parse_args()
    if args.report:
        report(args.nq); return
    if args.verify:
        print("PREFLIGHT: engine vs independent dense unitary")
        blocked = []
        for nq in args.nq:
            try:
                dev = verify_engine(nq, args.depth, args.angle_dim)
                ok = "PASS" if dev < 1e-12 else "FAIL"
                print(f"  n_q={nq}: max |dpsi| = {dev:.3e}   [{ok}]")
            except NotImplementedError as e:
                print(f"  n_q={nq}: BLOCKED -- {e}")
                blocked.append(nq)
            except Exception as e:
                print(f"  n_q={nq}: ERROR -- {type(e).__name__}: {e}")
        if blocked:
            print("\n" + "=" * 70)
            print("This copy of src/qfan/born.py still carries the n_q == 3")
            print("guard of an older release, so the sweep cannot run. Use the")
            print("born.py shipped with this package, which supports any n_q,")
            print("then re-verify:")
            print("")
            print("    python tests/test_born.py         # expect 15/15")
            print("    python scripts/sweep_nq.py --verify --nq "
                  + " ".join(str(n) for n in sorted(set(blocked))))
            print("=" * 70)
        else:
            print("\nAll n_q PASS. The sweep is ready to run:")
            print("    python scripts/sweep_nq.py --nq "
                  + " ".join(str(n) for n in args.nq)
                  + " --epochs 150 --budget-s 280   # repeat until COMPLETE")
        return
    t_start = time.time()
    if args.loop:
        print(f"[loop] running until all of {args.nq} are COMPLETE "
              f"(ceiling {args.max_hours} h). Safe to interrupt: every chunk "
              f"is checkpointed.")
    while True:
        for nq in args.nq:
            if (OUT / f"sweep_nq{nq}_result.json").exists():
                continue
            train_one(nq, args.epochs, args.budget_s, args.depth,
                      args.angle_dim)
        done = [nq for nq in args.nq
                if (OUT / f"sweep_nq{nq}_result.json").exists()]
        if not args.loop or len(done) == len(args.nq):
            break
        el = (time.time() - t_start) / 3600.0
        if el > args.max_hours:
            print(f"[loop] stopped at the {args.max_hours} h ceiling; "
                  f"{len(done)}/{len(args.nq)} complete. Re-run to continue.")
            break
        print(f"[loop] {len(done)}/{len(args.nq)} complete, {el:.2f} h "
              f"elapsed; continuing.")
    report()


if __name__ == "__main__":
    main()
