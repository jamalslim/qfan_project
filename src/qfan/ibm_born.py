"""
IBM hardware deployment of the QFAN sampler.

Deploys the TRAINED model (theta, encoding A/b, per-block decoders W) for
GENERATION on IBM Quantum hardware: per block and per sample, the shared
3-qubit circuit is executed k times in each of the G measurement settings,
the per-shot parities are averaged, and the block is read out through the
shot-noise-aware decoder. Training stays on the exact simulator; hardware
runs the physical sampler.

Design
------
  QiskitCircuitFactory   builds the (identical) circuit in qiskit, with the
                         basis rotations for the X / Y settings. Bit
                         convention matches the numpy engine: outcome
                         integer = int(bitstring, 2), bit q = qubit q.
  RecordExecutor         abstraction producing k-shot parity averages
                         f_hat (n, pf) for a batch of sketches:
     * LocalRecordExecutor  exact Born sampling on the numpy statevector
                            engine (dry-run / CI; no qiskit needed)
     * AerRecordExecutor    qiskit-aer shot sampling of the SAME circuits
                            (validates the qiskit circuit construction
                            against the numpy engine)
     * IBMRecordExecutor    qiskit-ibm-runtime SamplerV2 on real hardware
  generate_progressive_records
                         free-running autoregressive generation through an
                         arbitrary executor (the only sampler difference
                         between simulator and hardware paths).
  load_trained_model     deterministic reconstruction of the trained model
                         from the saved run (theta/A/b from the results
                         file; decoders refit in closed form from the
                         training cache -- no retraining).

Shot accounting (generation): G*k = 3*64 = 192 shots per block per sample;
B = 6 blocks at d = 12  =>  1152 shots per generated shower, executed as
G sequential-setting circuit batches per block (autoregressive dependency
forces B sequential rounds; within a round all samples/settings batch).

Typical hardware run (adjust backend/session to your account):

    python scripts/run_d12_born_ibm.py --executor ibm --backend ibm_fez \
        --n-samples 200
"""

from dataclasses import dataclass
from math import pi
from typing import List, Optional

import numpy as np

from .born import (BornBankSpec, StatevectorBornBank, BornBlockModel,
                   noise_aware_ridge_fit)
from .sketch import OnlineCountSketch, build_blocks, build_prefix_sketch_cache


# =====================================================================
# qiskit circuit construction (lazy import; matches the numpy engine)
# =====================================================================

class QiskitCircuitFactory:
    """Builds the QFAN circuit in qiskit for a given sketch-angle row and
    measurement setting. Convention check: qiskit bitstrings are little-
    endian, so int(bitstring, 2) equals the numpy engine's outcome index
    (bit q = qubit q). Validated by AerRecordExecutor against the exact
    engine (tests/test_born.py T7 checks the same convention for
    expectations)."""

    def __init__(self, bank: StatevectorBornBank):
        from qiskit import QuantumCircuit  # noqa: F401  (import check)
        self.bank = bank

    def build(self, angles_row: np.ndarray, theta: np.ndarray,
              setting: str):
        from qiskit import QuantumCircuit
        b = self.bank
        qc = QuantumCircuit(b.nq, b.nq)
        t = 0
        for _layer in range(b.depth):
            for kk in range(b.L):
                q = kk % b.nq
                if (kk % 2) == 0:
                    qc.ry(pi * float(angles_row[kk]), q)
                else:
                    qc.rz(pi * float(angles_row[kk]), q)
            for q in range(b.nq):
                qc.rz(float(theta[t]), q); t += 1
                qc.ry(float(theta[t]), q); t += 1
            if b.nq >= 2:
                for q in range(b.nq - 1):
                    qc.cz(q, q + 1)
                if b.nq > 2:
                    qc.cz(b.nq - 1, 0)
        if setting == "X":
            for q in range(b.nq):
                qc.h(q)
        elif setting == "Y":
            for q in range(b.nq):
                qc.sdg(q); qc.h(q)
        elif setting != "Z":
            raise ValueError(setting)
        qc.measure(range(b.nq), range(b.nq))
        return qc


_SETTINGS = ("Z", "X", "Y")


def _records_to_features(bank: StatevectorBornBank,
                         records) -> np.ndarray:
    """records: (G, n, k) outcome integers -> k-shot parity averages
    (n, pf), feature layout identical to bank.expectation_features."""
    feats = []
    for g in range(bank.G):
        feats.append(bank.T[np.asarray(records[g], np.int64)].mean(axis=1))
    return np.concatenate(feats, axis=1)


# =====================================================================
# executors
# =====================================================================

class LocalRecordExecutor:
    """Exact Born sampling on the numpy engine (dry-run path)."""

    def __init__(self, bank: StatevectorBornBank, seed: int = 0):
        self.bank = bank
        self.rng = np.random.default_rng(seed)

    def run(self, S: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        b = self.bank
        P = b.setting_probs(S, theta)
        n = P.shape[1]
        rec = np.zeros((b.G, n, k), dtype=np.int64)
        for g in range(b.G):
            cdf = np.cumsum(P[g], axis=1)
            cdf[:, -1] = 1.0
            u = self.rng.random((n, k))
            for i in range(n):
                rec[g, i] = np.searchsorted(cdf[i], u[i], side="right")
        return np.clip(rec, 0, b.dim - 1)


class NoisyLocalExecutor(LocalRecordExecutor):
    """Local executor with a hardware-like noise model: global depolarizing
    channel (probability p_depol: record replaced by a uniform outcome) plus
    independent per-qubit readout bit flips (probability eps_ro). Damps
    weight-1 parities by (1-p)(1-2*eps) and weight-2 by (1-p)(1-2*eps)^2 --
    the channel family observed on ibm_fez (measured lambda ~ 0.74). Used to
    validate the calibration/mitigation pipeline before spending QPU time."""

    def __init__(self, bank, seed=0, p_depol=0.15, eps_ro=0.03):
        super().__init__(bank, seed=seed)
        self.p_depol = float(p_depol)
        self.eps_ro = float(eps_ro)

    def run(self, S, theta, k):
        rec = super().run(S, theta, k)
        b = self.bank
        u = self.rng.random(rec.shape)
        rec = np.where(u < self.p_depol,
                       self.rng.integers(0, b.dim, rec.shape), rec)
        for q in range(b.nq):
            flip = (self.rng.random(rec.shape) < self.eps_ro)
            rec = np.where(flip, rec ^ (1 << q), rec)
        return rec


def calibrate_feature_response(bank: StatevectorBornBank,
                               executor,
                               S_cal: np.ndarray,
                               theta: Optional[np.ndarray] = None,
                               k_cal: int = 256,
                               lam_bounds=(0.2, 1.25)):
    """
    Measure the hardware's per-feature affine response
        E[f_hat_l | S] ~ lam_l * F_l(S) + c_l
    on training-side calibration sketches, by comparing executor-measured
    mean parities against the exact simulator features. Per feature, a
    least-squares fit across calibration inputs gives (lam_l, c_l); features
    whose ideal variation is too small for a stable slope fall back to a
    ratio-of-means estimate. Cost: |S_cal| x G circuits x k_cal shots
    (~37k shots at the defaults -- negligible next to generation).

    Returns (lam (pf,), c (pf,)). Generation then applies the inverse
    affine map, f_corr = (f_hat - c) / lam, which unbiases the read-out;
    this is measurement-error mitigation expressed at exactly the level
    where the model is linear.
    """
    th = bank.get_theta() if theta is None else np.asarray(theta, np.float64)
    F_ideal = bank.expectation_features(bank.setting_probs(S_cal, th))
    rec = executor.run(S_cal, th, k_cal)
    F_meas = _records_to_features(bank, rec)
    pf = F_ideal.shape[1]
    lam = np.ones(pf)
    c = np.zeros(pf)
    for l in range(pf):
        x, y = F_ideal[:, l], F_meas[:, l]
        vx = float(np.var(x))
        if vx > 1e-4:
            sl = float(np.cov(x, y, bias=True)[0, 1] / vx)
            lam[l] = sl
            c[l] = float(y.mean() - sl * x.mean())
        elif abs(x.mean()) > 5e-2:
            lam[l] = float(y.mean() / x.mean())
            c[l] = 0.0
        # else: leave identity (feature carries no calibratable signal)
    lam = np.clip(lam, lam_bounds[0], lam_bounds[1])

    # hardware single-record feature variance IN CORRECTED UNITS
    # (for the hardware-aware decoder refit): Var[(f-c)/lam] per feature,
    # estimated from the calibration records' shot-to-shot spread
    per_shot = []
    for g in range(bank.G):
        # per-record parities: (n_cal, k_cal, 6)
        pr = bank.T[np.asarray(rec[g], np.int64)]
        per_shot.append(pr)
    per_shot = np.concatenate(per_shot, axis=2)          # (n_cal, k_cal, pf)
    var_raw = per_shot.var(axis=1).mean(axis=0)          # (pf,)
    var_corr = var_raw / lam**2
    return lam, c, var_corr


def refit_decoders_hardware(tm, lam, c, var_corr, ridge_alpha=1e-2,
                            k_shots=None):
    """
    Hardware-aware decoder refit (closed form, no retraining): replace the
    ideal single-record covariance in the Gauss--Markov decoder by the
    MEASURED corrected-feature variance,
        W_hw = (F^T F + n * diag(var_corr)/k + alpha I)^{-1} F^T Y ,
    fit on the same training cache used for the ideal decoders. The
    diagonal approximation of the hardware covariance is deliberate: the
    off-diagonal (state-dependent) part is what the calibration cannot
    cheaply resolve, and overestimating noise merely shrinks -- it never
    destabilizes. Returns a new model list; tm is not modified."""
    from .born import BornBlockModel
    import copy
    k = k_shots or tm.k_gen
    # rebuild the training cache deterministically (as in load_trained_model)
    new_models = []
    F_by_block, Y_by_block = tm._fit_context  # stashed by load_trained_model
    for (start, bsz), (F, Yb) in zip(tm.blocks, zip(F_by_block, Y_by_block)):
        n = F.shape[0]
        M = F.T @ F + n * np.diag(var_corr) / float(k) \
            + float(ridge_alpha) * np.eye(F.shape[1])
        W = np.linalg.solve(M, F.T @ Yb)
        new_models.append(BornBlockModel(W=W, start=start, bsz=bsz))
    return new_models


class AerRecordExecutor:
    """qiskit-aer shot sampling of the qiskit-built circuits. Use this
    once before any hardware run: it validates the circuit construction
    and bit conventions against the exact engine."""

    def __init__(self, bank: StatevectorBornBank, seed: int = 0):
        from qiskit_aer import AerSimulator
        self.bank = bank
        self.factory = QiskitCircuitFactory(bank)
        self.backend = AerSimulator()
        self._seed0 = int(seed)
        self._ncall = 0     # CRITICAL: a FIXED seed_simulator would reuse
                            # the identical random stream in every job,
                            # correlating the Born noise across autoregressive
                            # blocks and settings (observed as a washed-out,
                            # over-positive correlation matrix). The seed
                            # must advance per job.

    def run(self, S: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        b = self.bank
        angles = b.angles_from_sketch(S)
        n = angles.shape[0]
        rec = np.zeros((b.G, n, k), dtype=np.int64)
        for g, setting in enumerate(_SETTINGS[: b.G]):
            circs = [self.factory.build(angles[i], theta, setting)
                     for i in range(n)]
            self._ncall += 1
            res = self.backend.run(
                circs, shots=k, memory=True,
                seed_simulator=self._seed0 + 7919 * self._ncall).result()
            for i in range(n):
                mem = res.get_memory(i)
                rec[g, i] = [int(bs.replace(" ", ""), 2) for bs in mem]
        return rec


class AerNoiseModelExecutor:
    """qiskit-aer with the noise model built from the REAL backend's current
    calibration data (AerSimulator.from_backend). Free discrimination test:
    if the calibration-predicted response lambda matches the observed
    hardware lambda (~0.74), device noise fully explains the run; if it
    predicts ~0.9, the hardware run exceeded its own calibration -- bad
    layout, drift, or an unmodeled effect. Requires qiskit-ibm-runtime for
    fetching the backend, but costs no QPU time."""

    def __init__(self, bank: StatevectorBornBank, backend_name: str,
                 seed: int = 0, optimization_level: int = 1):
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_aer import AerSimulator
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager)
        self.bank = bank
        self.factory = QiskitCircuitFactory(bank)
        real = QiskitRuntimeService().backend(backend_name)
        self.backend = AerSimulator.from_backend(real)
        self._seed0 = int(seed)
        self._ncall = 0     # advancing seed: see AerRecordExecutor note
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=real)

    def run(self, S: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        b = self.bank
        angles = b.angles_from_sketch(S)
        n = angles.shape[0]
        rec = np.zeros((b.G, n, k), dtype=np.int64)
        for g, setting in enumerate(_SETTINGS[: b.G]):
            circs = [self.factory.build(angles[i], theta, setting)
                     for i in range(n)]
            isa = self.pm.run(circs)
            self._ncall += 1
            res = self.backend.run(
                isa, shots=k, memory=True,
                seed_simulator=self._seed0 + 7919 * self._ncall).result()
            for i in range(n):
                mem = res.get_memory(i)
                rec[g, i] = [int(bs.replace(" ", ""), 2) for bs in mem]
        return rec


def probe_channel(bank: StatevectorBornBank, executor,
                  S_cal: np.ndarray,
                  theta: Optional[np.ndarray] = None,
                  k_cal: int = 256):
    """
    Channel probe (~|S_cal| x G x k_cal shots, ~37k at defaults; free on
    aer-noise): measures the per-feature response lambda over the
    calibration sketches and DECOMPOSES the channel analytically. With
    lambda_1 = (1-p)(1-2*eps) for weight-1 parities and
    lambda_2 = (1-p)(1-2*eps)^2 for weight-2, the two medians identify
        eps_readout = (1 - lambda_2/lambda_1)/2 ,
        p_depol     = 1 - lambda_1^2/lambda_2 .
    A near-zero eps with large p indicates depolarizing-like decoherence;
    large eps indicates readout flips; a strongly setting-dependent
    lambda (Z healthy, X/Y damped) would instead point at a basis-rotation
    or coherent-error problem -- i.e., the code-vs-noise discriminator.
    """
    lam, c, var_corr = calibrate_feature_response(
        bank, executor, S_cal, theta=theta, k_cal=k_cal)
    names = bank.feature_names
    print(f"{'feature':>10s} {'lambda':>8s} {'offset':>9s}")
    for l in range(bank.pf):
        print(f"{names[l]:>10s} {lam[l]:8.3f} {c[l]:9.4f}")
    w1_mask = np.zeros(bank.pf, bool)
    for g in range(bank.G):
        w1_mask[g * 6: g * 6 + 3] = True
    l1 = float(np.median(lam[w1_mask]))
    l2 = float(np.median(lam[~w1_mask]))
    print(f"\nmedian lambda: weight-1 = {l1:.3f}   weight-2 = {l2:.3f}")
    per_setting = [float(np.median(lam[g * 6:(g + 1) * 6]))
                   for g in range(bank.G)]
    print("median lambda per setting (Z, X, Y):",
          " ".join(f"{v:.3f}" for v in per_setting))
    if max(per_setting) - min(per_setting) > 0.15:
        print("=> strongly SETTING-DEPENDENT response: suspect basis "
              "rotations / coherent errors, NOT plain noise. Investigate "
              "before mitigating.")
    else:
        eps = max(0.0, 0.5 * (1.0 - l2 / max(l1, 1e-6)))
        pdep = max(0.0, 1.0 - l1 ** 2 / max(l2, 1e-6))
        print(f"=> channel decomposition: readout flip eps ~ {eps:.3f} "
              f"per qubit, depolarizing fraction p ~ {pdep:.3f}")
        print(f"   (healthy Heron triple expectation: eps ~ 0.01-0.02, "
              f"p ~ 0.03-0.08)")
    return lam, c


class IBMRecordExecutor:
    """qiskit-ibm-runtime SamplerV2 execution on real hardware.

    Notes for real runs:
      * requires `pip install qiskit-ibm-runtime` and a saved account
        (QiskitRuntimeService.save_account) or IBM_QUANTUM_TOKEN env;
      * circuits are transpiled once per block round with a preset pass
        manager (optimization_level 1) to a fixed 3-qubit layout chosen
        from backend error data; the wrap-around CZ costs at most 1 SWAP
        on heavy-hex;
      * one SamplerV2 job per (block, setting) round with n_samples PUBs;
        at n=200 samples this is 6 blocks x 3 settings = 18 jobs of 200
        circuits x k shots -- run inside a Session to avoid requeueing;
      * per-shot bitstrings are read from the result's bit arrays.
    """

    def __init__(self, bank: StatevectorBornBank, backend_name: str,
                 session=None, optimization_level: int = 3,
                 initial_layout: Optional[List[int]] = None,
                 dynamical_decoupling: bool = True):
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager)
        self.bank = bank
        self.factory = QiskitCircuitFactory(bank)
        service = QiskitRuntimeService()
        self.backend = service.backend(backend_name)
        self.sampler = SamplerV2(mode=session if session is not None
                                 else self.backend)
        if dynamical_decoupling:
            try:
                self.sampler.options.dynamical_decoupling.enable = True
                self.sampler.options.dynamical_decoupling.sequence_type = \
                    "XpXm"
            except Exception:
                pass
        if initial_layout is None:
            initial_layout = self.pick_best_triple()
            print(f"[LAYOUT] physical qubits {initial_layout} "
                  f"(min two-qubit + readout error chain)")
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=self.backend,
            initial_layout=initial_layout)

    def pick_best_triple(self) -> List[int]:
        """Choose the connected 3-qubit chain a-b-c minimizing the summed
        two-qubit gate error plus readout errors, from backend target data.
        A bad automatic layout is a plausible contributor to strong parity
        damping; pinning the layout removes that variable."""
        t = self.backend.target
        # collect 2q error per edge and readout error per qubit
        e2 = {}
        for op in t.operation_names:
            try:
                props = t[op]
            except Exception:
                continue
            for qargs, ip in props.items():
                if qargs is None or len(qargs) != 2 or ip is None:
                    continue
                if ip.error is not None:
                    key = tuple(sorted(qargs))
                    e2[key] = min(e2.get(key, 1.0), float(ip.error))
        ero = {}
        try:
            for qargs, ip in t["measure"].items():
                if ip is not None and ip.error is not None:
                    ero[qargs[0]] = float(ip.error)
        except Exception:
            pass
        best, best_score = None, None
        adj = {}
        for (a, b) in e2:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        for b, nbrs in adj.items():
            for a in nbrs:
                for cq in nbrs:
                    if cq <= a:
                        continue
                    score = (e2[tuple(sorted((a, b)))]
                             + e2[tuple(sorted((b, cq)))]
                             + ero.get(a, 0.02) + ero.get(b, 0.02)
                             + ero.get(cq, 0.02))
                    if best_score is None or score < best_score:
                        best, best_score = [a, b, cq], score
        return best if best is not None else [0, 1, 2]

    def run(self, S: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        b = self.bank
        angles = b.angles_from_sketch(S)
        n = angles.shape[0]
        rec = np.zeros((b.G, n, k), dtype=np.int64)
        for g, setting in enumerate(_SETTINGS[: b.G]):
            circs = [self.factory.build(angles[i], theta, setting)
                     for i in range(n)]
            isa = self.pm.run(circs)
            job = self.sampler.run(isa, shots=k)
            res = job.result()
            for i in range(n):
                # classical register name defaults to 'c' for measure();
                # take the first (only) bit array of the pub result
                databin = res[i].data
                arr = getattr(databin, "c", None)
                if arr is None:  # fall back to the first field
                    arr = next(iter(databin.__dict__.values()))
                bitstrings = arr.get_bitstrings()
                rec[g, i] = [int(bs.replace(" ", ""), 2)
                             for bs in bitstrings[:k]]
        return rec


# =====================================================================
# autoregressive generation through an executor
# =====================================================================

def generate_progressive_records(bank: StatevectorBornBank,
                                 models: List[BornBlockModel],
                                 d: int, blocks,
                                 sketcher: OnlineCountSketch,
                                 n_samples: int,
                                 executor,
                                 k: int,
                                 theta: Optional[np.ndarray] = None,
                                 clip_nonnegative: bool = True,
                                 feature_correction=None,
                                 record_share: float = 1.0,
                                 verbose: bool = True) -> np.ndarray:
    """Free-running generation identical to sample_progressive_born, but
    with the Born records supplied by `executor` (local / aer / IBM).
    feature_correction=(lam, c) applies the calibrated inverse response
    f_corr = (f_hat - c)/lam (see calibrate_feature_response)."""
    th = bank.get_theta() if theta is None else np.asarray(theta,
                                                           np.float64)
    out = np.zeros((n_samples, d), dtype=np.float64)
    Sraw, cur_len = sketcher.init_state(n_samples)
    for bi, (start, bsz) in enumerate(blocks):
        Sprefix = sketcher.mixed(Sraw, cur_len)
        if record_share < 1.0 and bsz > 1:
            k_sh = int(round(record_share * k))
            k_ex = k - k_sh
            rec = executor.run(Sprefix, th, k_sh + bsz * k_ex)
            cols = []
            for j in range(bsz):
                sl = np.concatenate([np.arange(k_sh),
                                     k_sh + j * k_ex + np.arange(k_ex)])
                f_j = np.concatenate(
                    [bank.T[np.asarray(rec[g], np.int64)[:, sl]].mean(axis=1)
                     for g in range(bank.G)], axis=1)
                if feature_correction is not None:
                    lam, c = feature_correction
                    f_j = (f_j - c[None, :]) / lam[None, :]
                cols.append(f_j @ models[bi].W[:, j])
            Yblk = np.stack(cols, axis=1)
            if getattr(models[bi], "b0", None) is not None:
                Yblk = Yblk + models[bi].b0[None, :]
            if clip_nonnegative:
                Yblk = np.maximum(Yblk, 0.0)
            out[:, start:start + bsz] = Yblk
            cur_len = sketcher.update_inplace(Sraw, cur_len, Yblk)
            if verbose:
                print(f"  [GEN] block {bi + 1}/{len(blocks)} done "
                      f"(record_share={record_share})")
            continue
        rec = executor.run(Sprefix, th, k)
        f_hat = _records_to_features(bank, rec)
        if feature_correction is not None:
            lam, c = feature_correction
            f_hat = (f_hat - c[None, :]) / lam[None, :]
        Yblk = f_hat @ models[bi].W
        if getattr(models[bi], "b0", None) is not None:
            Yblk = Yblk + models[bi].b0[None, :]
        if clip_nonnegative:
            Yblk = np.maximum(Yblk, 0.0)
        out[:, start:start + bsz] = Yblk
        cur_len = sketcher.update_inplace(Sraw, cur_len, Yblk)
        if verbose:
            print(f"  [GEN] block {bi + 1}/{len(blocks)} done "
                  f"({bank.G} settings x {n_samples} circuits x {k} shots)")
    return out


# =====================================================================
# deterministic model reconstruction from a saved run
# =====================================================================

@dataclass
class TrainedBornModel:
    bank: StatevectorBornBank
    models: List[BornBlockModel]
    sketcher: OnlineCountSketch
    blocks: list
    d: int
    k_gen: int
    record_share: float = 1.0            # LEARNED rho (train_rho); the
                                         # deployment path MUST use it
    S_cal: Optional[np.ndarray] = None   # training-side sketches for
                                         # hardware feature calibration


def load_trained_model(results_npz: str, data_dir: str,
                       ridge_alpha: float = 1e-2) -> TrainedBornModel:
    """Rebuild the trained model exactly from the saved run: trained
    angles from the results file, decoders refit in closed form on the
    (deterministically reconstructed) training cache. No retraining."""
    from .config import d12_config
    from .data import load_dataset, select_subset_indices, _resolve_subset_n
    from sklearn.model_selection import train_test_split

    r = np.load(results_npz, allow_pickle=True)
    meta = r["meta"][0]
    cfg = d12_config()

    if "Y_tr" in r.files:
        # exact provenance: the training split is stored in the artifact;
        # no dataset file needed and no risk of a drifted/missing file
        Y_tr = np.asarray(r["Y_tr"], np.float64)
    else:
        X_all = load_dataset(str(data_dir) + "/" + cfg.data.data_path)
        subset_n = _resolve_subset_n(X_all.shape[0],
                                     subset_n=cfg.data.subset_n,
                                     subset_frac=cfg.data.subset_frac)
        if cfg.data.use_subset and cfg.data.subset_apply == "before_split":
            idx = select_subset_indices(X_all, subset_n=subset_n,
                                        mode=cfg.data.subset_mode,
                                        seed=cfg.data.subset_seed)
            X_all = X_all[idx]
        tr_idx, _ = train_test_split(np.arange(len(X_all)),
                                     test_size=cfg.data.test_size,
                                     random_state=42)
        Y_tr = X_all[tr_idx]
    d = Y_tr.shape[1]
    blocks = build_blocks(d, int(meta.get("block_size", 2)))

    sketcher = OnlineCountSketch(
        sketch_dim=cfg.sketch.sketch_dim,
        max_dim=max(cfg.sketch.max_dim_sketch, d),
        use_mixer=cfg.sketch.use_mixer, seed=cfg.data.seed,
        nonlinearity=cfg.sketch.nonlinearity, len_norm=cfg.sketch.len_norm)
    cache = build_prefix_sketch_cache(Y_tr, blocks, sketcher)

    spec = BornBankSpec(n_qubits=3, depth=2,
                        angle_dim=cfg.bank.angle_dim, include_y=True,
                        gen_shots_k=int(meta.get("k_gen", 64)))
    bank = StatevectorBornBank(cfg.sketch.sketch_dim, spec,
                               seed=cfg.data.seed + 13)
    bank.set_theta(np.asarray(r["theta_final"], np.float64))
    bank.A = np.asarray(r["A_final"], np.float64)
    bank.b = np.asarray(r["b_final"], np.float64)

    k_gen = int(meta.get("k_gen", 64))
    models = []
    F_by_block, Y_by_block = [], []
    for bi, (start, bsz) in enumerate(blocks):
        S = cache[bi].astype(np.float64, copy=False)
        Yb = Y_tr[:, start:start + bsz].astype(np.float64, copy=False)
        P = bank.setting_probs(S)
        F = bank.expectation_features(P)
        Sig = bank.parity_covariance_sum(P)
        W, _ = noise_aware_ridge_fit(F, Yb, Sig, ridge_alpha, k=k_gen)
        models.append(BornBlockModel(W=W, start=start, bsz=bsz))
        F_by_block.append(F)
        Y_by_block.append(Yb)

    # calibration sketches: a spread of training-side prefixes across all
    # blocks (8 samples x B blocks), used to measure the hardware feature
    # response (never touches test data)
    idx_cal = np.linspace(0, cache.shape[1] - 1, 8).astype(int)
    S_cal = np.concatenate([cache[bi, idx_cal].astype(np.float64)
                            for bi in range(len(blocks))], axis=0)

    tm = TrainedBornModel(bank=bank, models=models, sketcher=sketcher,
                          blocks=blocks, d=d, k_gen=k_gen,
                          record_share=float(meta.get("record_share", 1.0)),
                          S_cal=S_cal)
    tm._fit_context = (F_by_block, Y_by_block)
    return tm
