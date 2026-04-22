"""
The quantum layer of QFAN.

Jamal Slim
jamal.slim@desy.de


Contains:
  - Pauli string construction and qubit-wise-commuting grouping (paper Lemma 1)
  - TrainableShotPauliBank: the shared parameterized quantum circuit
    (paper Fig. 3; nq qubits, L layers, data re-uploading RY/RZ, variational
    RZ-RY, CZ ring with wrap-around for nq >= 3)
  - compute_features_chunked: memory-bounded wrapper around bank.features
  - parameter_shift_features_and_jacobian: the core quantum-gradient primitive
    (Mitarai 2018; Schuld 2019)

All circuits are executed on qiskit-aer's aer_simulator, identical to the
paper's simulator path. On real hardware (IBM ibm_fez) the same circuit
logic transfers with at most one SWAP for the wrap-around CZ.
"""

from dataclasses import dataclass
from math import pi
from typing import List, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .utils import _stable_sigmoid


# -------------------- Pauli bookkeeping --------------------

def _pauli_strings(nq: int, measure_weight2: bool = True) -> List[str]:
    paulis: List[str] = []
    for q in range(nq):
        z = ['I'] * nq; z[q] = 'Z'
        x = ['I'] * nq; x[q] = 'X'
        paulis.append(''.join(z)); paulis.append(''.join(x))
    if measure_weight2 and nq >= 2:
        for i in range(nq):
            for j in range(i + 1, nq):
                zz = ['I'] * nq; zz[i] = 'Z'; zz[j] = 'Z'
                xx = ['I'] * nq; xx[i] = 'X'; xx[j] = 'X'
                paulis.append(''.join(zz)); paulis.append(''.join(xx))
    return list(dict.fromkeys(paulis))


def _basis_for_pauli_string(pstr: str):
    bases = []
    for ch in pstr:
        if ch == 'I':   bases.append(None)
        elif ch == 'Z': bases.append('Z')
        elif ch == 'X': bases.append('X')
        elif ch == 'Y': bases.append('Y')
        else: raise ValueError("Invalid Pauli char.")
    return bases


def _merge_bases(b1, b2):
    out = []
    for a, b in zip(b1, b2):
        if a is None:   out.append(b)
        elif b is None: out.append(a)
        elif a == b:    out.append(a)
        else: return None
    return out


def group_paulis_qubitwise_commuting(paulis: List[str]):
    """Qubit-wise commuting grouping; returns list of (basis_per_qubit, [paulis]).
    For the paper's {Z_i, X_i, Z_iZ_j, X_iX_j} family this produces G=2 groups
    (all-Z and all-X); Lemma 1 in the QFAN paper."""
    bases = {p: _basis_for_pauli_string(p) for p in paulis}
    groups = []
    for p in paulis:
        bp = bases[p]
        placed = False
        for gi in range(len(groups)):
            gb, glist = groups[gi]
            merged = _merge_bases(gb, bp)
            if merged is not None:
                groups[gi] = (merged, glist + [p]); placed = True; break
        if not placed:
            groups.append((bp, [p]))
    return groups


def _apply_measurement_basis_by_index(qc: QuantumCircuit, basis_per_qubit):
    for qi, b in enumerate(basis_per_qubit):
        if b is None or b == 'Z': continue
        if b == 'X': qc.h(qi)
        elif b == 'Y': qc.sdg(qi); qc.h(qi)
        else: raise ValueError("Unknown basis.")


def _parity_sum_from_counts(counts: dict, pstr: str):
    active = [i for i, ch in enumerate(pstr) if ch != 'I']
    shots_total = 0; psum = 0
    for bitstr, c in counts.items():
        prod = 1
        for qi in active:
            b = int(bitstr[-1 - qi])
            prod *= (1 if b == 0 else -1)
        psum += prod * int(c); shots_total += int(c)
    return int(psum), int(shots_total)


# -------------------- the shot-based Pauli bank --------------------

@dataclass
class ShotBankSpec:
    n_qubits: int = 3
    depth: int = 2
    angle_dim: int = 8
    measure_weight2: bool = True
    shots: int = 256


class TrainableShotPauliBank:
    """
    Shared-theta parameterized quantum circuit producing
        pf = n_q^2 + n_q   Pauli features per sample,
    estimated from G=2 measurement groups (all-Z, all-X).

    Sketch -> angle projection:
        a = sigma(S A^T + b)   in [0,1]^L     (A, b fixed at init)

    Circuit (one layer):
        for k=0..L-1:           RY or RZ (alternating) of pi * a_k on qubit k mod nq
        for q=0..nq-1:          RZ(theta) . RY(theta)     (2 theta per qubit per layer)
        CZ chain q -> q+1 and wrap-around if nq > 2
    Total theta count: p_theta = 2 * depth * n_q.

    Executed on qiskit-aer aer_simulator with `shots` per circuit. Same
    circuit logic transfers to IBM Heron r2 (ibm_fez) with at most one SWAP.
    """
    def __init__(self, sketch_dim: int, spec: ShotBankSpec, seed: int = 7):
        self.spec = spec
        self.m = int(sketch_dim)
        self.rng = np.random.default_rng(seed)

        self.nq = int(spec.n_qubits)
        self.depth = int(spec.depth)
        self.L = int(spec.angle_dim)
        self.shots = int(spec.shots)

        self.A = (0.2 * self.rng.normal(size=(self.L, self.m))).astype(np.float64)
        self.b = (0.05 * self.rng.normal(size=(self.L,))).astype(np.float64)

        self.n_var = self.depth * self.nq * 2
        self.theta = self.rng.normal(0, pi / 4, size=(self.n_var,)).astype(np.float64)

        self.data_params = ParameterVector('a', self.L)
        self.var_params  = ParameterVector('t', self.n_var)
        self.base_circuit = self._build_param_circuit()

        self.paulis = _pauli_strings(self.nq, measure_weight2=self.spec.measure_weight2)
        self.groups = group_paulis_qubitwise_commuting(self.paulis)
        self.pauli_index = {p: i for i, p in enumerate(self.paulis)}

        try:
            from qiskit_aer import Aer
            self.backend = Aer.get_backend("aer_simulator")
        except Exception as e:
            raise RuntimeError("qiskit-aer not available; pip install qiskit-aer") from e
        try:
            self.backend.set_options(seed_simulator=seed)
        except Exception:
            pass

    def feature_dim(self) -> int:
        return len(self.paulis)

    def get_theta(self) -> np.ndarray:
        return self.theta.copy()

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, np.float64).ravel()
        if theta.size != self.n_var:
            raise ValueError("theta dim mismatch")
        self.theta = theta.copy()

    def _build_param_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.nq); t = 0
        for _layer in range(self.depth):
            for k in range(self.L):
                q = k % self.nq
                if (k % 2) == 0: qc.ry(pi * self.data_params[k], q)
                else:            qc.rz(pi * self.data_params[k], q)
            for q in range(self.nq):
                qc.rz(self.var_params[t], q); t += 1
                qc.ry(self.var_params[t], q); t += 1
            if self.nq >= 2:
                for q in range(self.nq - 1): qc.cz(q, q + 1)
                if self.nq > 2: qc.cz(self.nq - 1, 0)
        return qc

    def _angles_from_sketch(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, np.float64)
        return _stable_sigmoid(S @ self.A.T + self.b[None, :])

    def _bind_base(self, angles_row, theta_override=None):
        th = self.theta if theta_override is None \
            else np.asarray(theta_override, np.float64).ravel()
        pd = {self.data_params[i]: float(angles_row[i]) for i in range(self.L)}
        pv = {self.var_params[i]:  float(th[i])         for i in range(self.n_var)}
        return self.base_circuit.assign_parameters({**pd, **pv}, inplace=False)

    def _build_meas_circuit(self, bound: QuantumCircuit, basis_per_qubit) -> QuantumCircuit:
        qc = QuantumCircuit(self.nq, self.nq)
        qc.compose(bound, inplace=True)
        _apply_measurement_basis_by_index(qc, basis_per_qubit)
        qc.measure(range(self.nq), range(self.nq))
        return qc

    def features(self, S: np.ndarray, theta_override=None) -> np.ndarray:
        """
        Returns F in R^{n x pf}. Each column is a Pauli expectation estimated
        from `self.shots` shots in the appropriate measurement setting.
        """
        S = np.asarray(S, np.float64)
        if S.ndim == 1:
            S = S.reshape(1, -1)
        angles = self._angles_from_sketch(S)
        n = angles.shape[0]; p = self.feature_dim()
        F = np.zeros((n, p), dtype=np.float64)
        bound_list = [self._bind_base(angles[i], theta_override=theta_override)
                      for i in range(n)]
        for basis, plist in self.groups:
            circuits = [self._build_meas_circuit(bound_list[i], basis) for i in range(n)]
            job = self.backend.run(circuits, shots=self.shots)
            res = job.result()
            for i in range(n):
                counts = res.get_counts(i)
                for pa in plist:
                    j = self.pauli_index[pa]
                    psum, shots_total = _parity_sum_from_counts(counts, pa)
                    F[i, j] = psum / max(1, shots_total)
        return F


def compute_features_chunked(bank: TrainableShotPauliBank, S: np.ndarray,
                             theta_override=None, chunk_size: int = 256) -> np.ndarray:
    S = np.asarray(S, np.float64)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    n = S.shape[0]
    if chunk_size is None or chunk_size <= 0 or n <= chunk_size:
        return bank.features(S, theta_override=theta_override)
    chunks = []
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        chunks.append(bank.features(S[s:e], theta_override=theta_override))
    return np.vstack(chunks)


# -------------------- parameter-shift gradient of F -------------------

def parameter_shift_features_and_jacobian(bank: TrainableShotPauliBank,
                                          S: np.ndarray,
                                          theta: np.ndarray,
                                          chunk_size: int = 256
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameter-shift rule (Mitarai et al. 2018, Schuld et al. 2019):
        dF / d theta_k  =  (1/2) [F(theta + pi/2 e_k) - F(theta - pi/2 e_k)].

    Each theta_k enters exactly ONE RY or RZ gate (generator has eigenvalues
    +-1/2), so the simple two-term rule is EXACT.

    Returns
    -------
    F : (n, p_f)            features at theta
    J : (p_theta, n, p_f)   J[k] = dF / d theta_k

    Cost: (1 + 2 p_theta) * G * n circuit-group evaluations per call.
    For QFAN at d=12 (p_theta=12, G=2): 50 n circuits per batch.
    For QFAN at d=25 (p_theta=18, G=2): 74 n circuits per batch.
    Per-component variance is 1/N_s (shot noise only), uniformly bounded.
    """
    S = np.asarray(S, np.float64)
    theta = np.asarray(theta, np.float64).ravel()
    n_theta = theta.size

    F = compute_features_chunked(bank, S, theta_override=theta, chunk_size=chunk_size)
    n, p_f = F.shape
    J = np.zeros((n_theta, n, p_f), dtype=np.float64)

    shift = 0.5 * pi
    for k in range(n_theta):
        tp = theta.copy(); tp[k] += shift
        tm = theta.copy(); tm[k] -= shift
        Fp = compute_features_chunked(bank, S, theta_override=tp, chunk_size=chunk_size)
        Fm = compute_features_chunked(bank, S, theta_override=tm, chunk_size=chunk_size)
        J[k] = 0.5 * (Fp - Fm)
    return F, J
