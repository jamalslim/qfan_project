#!/usr/bin/env python3
"""
Export the QFAN circuits as OpenQASM 2.0.

The circuit is the same at every autoregressive step. Only the encoding angles
change, since they are computed from the sketch of the pixels generated so far.
So a complete description is: one file per measurement setting per image size,
with the trained circuit parameters baked in and the encoding angles taken from
a specified block.

    python export_qasm.py --d 25 --block 0 --out qasm/

writes qasm/qfan_d25_block00_Z.qasm and the X and Y counterparts.

    python export_qasm.py --d 25 --all-blocks --out qasm/

writes all blocks, that is 13 x 3 files at d=25 and 6 x 3 at d=12.

The circuit structure, per layer, is

    data re-uploading   RY(pi a_k) on qubit k mod n_q for even k
                        RZ(pi a_k) on qubit k mod n_q for odd k
    trainable           RZ(theta), RY(theta) on every qubit
    entangling          CZ ring, (0,1) (1,2) ... (n_q-1, 0)

repeated `depth` times, followed by the basis rotation

    Z   nothing
    X   H on every qubit
    Y   Sdg then H on every qubit

and a measurement of every qubit. Qiskit bitstring ordering is little-endian,
so bit 0 of the returned string is qubit 0.

No qiskit is required to produce the files. If qiskit is installed the script
additionally checks that the emitted QASM reproduces the statevector engine's
outcome probabilities, which is the guarantee that the exported circuits are
the ones the paper's results were computed with.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve()
PROJECT_ROOT = next(c for c in [ROOT.parent] + list(ROOT.parents)
                    if (c / "src" / "qfan").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

PI = np.pi


def emit_qasm(angles, theta, nq, depth, L, setting):
    """Return an OpenQASM 2.0 string. Mirrors QiskitCircuitFactory.build()
    gate for gate, so the two are interchangeable."""
    o = ["OPENQASM 2.0;", 'include "qelib1.inc";',
         f"qreg q[{nq}];", f"creg c[{nq}];", ""]
    t = 0
    for layer in range(depth):
        o.append(f"// ---- layer {layer}: data re-uploading ----")
        for kk in range(L):
            q = kk % nq
            ang = PI * float(angles[kk])
            o.append(f"{'ry' if kk % 2 == 0 else 'rz'}({ang:.17g}) q[{q}];")
        o.append(f"// ---- layer {layer}: trainable rotations ----")
        for q in range(nq):
            o.append(f"rz({float(theta[t]):.17g}) q[{q}];"); t += 1
            o.append(f"ry({float(theta[t]):.17g}) q[{q}];"); t += 1
        if nq >= 2:
            o.append(f"// ---- layer {layer}: CZ ring ----")
            for q in range(nq - 1):
                o.append(f"cz q[{q}],q[{q+1}];")
            if nq > 2:
                o.append(f"cz q[{nq-1}],q[0];")
        o.append("")
    if setting == "X":
        o.append("// ---- X basis ----")
        o += [f"h q[{q}];" for q in range(nq)]
    elif setting == "Y":
        o.append("// ---- Y basis ----")
        for q in range(nq):
            o += [f"sdg q[{q}];", f"h q[{q}];"]
    elif setting != "Z":
        raise ValueError(setting)
    o.append("")
    o += [f"measure q[{q}] -> c[{q}];" for q in range(nq)]
    return "\n".join(o) + "\n"


def verify(qasm, probs_ref):
    """If qiskit is available, check the QASM reproduces the engine."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError:
        return None
    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.remove_final_measurements(inplace=True)
    p = np.abs(Statevector(qc).data) ** 2
    return float(np.abs(p - probs_ref).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, choices=[12, 25], required=True)
    ap.add_argument("--block", type=int, default=0,
                    help="which autoregressive step to take the angles from")
    ap.add_argument("--all-blocks", action="store_true")
    ap.add_argument("--sample", type=int, default=0,
                    help="which conditioning sample to use")
    ap.add_argument("--out", default="qasm")
    args = ap.parse_args()

    import train as T
    T.D_PIXELS = args.d
    T.TAG = T._PER_D[args.d]["tag"]
    (cfg, Y_tr, Y_te, d, blocks, sk, cache, spec, bank) = T.build_problem(args.d)

    r = np.load(PROJECT_ROOT / "outputs" / f"model_{T.TAG}.npz",
                allow_pickle=True)
    theta = r["theta_final"]
    bank.set_theta(theta)
    bank.A = r["A_final"].copy()
    bank.b = r["b_final"].copy()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = range(len(blocks)) if args.all_blocks else [args.block]

    n = 0
    worst = 0.0
    for bi in todo:
        S = cache[bi][args.sample:args.sample + 1].astype(np.float64)
        angles = bank.angles_from_sketch(S)[0]
        P = bank.setting_probs(S)          # (G, 1, 2^nq) exact reference
        for g, setting in enumerate(("Z", "X", "Y")):
            q = emit_qasm(angles, theta, bank.nq, bank.depth, bank.L, setting)
            f = out / f"qfan_d{args.d}_block{bi:02d}_{setting}.qasm"
            f.write_text(q)
            n += 1
            e = verify(q, P[g, 0])
            if e is not None:
                worst = max(worst, e)

    print(f"[QASM] wrote {n} files to {out}/")
    print(f"       n_q={bank.nq}  depth={bank.depth}  angle_dim={bank.L}  "
          f"theta={bank.n_var}")
    if worst:
        print(f"       verified against the statevector engine, "
              f"max |dP| = {worst:.2e}")
    else:
        print("       install qiskit to cross-check the files against the "
              "engine")


if __name__ == "__main__":
    main()
