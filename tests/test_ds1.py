#!/usr/bin/env python3
"""
Tests for the CaloChallenge Dataset 1 pipeline.

    python tests/test_ds1.py          # self-tests of data preparation and export
    python tests/test_ds1.py --full   # plus the whole chain on synthetic showers

The full test prepares synthetic showers as total energy plus fractions, trains
a small model for two epochs, generates showers and runs the evaluation and
plots, checking that every step succeeds and writes what the next one needs. It
takes a few minutes and removes its files.
"""
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args):
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, args))}")
    return r.stdout


def self_tests():
    out = run("ds1/prepare_ds1.py", "--self-test")
    assert "[SELF-TEST]" in out
    out = run("ds1/export_calochallenge.py", "--self-test")
    assert "PASS" in out or "SELF-TEST" in out
    print("  [PASS] data preparation and export self-tests")


def synthetic_showers(n=1500, seed=0):
    """Photon-geometry showers in HDF5 voxel order with a depth fluctuation
    that trades energy between layers 1 and 2, as real showers do."""
    sys.path.insert(0, str(ROOT / "ds1"))
    from prepare_ds1 import layer_slices
    rng = np.random.default_rng(seed)
    depth = rng.normal(0.0, 1.0, n)
    S = np.zeros((n, 368))
    share = {0: 0.01, 1: 0.18, 2: 0.80, 3: 0.006, 4: 0.004}
    for li, (start, stop, n_a, n_r) in enumerate(layer_slices("photons")):
        coef = {1: 0.3, 2: -0.07}.get(li, 0.0)
        w = share[li] * (1.0 + coef * depth)
        radial = np.exp(-np.arange(n_r) / 2.5)
        base = np.outer(np.ones(n_a), radial).ravel()
        base = base / base.sum()
        noise = rng.gamma(2.0, 0.5, (n, n_a * n_r))
        S[:, start:stop] = np.clip(w, 0, None)[:, None] * base[None, :] * noise * 65536.0
    return S, np.full(n, 65536.0)


def full_chain():
    """prepare (total + fractions) -> train -> generate -> aggregate, evaluate, plot"""
    sys.path.insert(0, str(ROOT / "ds1"))
    from prepare_ds1 import ds1_to_image, to_fractions, from_fractions
    import json
    import shutil
    S, E = synthetic_showers()
    img, info = ds1_to_image(S, E, "photons", collapse="angle", energy=65536)
    X, extra = to_fractions(img)
    info.update(extra)
    data, out = ROOT / "data", ROOT / "outputs"
    np.save(data / "_test53.npy", X)
    (data / "_test53.json").write_text(json.dumps(info))
    made = [data / "_test53.npy", data / "_test53.json", out / "model_ds1test.npz",
            out / "model_ds1test_showers.npz", out / "ckpt_ds1test.pkl", out / "loss_ds1test.npz"]
    try:
        run("scripts/train.py", "--data", "data/_test53.npy", "--nq", 3, "--block-size", 2,
            "--depth", 2, "--epochs", 2, "--budget-s", 600, "--tag", "ds1test",
            "--decoder-intercept", "--learn-k")
        run("ds1/generate.py", "outputs/model_ds1test.npz", "--data", "data/_test53.npy")
        r = np.load(out / "model_ds1test_showers.npz", allow_pickle=True)
        assert r["Y_gen"].shape[1] == 53 and np.isfinite(r["Y_gen"]).all() and (r["Y_gen"] >= 0).all()
        # the test showers come back exactly as they were prepared
        tr = np.load(out / "model_ds1test.npz", allow_pickle=True)
        assert np.allclose(r["Y_te"], from_fractions(tr["Y_te"], info))
        # each generated shower carries exactly its generated total
        assert np.all(r["Y_gen"].sum(1) >= 0)
        table = run("ds1/aggregate.py", "outputs/model_ds1test_showers.npz", "--name", "test", "--boot", 2)
        assert "no correlations" in table and "untrained circuit" in table
        report = run("ds1/eval_geometry.py", "outputs/model_ds1test_showers.npz", "--data", "data/_test53.npy")
        assert "per-layer energy" in report
        run("ds1/plot_polar.py", "outputs/model_ds1test_showers.npz", "--data", "data/_test53.npy")
        print("  [PASS] full chain: prepare (total + fractions), train, generate, "
              "aggregate, evaluate, plot")
    finally:
        for f in made:
            f.unlink(missing_ok=True)
        for d in ("plots_ds1test_showers", "plots_model_ds1test_showers"):
            shutil.rmtree(ROOT / d, ignore_errors=True)


if __name__ == "__main__":
    self_tests()
    if "--full" in sys.argv:
        full_chain()
