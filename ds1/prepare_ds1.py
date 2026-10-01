#!/usr/bin/env python3
"""
Build a QFAN training image from CaloChallenge Dataset 1.

Dataset 1 is public on Zenodo, DOI 10.5281/zenodo.8099322. It holds GEANT4
showers in the ATLAS calorimeter, from photons and from charged pions. Use
dataset_1_photons_1.hdf5 for training and dataset_1_photons_2.hdf5 for
evaluation.

    pip install h5py
    python ds1/prepare_ds1.py --self-test
    python ds1/prepare_ds1.py --src dataset_1_photons_1.hdf5 --list-energies
    python ds1/prepare_ds1.py --src dataset_1_photons_1.hdf5 --check
    python ds1/prepare_ds1.py --src dataset_1_photons_1.hdf5 \\
           --energy 65536 --collapse angle --out data/cal_shower_img_53q.npy

WHY DATASET 1

Its incident energies are discrete, 15 values from 256 MeV to 4.2 TeV in
powers of two. Selecting one of them gives thousands of showers at exactly
that energy. QFAN is not conditioned on incident energy, so this avoids the
approximate energy band that the continuous spectrum of Dataset 2 forces.

THE GEOMETRY

The ATLAS binning is irregular. Each layer has its own number of radial and
angular bins, written here as (angular, radial):

    photons  (1, 8)  (10, 16)  (10, 19)  (1, 5)  (1, 5)                  368
    pions    (1, 8)  (10, 10)  (10, 10)  (1, 5)  (10, 15)  (10, 16)  (1, 10)  533

Within a layer the flattened voxels run radial fastest, then angle, so a layer
block reshapes to (n_alpha, n_r). Layers follow each other in order, so a
layer is found by its cumulative offset, not by one global reshape.

One published source lists the pion geometry with an extra (1, 5) layer,
eight layers summing to 538. That is a typo; the voxel totals asserted below
catch it, and any other inconsistency of the same kind.

THE LADDER OF IMAGE SIZES

    --collapse layer      one value per layer                    5 or 7
    --collapse angle      radial profile of every layer         53 or 74
    --collapse none       every voxel                          368 or 533

Summing over angle loses little: electromagnetic showers are close to
azimuthally symmetric. Pixels are ordered layer-major, so the autoregressive
chain follows the longitudinal development of the shower.

NORMALIZATION

Each voxel is divided by the incident energy, then the array is scaled by one
global constant so its 99.9th percentile sits at 0.5, the regime of the 12-
and 25-pixel images. Values above it are kept: capping them (--clip, off by
default) trims the shower core in about a third of the 368-voxel showers,
where the brightest values all sit in the few core voxels of layer 2. The
factor is saved alongside the data so the transform can be inverted.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

# (n_alpha, n_r) per layer, and the ATLAS layer identifier for reporting
GEOMETRY = {
    "photons": dict(layers=[(1, 8), (10, 16), (10, 19), (1, 5), (1, 5)],
                    atlas_id=[0, 1, 2, 3, 12], n_vox=368),
    "pions": dict(layers=[(1, 8), (10, 10), (10, 10), (1, 5), (10, 15),
                          (10, 16), (1, 10)],
                  atlas_id=[0, 1, 2, 3, 12, 13, 14], n_vox=533),
}
for _p, _g in GEOMETRY.items():
    _tot = sum(a * r for a, r in _g["layers"])
    assert _tot == _g["n_vox"], f"{_p}: layers sum to {_tot}, not {_g['n_vox']}"

ENERGIES_MEV = [2 ** k for k in range(8, 23)]        # 256 MeV ... 4.19 TeV


def layer_slices(particle):
    """(start, stop, n_alpha, n_r) for each layer in the flat array."""
    out, start = [], 0
    for n_a, n_r in GEOMETRY[particle]["layers"]:
        out.append((start, start + n_a * n_r, n_a, n_r))
        start += n_a * n_r
    return out


def snap_energy(e_mev):
    """Nearest of the 15 generated energies."""
    return min(ENERGIES_MEV, key=lambda x: abs(np.log(x) - np.log(e_mev)))


def ds1_to_image(showers, energies, particle="photons", collapse="angle",
                 energy=None, n_max=None, clip_quantile=0.999, target=0.5,
                 seed=42, order="standard", clip=False):
    """Pure-NumPy core. Returns (image, info).

    showers   (N, 368 or 533), flattened radial fastest within each layer
    energies  (N,) or (N, 1) incident energy in MeV
    energy    MeV; snapped to the nearest generated value. None keeps all.
    """
    g = GEOMETRY[particle]
    showers = np.asarray(showers, np.float64)
    energies = np.asarray(energies, np.float64).reshape(-1)
    if showers.shape[1] != g["n_vox"]:
        raise ValueError(f"{particle}: expected {g['n_vox']} voxels, "
                         f"got {showers.shape[1]}")

    # ---- one incident energy --------------------------------------------
    chosen = None
    if energy is not None:
        chosen = snap_energy(energy)
        keep = np.abs(np.log(energies) - np.log(chosen)) < 0.01
        showers, energies = showers[keep], energies[keep]
        if len(showers) == 0:
            raise ValueError(f"no showers at {chosen} MeV")

    rng = np.random.default_rng(seed)
    if n_max is not None and len(showers) > n_max:
        idx = np.sort(rng.choice(len(showers), n_max, replace=False))
        showers, energies = showers[idx], energies[idx]

    # ---- geometry, layer by layer ---------------------------------------
    blocks = []
    for start, stop, n_a, n_r in layer_slices(particle):
        L = showers[:, start:stop].reshape(-1, n_a, n_r)   # (N, alpha, r)
        if collapse == "none":
            if order == "ring":
                # ring-major: each radial ring's angular voxels sit together
                blocks.append(L.transpose(0, 2, 1).reshape(len(L), -1))
            else:
                blocks.append(L.reshape(len(L), -1))
        elif collapse == "angle":
            blocks.append(L.sum(axis=1))                     # (N, n_r)
        elif collapse == "layer":
            blocks.append(L.sum(axis=(1, 2))[:, None])       # (N, 1)
        else:
            raise ValueError(f"unknown collapse {collapse!r}")
    image = np.concatenate(blocks, axis=1)                   # layer-major

    # ---- normalization --------------------------------------------------
    # One global scale for every voxel. Scaling each pixel separately was
    # tried and hurt: it amplifies the shot noise of the sparsest pixels.
    image = image / energies[:, None]
    q = float(np.quantile(image, clip_quantile))
    scale = target / q if q > 0 else 1.0
    image = image * scale
    image = np.clip(image, 0.0, target) if clip else np.maximum(image, 0.0)

    block_widths = None
    if collapse == "none" and order == "ring":
        block_widths = []
        for n_a, n_r in g["layers"]:
            if n_a > 1:
                block_widths += [n_a] * n_r          # one block per ring
            else:
                block_widths += [2] * (n_r // 2) + ([1] if n_r % 2 else [])
    info = dict(particle=particle, n_showers=int(len(image)),
                d=int(image.shape[1]), collapse=collapse,
                energy_mev=chosen, norm="global", energy_scale=scale,
                clip_quantile=clip_quantile, target=target, clipped=bool(clip),
                frac_zero=float((image == 0).mean()),
                frac_clipped=float((image >= target).mean()),
                layer_widths=[int(b.shape[1]) for b in blocks],
                order=order if collapse == "none" else "standard",
                block_widths=block_widths)
    return image, info


def sanity_check(showers, particle="photons", verbose=True):
    """Checks to run on the REAL file, since the ordering cannot be tested
    here against real data.

    The decisive one is azimuthal flatness. The angular bins are equal-width
    and an electromagnetic shower is close to azimuthally symmetric, so the
    angular profile of a segmented layer must be nearly flat, while the radial
    profile must not be. If the axes were swapped, the roles invert. So the
    ratio CV(radial) / CV(angular) is well above 1 for the correct reading
    and below 1 for a swapped one.
    """
    g = GEOMETRY[particle]
    ok = True
    if showers.shape[1] != g["n_vox"]:
        print(f"  [FAIL] {showers.shape[1]} voxels, expected {g['n_vox']}")
        return False
    E = showers.sum(axis=0)
    tot = E.sum()
    lines = []
    ratios = []
    for i, (start, stop, n_a, n_r) in enumerate(layer_slices(particle)):
        frac = E[start:stop].sum() / tot if tot > 0 else 0
        line = (f"    layer {g['atlas_id'][i]:>2}  ({n_a:>2} x {n_r:>2})  "
                f"{100*frac:5.1f}% of energy")
        if n_a > 1:
            L = E[start:stop].reshape(n_a, n_r)
            ang, rad = L.sum(axis=1), L.sum(axis=0)
            cv_a = ang.std() / (ang.mean() + 1e-300)
            cv_r = rad.std() / (rad.mean() + 1e-300)
            ratio = cv_r / (cv_a + 1e-300)
            ratios.append((frac, ratio))
            line += f"   angular CV {cv_a:.3f}  radial CV {cv_r:.3f}  ratio {ratio:.1f}"
        lines.append(line)
    if verbose:
        print("  per-layer energy and axis-order diagnostic")
        print("\n".join(lines))
    # judge on the most energetic segmented layer, where statistics are best
    if ratios:
        frac, ratio = max(ratios)
        if ratio < 1.5:
            ok = False
            if verbose:
                print(f"  [FAIL] in the densest segmented layer the angular profile "
                      f"is not flatter than the radial one (ratio {ratio:.2f}). "
                      f"The axis order is probably swapped.")
        elif verbose:
            print(f"  [PASS] angular profile flat relative to radial "
                  f"(ratio {ratio:.1f}): axis order consistent")
    return ok


def load_hdf5(path):
    """Thin I/O wrapper. The only part not exercised by self_test()."""
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required to read CaloChallenge files: "
                 "pip install h5py")
    with h5py.File(path, "r") as f:
        return f["showers"][:], f["incident_energies"][:]


def _synthetic(particle, N, rng, swap=False):
    """Showers with realistic structure: energy falling with radius, flat in
    angle, varying by layer. swap=True writes them angle-fastest, the WRONG
    order, so the self-test can confirm the check catches it."""
    g = GEOMETRY[particle]
    rows = []
    for n_a, n_r in g["layers"]:
        rad = np.exp(-np.arange(n_r) / max(1.0, n_r / 4))
        rad /= rad.sum()
        L = (rad[None, None, :] * np.ones((N, n_a, 1)) / n_a)
        L = L * (1 + 0.15 * rng.normal(size=L.shape))
        if swap:
            L = np.swapaxes(L, 1, 2)                 # (N, r, alpha) -> wrong
        rows.append(np.clip(L, 0, None).reshape(N, -1))
    layer_w = rng.dirichlet(np.ones(len(g["layers"])) * 3)
    return np.concatenate([r * w for r, w in zip(rows, layer_w)], axis=1)


def _fractions_round_trip():
    rng = np.random.default_rng(3)
    img = rng.gamma(0.6, 1.0, (400, 53)) * (rng.random((400, 53)) > 0.2)
    for power in (1.0, 0.3):
        X, info = to_fractions(img, power)
        back = from_fractions(X, info)
        assert X.shape == (400, 54) and np.allclose(back, img, rtol=1e-9, atol=1e-12)
        assert np.allclose(back.sum(1), img.sum(1))


def self_test():
    rng = np.random.default_rng(0)
    for particle, g in GEOMETRY.items():
        # 1. one-hot: every voxel lands where the ordering says
        for li, (start, stop, n_a, n_r) in enumerate(layer_slices(particle)):
            for a, r in {(0, 0), (n_a - 1, n_r - 1), (n_a // 2, n_r // 2)}:
                S = np.zeros((1, g["n_vox"]))
                S[0, start + a * n_r + r] = 1.0
                blk = S[0, start:stop].reshape(n_a, n_r)
                assert blk[a, r] == 1.0, f"{particle} layer {li}: order wrong"
                img, _ = ds1_to_image(S, [1.0], particle, "angle",
                                      target=1.0, clip_quantile=1.0)
                off = sum(w for w in
                          [n for _, n in g["layers"][:li]])
                assert img[0, off + r] > 0 and np.count_nonzero(img) == 1

        # 2. sizes of every rung
        S = rng.random((3, g["n_vox"]))
        for c, want in (("none", g["n_vox"]),
                        ("angle", sum(n for _, n in g["layers"])),
                        ("layer", len(g["layers"]))):
            _, info = ds1_to_image(S, np.ones(3), particle, c)
            assert info["d"] == want, f"{particle} {c}: d={info['d']}"

        # 3. every collapse conserves energy before normalization
        for c in ("none", "angle", "layer"):
            parts = []
            for start, stop, n_a, n_r in layer_slices(particle):
                L = S[:, start:stop].reshape(-1, n_a, n_r)
                parts.append(L.reshape(3, -1) if c == "none" else
                             L.sum(1) if c == "angle" else
                             L.sum((1, 2))[:, None])
            assert np.allclose(np.concatenate(parts, 1).sum(1), S.sum(1))

        # 4. energy selection snaps to the generated grid
        E = np.array([65536.0, 65536.0, 131072.0])
        _, info = ds1_to_image(S, E, particle, "angle", energy=65000)
        assert info["n_showers"] == 2 and info["energy_mev"] == 65536

        # 5. the real-data check passes on correct order, fails on swapped
        good = _synthetic(particle, 400, rng, swap=False)
        bad = _synthetic(particle, 400, rng, swap=True)
        assert sanity_check(good, particle, verbose=False), \
            f"{particle}: sanity check rejected correctly ordered data"
        assert not sanity_check(bad, particle, verbose=False), \
            f"{particle}: sanity check accepted swapped data"

    # 6. ring order: every block is the angular voxels of one (layer, ring)
    for particle, g in GEOMETRY.items():
        S = np.zeros((1, g["n_vox"]))
        for li, (start, stop, n_a, n_r) in enumerate(layer_slices(particle)):
            for a_ in range(n_a):
                for r_ in range(n_r):          # encode (layer, ring) in the value
                    S[0, start + a_ * n_r + r_] = 1000 * li + r_ + 1
        img, info = ds1_to_image(S, [1.0], particle, "none", order="ring",
                                 target=1e9, clip_quantile=1.0)
        v = img[0] / img[0].max() * S.max()
        c = 0
        for w in info["block_widths"]:
            vals = np.round(v[c:c + w])
            assert len(set(vals.tolist())) == 1 or w <= 2, \
                f"{particle}: a ring block mixes rings"
            c += w
        assert c == g["n_vox"] and sum(info["layer_widths"]) == g["n_vox"]
    _fractions_round_trip()
    print("[SELF-TEST] total + fractions: the inverse restores every shower exactly")
    print("[SELF-TEST] ring order: each ring block holds exactly one ring")
    print("[SELF-TEST] both geometries: voxel totals, per-layer ordering, "
          "every rung, energy conservation, energy snapping, and the "
          "axis-order check (passes correct order, rejects swapped) all "
          "verified")


# ---- total + fractions -----------------------------------------------------
# Each shower as its total deposited energy followed by the fraction of that
# energy in every pixel. QFAN generates the total first and the fractions after
# it; the inverse normalises the fractions and multiplies by the total, so energy
# is conserved exactly by construction. Both steps are deterministic, and the
# inverse uses only the constants stored at preparation.

T_CENTER, T_WIDTH = 0.3, 0.1        # where the scaled total sits in the prepared data


def to_fractions(image, power=1.0):
    """(N, d) image -> (N, d + 1) prepared data [scaled total, scaled fractions]
    and the constants needed to invert it. With power < 1 each fraction is
    raised to that power first, which compresses the long tails: the Born
    noise of a ridge-decoded value cannot exceed half its mean, and in the
    compressed representation far fewer voxels need more than that."""
    t = image.sum(axis=1)
    center = float(np.median(t))
    width = float(1.4826 * np.median(np.abs(t - center))) or 1.0
    f = image / np.where(t > 0, t, 1.0)[:, None]
    u = f ** power
    q = float(np.quantile(u, 0.999))
    fscale = 0.5 / q if q > 0 else 1.0
    X = np.column_stack([T_CENTER + T_WIDTH * (t - center) / width, u * fscale])
    return X, dict(represent="fractions", total_center=center, total_width=width,
                   frac_scale=fscale, power=float(power))


def from_fractions(X, info):
    """Exact inverse of to_fractions, applied to prepared or generated data:
    the total is unscaled, the fractions are made non-negative and normalised to
    sum to one, and the image is their product."""
    X = np.asarray(X, np.float64)
    t = info["total_center"] + info["total_width"] * (X[:, 0] - T_CENTER) / T_WIDTH
    f = np.maximum(X[:, 1:], 0.0) ** (1.0 / info.get("power", 1.0))
    s = f.sum(axis=1, keepdims=True)
    f = np.where(s > 0, f / np.where(s > 0, s, 1.0), 1.0 / f.shape[1])
    return np.maximum(t, 0.0)[:, None] * f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--particle", default="photons", choices=list(GEOMETRY))
    ap.add_argument("--collapse", default="angle",
                    choices=["none", "angle", "layer"])
    ap.add_argument("--energy", type=float, default=None,
                    help="incident energy in MeV, snapped to the grid")
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--fractions", action="store_true",
                    help="write each shower as its total energy followed by the "
                         "fraction of it in every pixel (d + 1 columns); QFAN then "
                         "generates the total itself and energy is conserved exactly")
    ap.add_argument("--power", type=float, default=1.0,
                    help="with --fractions: raise each fraction to this power (e.g. 0.3), "
                         "compressing the long tails; inverted exactly after generation")
    ap.add_argument("--clip", action="store_true",
                    help="cap values at the scale target, as older versions did "
                         "(not recommended: it trims the shower core)")
    ap.add_argument("--order", default="standard", choices=["standard", "ring"],
                    help="with --collapse none: 'ring' puts each ring's angular "
                         "voxels together and cuts one block per ring")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list-energies", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="run the axis-order check on the real file")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if not a.src:
        ap.error("--src is required (or use --self-test)")

    showers, energies = load_hdf5(a.src)
    energies = np.asarray(energies).reshape(-1)

    if a.list_energies:
        print(f"  {'energy':>12}  {'showers':>8}")
        for e in ENERGIES_MEV:
            n = int((np.abs(np.log(energies) - np.log(e)) < 0.01).sum())
            if n:
                lab = f"{e/1000:g} GeV" if e >= 1000 else f"{e} MeV"
                print(f"  {lab:>12}  {n:>8}")
        return

    if a.check:
        ok = sanity_check(showers, a.particle)
        sys.exit(0 if ok else 1)

    if not a.out:
        ap.error("--out is required")
    image, info = ds1_to_image(showers, energies, a.particle, a.collapse,
                               a.energy, a.n_max, order=a.order, clip=a.clip)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.fractions:
        X, extra = to_fractions(image, a.power)
        info.update(extra)
        np.save(out, X)
    else:
        np.save(out, image)
    info["source"] = str(a.src)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))

    print(f"[DS1] wrote {out}  shape {np.load(out, mmap_mode='r').shape}"
          + ("  (total + fractions)" if a.fractions else ""))
    print(f"      {a.particle}, layer widths {info['layer_widths']}")
    print(f"      energy {info['energy_mev']} MeV, {info['n_showers']} showers")
    print(f"      zeros {100*info['frac_zero']:.1f}%   "
          f"above the scale target {100*info['frac_clipped']:.2f}%"
          + ("  (capped)" if a.clip else "  (kept)"))
    if info["n_showers"] < 2000:
        print("      NOTE: under 2000 showers. The 12- and 25-pixel runs used "
              "4800 for training; expect noisier results.")
    if info["frac_zero"] > 0.3:
        print("      NOTE: over 30% of entries are zero, where the 12- and "
              "25-pixel data have none.")


if __name__ == "__main__":
    main()
