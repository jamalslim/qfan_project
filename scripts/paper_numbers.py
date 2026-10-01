#!/usr/bin/env python3
"""
paper_numbers.py -- compute every number the QFAN QFAN manuscript needs and
emit paste-ready LaTeX tables.

Self-contained: needs only numpy + scipy. It does NOT import the qfan package,
so it runs anywhere the .npz artifacts are present.

WHAT IT PRODUCES
  1. Headline table       : raw and calibrated, per scale, per backend
  2. Certificate table    : C1..C6 at both scales  (Table `tab:certificates`)
  3. Per-pixel W1 table   : d=12, per backend      (Table `tab:w1_pixels`)
  4. Energy-observable    : W1(E), Var(E), sigma(E), KS(E) per configuration
  5. Correlation norms    : off / within / cross MAE and ||dC||_F/d
  6. Spearman off-diag    : rank-correlation error
  7. Bootstrap 95% CIs    : on every headline metric
  8. W1 floor             : split-half of the test set (irreducible)
  9. A MISSING report     : exactly which paper placeholders remain unfilled

USAGE
    python paper_numbers.py --outputs outputs/ --latex tables/
    python paper_numbers.py --outputs outputs/ --scale 25 --verbose

ARRAY CONVENTIONS EXPECTED IN THE .npz FILES
    Y_te   test/MC truth              Y_gen / Y_head   model samples
    Y_mean C1 conditional means       Y_cls  C2 classical features
    Y_scr  C3 scrambled sketch        Y_ini  C4 untrained circuit
    Y_cop  C5 copula-calibrated       Y_lsq  C6 plain least squares
Missing arrays are skipped and reported, never silently zero-filled.
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, rankdata

# ----------------------------------------------------------------------
# block layout: QFAN uses b=2 with a singleton final block when d is odd
# ----------------------------------------------------------------------
def blocks_for(d, b=2):
    out, s = [], 0
    while s < d:
        out.append((s, min(b, d - s)))
        s += b
    return out


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def degeneracy(Y):
    """Detect a collapsed generator. Returns (n_distinct_rows, n_const_cols).

    If the sampler has no stochastic component every generated row is
    identical; the correlation matrix is then UNDEFINED and any number
    derived from it is an artifact of whatever NaN-fill convention is used.
    This must be reported, never silently filled."""
    Y = np.asarray(Y, float)
    n_distinct = len(np.unique(np.round(Y, 12), axis=0))
    n_const = int((Y.std(axis=0) <= 1e-12).sum())
    return n_distinct, n_const


def corr_safe(Y):
    """Pearson correlation, constant columns -> 0 correlation, diag 1."""
    Y = np.asarray(Y, float)
    sd = Y.std(axis=0)
    ok = sd > 1e-12
    C = np.eye(Y.shape[1])
    if ok.sum() > 1:
        C_ok = np.corrcoef(Y[:, ok].T)
        idx = np.where(ok)[0]
        C[np.ix_(idx, idx)] = np.nan_to_num(C_ok, nan=0.0)
    return C


def spearman_corr(Y):
    R = np.apply_along_axis(rankdata, 0, np.asarray(Y, float))
    return corr_safe(R)


def masks(d, blocks):
    off = ~np.eye(d, dtype=bool)
    within = np.zeros((d, d), bool)
    for s, bs in blocks:
        within[s:s + bs, s:s + bs] = True
    within &= off
    cross = off & ~within
    return off, within, cross


def corr_summary(C_ref, C_gen, d, blocks):
    off, within, cross = masks(d, blocks)
    D = np.abs(C_gen - C_ref)
    return dict(
        off=float(D[off].mean()),
        within=float(D[within].mean()) if within.any() else float("nan"),
        cross=float(D[cross].mean()) if cross.any() else float("nan"),
        fro_over_d=float(np.linalg.norm(C_gen - C_ref, "fro") / d),
    )


def energy_stats(Y_ref, Y):
    Er, E = Y_ref.sum(1), Y.sum(1)
    return dict(
        w1_E=float(wasserstein_distance(Er, E)),
        var_E=float(E.var()),
        sigma_E=float(E.std()),
        ks_E=float(ks_2samp(Er, E).statistic),
        mean_E=float(E.mean()),
    )


def full_metrics(Y_ref, Y, blocks=None, b=2):
    Y_ref = np.asarray(Y_ref, float)
    Y = np.asarray(Y, float)
    d = Y_ref.shape[1]
    if blocks is None:
        blocks = blocks_for(d, b)
    w1 = np.array([wasserstein_distance(Y_ref[:, j], Y[:, j])
                   for j in range(d)])
    cs = corr_summary(corr_safe(Y_ref), corr_safe(Y), d, blocks)
    sp = corr_summary(spearman_corr(Y_ref), spearman_corr(Y), d, blocks)
    nd, nc = degeneracy(Y)
    m = dict(w1_mean=float(w1.mean()), w1_median=float(np.median(w1)),
             w1_max=float(w1.max()), w1_per_pixel=w1,
             sp_off=sp["off"], n=len(Y),
             n_distinct=nd, n_const_cols=nc, degenerate=(nd <= 1 or nc == d))
    m.update(cs)
    m.update(energy_stats(Y_ref, Y))
    return m


def bootstrap_ci(Y_ref, Y, blocks, n_boot=200, seed=0, alpha=0.05):
    """95% CI on the headline metrics, resampling BOTH sets by rows."""
    rng = np.random.default_rng(seed)
    d = Y_ref.shape[1]
    keys = ["w1_mean", "off", "sp_off", "w1_E"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, len(Y_ref), len(Y_ref))
        j = rng.integers(0, len(Y), len(Y))
        mm = full_metrics(Y_ref[i], Y[j], blocks)
        for k in keys:
            acc[k].append(mm[k])
    return {k: (float(np.quantile(v, alpha / 2)),
                float(np.quantile(v, 1 - alpha / 2))) for k, v in acc.items()}


def w1_floor(Y_te, seed=0, reps=20):
    """Irreducible W1 from finite test statistics: split-half of the truth."""
    rng = np.random.default_rng(seed)
    n, d = Y_te.shape
    vals = []
    for _ in range(reps):
        p = rng.permutation(n)
        a, b_ = Y_te[p[: n // 2]], Y_te[p[n // 2:]]
        vals.append(np.mean([wasserstein_distance(a[:, j], b_[:, j])
                             for j in range(d)]))
    return float(np.mean(vals))


# ----------------------------------------------------------------------
# artifact discovery
# ----------------------------------------------------------------------
CERTS = [
    ("Y_gen",  "headline",        "---"),
    ("Y_head", "headline(head)",  "---"),
    ("Y_mean", "C1 cond. means",  "Born measurement randomness"),
    ("Y_cls",  "C2 classical",    "quantum feature map"),
    ("Y_scr",  "C3 scrambled",    "prefix conditioning"),
    ("Y_ini",  "C4 untrained",    "training of theta"),
    ("Y_cop",  "C5 copula",       "--- (adds calibration)"),
    ("Y_lsq",  "C6 least squares", "noise-aware decoding"),
]


def load_npz(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: {e}")
        return None


def find_truth(r):
    for k in ("Y_te", "Y_te_ref", "Y_ref"):
        if k in r.files:
            return np.asarray(r[k], float)
    return None


def analyze_file(path, b=2, n_boot=0, verbose=False):
    r = load_npz(path)
    if r is None:
        return None
    Y_te = find_truth(r)
    if Y_te is None:
        if verbose:
            print(f"  [skip] {os.path.basename(path)}: no truth array")
        return None
    d = Y_te.shape[1]
    blocks = blocks_for(d, b)
    out = dict(file=os.path.basename(path), d=d, B=len(blocks),
               n_test=len(Y_te), rows={})
    for key, label, removed in CERTS:
        if key not in r.files:
            continue
        Y = np.asarray(r[key], float)
        if Y.ndim != 2 or Y.shape[1] != d:
            continue
        m = full_metrics(Y_te, Y, blocks)
        m["removed"] = removed
        if n_boot and key in ("Y_gen", "Y_head"):
            m["ci"] = bootstrap_ci(Y_te, Y, blocks, n_boot=n_boot)
        out["rows"][label] = m
    out["w1_floor"] = w1_floor(Y_te)
    if "meta" in r.files:
        try:
            mm = r["meta"][0]
            if isinstance(mm, dict):
                out["meta"] = {k: v for k, v in mm.items()
                               if not isinstance(v, dict)}
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def print_report(res):
    print(f"\n{'='*94}")
    print(f"FILE {res['file']}   d={res['d']}  B={res['B']}  "
          f"n_test={res['n_test']}   W1 floor={res['w1_floor']:.5f}")
    if "meta" in res:
        keep = {k: res["meta"][k] for k in
                ("k_shots", "k_gen", "block_size", "epochs", "depth",
                 "record_share", "executor", "backend", "k")
                if k in res["meta"]}
        if keep:
            print("  config:", keep)
    print(f"{'-'*94}")
    print(f"{'configuration':22s} {'W1':>8s} {'off':>8s} {'within':>8s} "
          f"{'cross':>8s} {'|dC|F/d':>8s} {'Sp off':>8s} {'W1(E)':>8s} "
          f"{'Var(E)':>8s}")
    for label, m in res["rows"].items():
        if m.get("degenerate"):
            print(f"{label:22s} {m['w1_mean']:8.5f} "
                  f"{'DEGENERATE':>8s} {'--':>8s} {'--':>8s} {'--':>8s} "
                  f"{'--':>8s} {m['w1_E']:8.4f} {m['var_E']:8.4f}")
            print(f"{'':22s}   ^ only {m['n_distinct']} distinct row(s), "
                  f"{m['n_const_cols']}/{res['d']} constant columns: the "
                  f"sampler has collapsed to a point mass.")
            print(f"{'':22s}     Correlation is UNDEFINED here. Do NOT quote a "
                  f"correlation error for this row.")
            continue
        print(f"{label:22s} {m['w1_mean']:8.5f} {m['off']:8.4f} "
              f"{m['within']:8.4f} {m['cross']:8.4f} {m['fro_over_d']:8.4f} "
              f"{m['sp_off']:8.4f} {m['w1_E']:8.4f} {m['var_E']:8.4f}")
        if "ci" in m:
            c = m["ci"]
            print(f"{'  95% CI':22s} [{c['w1_mean'][0]:.5f},{c['w1_mean'][1]:.5f}]"
                  f"  off[{c['off'][0]:.4f},{c['off'][1]:.4f}]"
                  f"  Sp[{c['sp_off'][0]:.4f},{c['sp_off'][1]:.4f}]"
                  f"  W1E[{c['w1_E'][0]:.4f},{c['w1_E'][1]:.4f}]")


def latex_certificates(res12, res25, path):
    """Table `tab:certificates` in the manuscript."""
    def get(res, label, field):
        if res is None or label not in res.get("rows", {}):
            return r"\needsnum{?}"
        m = res["rows"][label]
        if m.get("degenerate") and field in ("off", "within", "cross"):
            return r"undef."
        return f"{m[field]:.4f}"
    order = [("headline", "---"),
             ("C1 cond. means", "Born measurement randomness"),
             ("C2 classical", "quantum feature map"),
             ("C3 scrambled", "prefix conditioning"),
             ("C4 untrained", "training of $\\btheta$"),
             ("C5 copula", "--- (adds calibration)"),
             ("C6 least squares", "noise-aware decoding")]
    L = [r"\begin{table*}[t]", r"  \centering",
         r"  \caption{Ablation certificates. Each row removes one component "
         r"and refits all remaining stages. Generated by \texttt{paper\_numbers.py}.}",
         r"  \label{tab:certificates}", r"  \small",
         r"  \begin{tabular}{@{}llcccccc@{}}", r"    \toprule",
         r"    & & \multicolumn{3}{c}{$d{=}12$} & \multicolumn{3}{c}{$d{=}25$} \\",
         r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"    Configuration & Component removed & $\bar W_1$ & off & within "
         r"& $\bar W_1$ & off & within \\", r"    \midrule"]
    for label, removed in order:
        L.append(f"    {label} & {removed} & "
                 f"{get(res12,label,'w1_mean')} & {get(res12,label,'off')} & "
                 f"{get(res12,label,'within')} & "
                 f"{get(res25,label,'w1_mean')} & {get(res25,label,'off')} & "
                 f"{get(res25,label,'within')} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"[LaTeX] {path}")


def _column_label(fname):
    f = fname.lower()
    if "ibm" in f:
        return r"\texttt{ibm\_fez}"
    if "aer_noise" in f or "aer-noise" in f:
        return "Aer noise"
    if "aer" in f:
        return "Aer"
    if "noisy" in f:
        return "synth. noise"
    if "local" in f:
        return "local"
    return "simulator"


def latex_w1_pixels(results_for_d, path):
    """Per-pixel W1 table with ONE COLUMN PER ARTIFACT.

    Pass every file for a given image size (simulator and hardware) and the
    table comes out complete. Previously this took a single result and so
    silently produced a one-column table whenever the hardware run lived in
    a separate .npz -- which is the normal case."""
    cols = []
    for res in results_for_d:
        row = res["rows"].get("headline") or res["rows"].get("headline(head)")
        if row is None:
            continue
        cols.append((_column_label(res["file"]), row, res))
    if not cols:
        return
    d = results_for_d[0]["d"]
    spec = "@{}c" + "c" * len(cols) + "@{}"
    L = [r"\begin{table}[b]", r"  \centering",
         r"  \caption{Per-pixel Wasserstein-1 distances at $d{=}%d$. "
         r"Statistical floors: %s. Generated by \texttt{paper\_numbers.py}.}"
         % (d, ", ".join(f"{c[0]} {c[2]['w1_floor']:.4f} ($n{{=}}{c[2]['n_test']}$)"
                         for c in cols)),
         r"  \label{tab:w1_pixels}", r"  \small",
         r"  \begin{tabular}{%s}" % spec, r"    \toprule",
         r"    \textbf{Pixel} & " + " & ".join(f"\\textbf{{{c[0]}}}" for c in cols)
         + r" \\", r"    \midrule"]
    for j in range(d):
        L.append(f"    {j} & " + " & ".join(
            f"{c[1]['w1_per_pixel'][j]:.5f}" for c in cols) + r" \\")
    L.append(r"    \midrule")
    for stat, key in (("Mean", "w1_mean"), ("Median", "w1_median"),
                      ("Max", "w1_max")):
        L.append(f"    {stat} & " + " & ".join(
            f"{c[1][key]:.5f}" for c in cols) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"[LaTeX] {path}  ({len(cols)} column(s): "
          f"{', '.join(c[0] for c in cols)})")
    if len(cols) == 1:
        print("[LaTeX] NOTE: only one artifact for this scale was supplied, so "
              "the table\n        has a single column. Pass the hardware .npz "
              "as well for a complete table.")


def missing_report(found):
    print(f"\n{'='*94}\nPAPER PLACEHOLDERS -- STATUS\n{'='*94}")
    need = [
        ("d=12 simulator headline",  "d12_sim"),
        ("d=12 hardware headline",   "d12_hw"),
        ("d=25 simulator headline",  "d25_sim"),
        ("d=25 hardware headline",   "d25_hw"),
        ("C1..C6 at d=12",           "d12_cert"),
        ("C1..C6 at d=25",           "d25_cert"),
        ("energy stats per backend", "energy"),
    ]
    for label, key in need:
        print(f"  [{'OK ' if found.get(key) else '?? '}] {label}")
    if not found.get("d12_hw") or not found.get("d25_hw"):
        print("\n  Hardware rows are the main gap. Point --outputs at the")
        print("  directory holding the ibm_fez result files, or pass them")
        print("  explicitly with --files a.npz b.npz ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs",
                    help="directory containing the .npz artifacts")
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit file list (overrides --outputs glob)")
    ap.add_argument("--latex", default=None,
                    help="directory to write LaTeX table fragments")
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--boot", type=int, default=200,
                    help="bootstrap resamples for CIs (0 to disable)")
    ap.add_argument("--prefer", default=None,
                    help="substring selecting which artifact to "
                         "report per scale, e.g. model_d25")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(args.outputs, "*.npz")))
    if not files:
        sys.exit(f"no .npz files found in {args.outputs}")

    results, found = [], {}
    for f in files:
        res = analyze_file(f, b=args.block_size, n_boot=args.boot,
                           verbose=args.verbose)
        if res is None or not res["rows"]:
            continue
        results.append(res)
        print_report(res)
        tag = f"d{res['d']}"
        low = res["file"].lower()
        is_hw = any(s in low for s in ("ibm", "aer", "hw", "fez"))
        found[f"{tag}_hw" if is_hw else f"{tag}_sim"] = True
        if any(k.startswith("C") for k in res["rows"]):
            found[f"{tag}_cert"] = True
        found["energy"] = True

    # richest certificate file per scale drives the LaTeX
    def richest(d, prefer=None):
        """Pick the artifact for scale d. Prefers an explicit --prefer
        substring, then a model_* filename, then the most complete file.
        Without this, alphabetical order silently favours older variants
        (an older artifact before the current one), which puts the wrong
        numbers into the paper."""
        cand = [r for r in results if r["d"] == d]
        if not cand:
            return None
        if prefer:
            hit = [r for r in cand if prefer in r["file"]]
            if hit:
                return max(hit, key=lambda r: len(r["rows"]))
        for tag in ("model_", "samples_"):
            hit = [r for r in cand if tag in r["file"]]
            if hit:
                sel = max(hit, key=lambda r: len(r["rows"]))
                if len(cand) > 1:
                    print(f"[select] d={d}: using {sel['file']} "
                          f"(candidates: {', '.join(c['file'] for c in cand)})")
                return sel
        sel = max(cand, key=lambda r: len(r["rows"]))
        if len(cand) > 1:
            print(f"[select] d={d}: using {sel['file']} "
                  f"(candidates: {', '.join(c['file'] for c in cand)})")
            print("[select] WARNING: no model_ or samples_ file matched; verify "
                  "this is the artifact you intend to report.")
        return sel

    r12, r25 = richest(12, args.prefer), richest(25, args.prefer)
    if args.latex:
        os.makedirs(args.latex, exist_ok=True)
        latex_certificates(r12, r25, os.path.join(args.latex,
                                                  "tab_certificates.tex"))
        d12_all = [r for r in results if r["d"] == 12]
        # simulator column first, then hardware
        d12_all.sort(key=lambda r: (0 if _column_label(r["file"]) == "simulator"
                                    else 1, r["file"]))
        latex_w1_pixels(d12_all, os.path.join(args.latex, "tab_w1_pixels.tex"))

    missing_report(found)
    print("\nNote: correlation metrics are unaffected by rank-preserving")
    print("marginal calibration; W1 metrics are not. Report both variants.")


if __name__ == "__main__":
    main()
