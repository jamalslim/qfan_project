
"""
Generate paper-style plots from a saved results.npz file.

Jamal Slim
jamal.slim@desy.de



    python scripts/plot_results.py [run_tag]

run_tag defaults to "d12"; looks for outputs/qfan_v3_<run_tag>_results.npz
and outputs/qfan_v3_<run_tag>_loss_curve.npz, and writes:

    plots/loss_curve_<run_tag>.png        training curves
    plots/marginals_<run_tag>.png         per-pixel marginals (MC vs QFAN)
    plots/correlation_<run_tag>.png       Pearson correlation matrices
    plots/total_energy_<run_tag>.png      total-energy distribution
    plots/per_dim_w1_<run_tag>.png        per-pixel W1 distance
"""

import sys
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR      = PROJECT_ROOT / "outputs"
PLOT_DIR     = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(loss_data, run_tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    tl = loss_data["train_loss"]
    vl = loss_data["val_loss"]
    ve = loss_data["val_loss_ema"] if "val_loss_ema" in loss_data.files else None
    x = np.arange(len(vl))
    ax.plot(x, tl, label="train minibatch", alpha=0.4, color="tab:gray")
    ax.plot(x, vl, label="val at θ (held-fixed batch)", lw=2, color="tab:blue")
    if ve is not None and len(ve):
        ax.plot(x, ve, label="val EMA (β=0.9)", lw=1.5, ls="--", color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("sum of per-block multi-kernel MMD²")
    ax.set_title(f"Loss curve  [{str(loss_data.get('train_mode', 'exact'))}]")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(loss_data["grad_norm"])
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\|\nabla_\theta L\|$")
    ax.set_title("Gradient norm"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(loss_data["step_norm"], label="‖Δθ‖")
    if "lr" in loss_data.files:
        ax2 = ax.twinx()
        ax2.plot(loss_data["lr"], color="tab:red", alpha=0.7, label="lr")
        ax2.set_ylabel("learning rate", color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\|\theta_{t+1} - \theta_t\|$")
    ax.set_title("Step size / LR"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / f"loss_curve_{run_tag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def plot_marginals(results, run_tag: str):
    Y_te = results["Y_te"]; Y_gen = results["Y_gen"]; Y_gen_raw = results["Y_gen_raw"]
    d = Y_te.shape[1]
    ncol = 4
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.6 * nrow))
    axes = axes.ravel()
    for j in range(d):
        ax = axes[j]
        lo = min(Y_te[:, j].min(), Y_gen[:, j].min())
        hi = max(Y_te[:, j].max(), Y_gen[:, j].max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(Y_te[:, j], bins=bins, density=True, alpha=0.45, label="MC", color="tab:blue")
        ax.hist(Y_gen[:, j], bins=bins, density=True, alpha=0.45, label="QFAN (cal)", color="tab:green")
        ax.set_title(f"pixel {j}")
        if j == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    for j in range(d, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out_path = PLOT_DIR / f"marginals_{run_tag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def plot_correlation(results, run_tag: str):
    C_test = results["C_test"]; C_gen_raw = results["C_gen_raw"]; C_gen = results["C_gen"]
    d = C_test.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    titles = [
        "MC truth", "QFAN (raw)", "QFAN (cal)",
        "MC − QFAN raw", "MC − QFAN cal", "raw − cal",
    ]
    data = [C_test, C_gen_raw, C_gen,
            C_test - C_gen_raw, C_test - C_gen, C_gen_raw - C_gen]
    for ax, title, M in zip(axes.ravel(), titles, data):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("pixel"); ax.set_ylabel("pixel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = PLOT_DIR / f"correlation_{run_tag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def plot_total_energy(results, run_tag: str):
    Y_te = results["Y_te"]; Y_gen = results["Y_gen"]; Y_gen_raw = results["Y_gen_raw"]
    E_te  = Y_te.sum(axis=1)
    E_raw = Y_gen_raw.sum(axis=1)
    E_gen = Y_gen.sum(axis=1)
    edges = np.histogram_bin_edges(np.concatenate([E_te, E_raw, E_gen]), bins=50)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.hist(E_te,  bins=edges, density=True, alpha=0.45, label="MC",          color="tab:blue")
    ax.hist(E_raw, bins=edges, density=True, alpha=0.45, label="QFAN (raw)",  color="tab:orange")
    ax.hist(E_gen, bins=edges, density=True, alpha=0.45, label="QFAN (cal)",  color="tab:green")
    ax.set_xlabel(r"Total energy $E = \sum_j y_j$")
    ax.set_ylabel("density")
    ax.set_title("Total deposited energy")
    ax.legend(); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path = PLOT_DIR / f"total_energy_{run_tag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def plot_per_dim_w1(results, run_tag: str):
    w1_raw = results["w1_per_dim_raw"]
    w1_cal = results["w1_per_dim"]
    d = w1_raw.size
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    x = np.arange(d)
    ax.bar(x - 0.2, w1_raw, width=0.4, label="QFAN raw", alpha=0.75)
    ax.bar(x + 0.2, w1_cal, width=0.4, label="QFAN cal", alpha=0.75)
    ax.set_xlabel("pixel")
    ax.set_ylabel(r"Wasserstein-1 distance to MC")
    ax.set_title("Per-pixel W1")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    out_path = PLOT_DIR / f"per_dim_w1_{run_tag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def main():
    run_tag = sys.argv[1] if len(sys.argv) > 1 else "d12"
    results_path = OUT_DIR / f"qfan_v3_{run_tag}_results.npz"
    loss_path    = OUT_DIR / f"qfan_v3_{run_tag}_loss_curve.npz"
    if not results_path.exists():
        print(f"[ERROR] {results_path} not found. Run the corresponding script first.")
        sys.exit(1)

    print(f"[LOAD] {results_path}")
    results = np.load(results_path, allow_pickle=True)
    if loss_path.exists():
        print(f"[LOAD] {loss_path}")
        loss_data = np.load(loss_path, allow_pickle=True)
        plot_loss_curve(loss_data, run_tag)
    else:
        print(f"[WARN] {loss_path} not found; skipping loss curve.")

    plot_marginals(results, run_tag)
    plot_correlation(results, run_tag)
    plot_total_energy(results, run_tag)
    plot_per_dim_w1(results, run_tag)

    print("[DONE] All plots saved to", PLOT_DIR)


if __name__ == "__main__":
    main()
