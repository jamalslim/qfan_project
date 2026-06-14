# QFAN  — The code companion for the Quantum Feature Amplification Network Paper





> Slim, Monaco, Rehm, Krücker, Borras.
> *Quantum Feature Amplification Network (QFAN) as An Autoregressive Quantum Generative Model.*



training loop featuring:

  - analytic chain rule through the closed-form ridge `W(θ)`;
  - multi-kernel fixed-bandwidth MMD² (stationary loss);
  - Adam optimizer with linear warmup + cosine decay;
  - optional Gauss–Newton preconditioning via the feature Jacobian;
  - Polyak–Ruppert averaging of the θ iterate;
  - held-out validation loss monitor, evaluated at θ every step;
  - EMA-smoothed val-loss tracking for readable convergence plots.


## Layout

```
qfan_project/
├── README.md              this file
├── src/qfan/              importable library
│   ├── __init__.py
│   ├── config.py          dataclasses + d12 / d25 presets
│   ├── utils.py           numerical helpers
│   ├── data.py            dataset loading + subset selection
│   ├── sketch.py          count-sketch (paper Sec. III-B)
│   ├── mmd.py             multi-kernel MMD² + analytic gradient
│   ├── quantum.py         Pauli grouping + shot bank + parameter-shift
│   ├── ridge.py           closed-form ridge decoder
│   ├── train.py           EXACT-gradient training loop (core novelty)
│   ├── decoder.py         classical residual gate (paper Sec. III-F)
│   ├── pipeline.py        fit, sample, predict, rollout refine
│   ├── calibration.py     Gaussian-copula post-hoc correction
│   └── metrics.py         correlation-error summaries
├── scripts/               executable entry points
│   ├── _pipeline.py       end-to-end driver (shared by run_*)
│   ├── run_d12.py         paper Sec. VII main experiment
│   ├── run_d25.py         paper Appendix A extension
│   └── plot_results.py    paper-style figures from saved outputs
├── data/                  datasets (cal_shower_img_12q.npy included)
├── outputs/               generated .npz results (created on run)
└── plots/                 generated figures (created on run)
```

## Install

## Tested with python 3.12 

Python ≥ 3.9. Dependencies: 

```
numpy scipy scikit-learn matplotlib qiskit>=1.0 qiskit-aer
```

```bash
pip install numpy scipy scikit-learn matplotlib qiskit qiskit-aer
```

No setuptools package; scripts add `src/` to `sys.path` at runtime.

## Run

From the project root:

```bash
# paper's d=12 main experiment (Sec. VII)
python scripts/run_d12.py

# paper's d=25 Appendix A extension (requires data/cal_shower_img_25q.npy)
python scripts/run_d25.py

# generate plots from a saved run
python scripts/plot_results.py d12
```

## Ablations and switches

All hyperparameters are on the config dataclasses in `src/qfan/config.py`.
The most useful knobs:

  - **`cfg.train.mode`** — `"exact"` (this work) or `"spsa"` (paper baseline).
    The SPSA path uses the same MMD definition and validation monitor so the
    two val_loss curves are directly comparable.
  - **`cfg.train.use_precond`** — toggle the Gauss–Newton preconditioner.
    On small well-conditioned problems plain Adam can be faster; on real
    shot-noise-limited runs the preconditioner usually wins.
  - **`cfg.train.lr_decay`** — `"cosine"` or `"none"`.
  - **`cfg.train.pr_tail_frac`** — how much of the tail is Polyak–Ruppert-averaged.
  - **`cfg.bank.q_depth`** — encoder depth; `2` is paper-faithful for d=12,
    `3` for d=25 Appendix A.
  - **`cfg.copula.enabled`** — turn off the post-hoc calibrator to see the
    raw model's joint correlations.

## Reading the training output

Each epoch prints:

```
epoch   N/T  train=...  val=...  ema=...  ||g||=...  ||step||=...  lr=...
```

The relevant one for "is training working" is **`val`**: it is the
**sum of per-block multi-kernel MMD²** computed at the current θ on a
**held-fixed validation batch**, using the **bandwidth family fixed at
init**. This is a stationary objective and should descend monotonically
past the warmup phase. The `ema` column is the β=0.9 exponential moving
average of `val`, shown for readability; it does not affect training.

At the end of the run the script also prints:

```
[DONE] val_loss (PR-avg final): X.XXXXXX
```

This is the val loss evaluated at the Polyak–Ruppert-averaged θ, which
is the θ actually deployed for generation.

## Reference scales (d=12 CLIC dataset)

Numbers below are computed directly on the included `cal_shower_img_12q.npy`.

| Scenario | val_loss |
|---|---|
| Matched true samples (MMD estimator noise floor) | ~0.016 |
| Correct marginals only, correlations destroyed | ~0.28 |
| Diagonal Gaussian matched to MC mean/std | ~0.30 |
| Constant mean-only prediction (random-init ceiling) | ~1.77 |

An `val_loss` around **0.05–0.10** after rollout refinement is in the
paper's simulator territory (their reported MMD² ≈ 0.008 per block per
kernel, summed B×|Σ| gives ≈0.05). Below **~0.05** surpasses the paper
at this scale.

## Verified numerical properties

These are checked in the test harness (not shipped as unit tests, but
reproducible from the code):

  - Analytic `dL/dŶ` for multi-kernel MMD² matches central finite difference
    to **relative error ≈ 2 × 10⁻⁹** on a random (40, 3) test problem.
  - Full analytic `dL/dθ` chain rule matches central FD to
    **relative error ≈ 4 × 10⁻⁸** on a synthetic feature map with p_θ = 8.
  - On a deterministic classical stand-in for `F(θ, S)`, exact-gradient
    Adam drops val MMD² by **28.5%** in 30 epochs while SPSA at **4×** the
    gradient budget drops it by **0.07%** (essentially flat).

## References

Main references (full list in `src/qfan/__init__.py` and `train.py`):

  - Schuld, Bergholm, Gogolin, Izaac, Killoran. Phys. Rev. A 99, 032331 (2019).
    [arXiv:1811.11184] — parameter-shift rule.
  - Mitarai, Negoro, Kitagawa, Fujii. Phys. Rev. A 98, 032309 (2018). — QCL.
  - Stokes, Izaac, Killoran, Carleo. Quantum 4, 269 (2020). — QNG.
  - Gacon, Zoufal, Carleo, Woerner. Quantum 5, 567 (2021). — QN-SPSA.
  - Li, Chang, Cheng, Yang, Póczos. NeurIPS 2017. — MMD GAN.
  - Polyak, Juditsky. SIAM J. Control Optim. 30, 838 (1992). — PR averaging.
  - Kingma, Ba. ICLR 2015. [arXiv:1412.6980] — Adam.
  - Spall. IEEE Trans. Autom. Control 37, 332 (1992). — SPSA.
  - Charikar, Chen, Farach-Colton. ICALP 2002. — count-sketch.
  - Gretton et al. JMLR 13, 723 (2012). — MMD.
