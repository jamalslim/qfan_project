# QFAN, the Quantum Feature Amplification Network

QFAN is an autoregressive quantum generative model for calorimeter showers.
This repository trains the model, generates showers, and reproduces every
figure and number in the paper, for the 12- and 25-pixel CLIC images and for
the 368 voxels of CaloChallenge Dataset 1.

Everything runs on a laptop with NumPy and SciPy. Training d=12 takes a few
minutes, d=25 a couple of hours in resumable chunks, and the plots take
seconds.

## What the model does

A calorimeter shower image has d pixels. Rather than giving the circuit one
qubit per pixel, QFAN splits the image into consecutive blocks of b pixels and
generates them one block at a time, reusing the *same* small circuit at every
step (three qubits for the CLIC images, four for Dataset 1). Each step is conditioned on a fixed-length sketch summarizing the
pixels generated so far, so the circuit input never grows with the image.
Adding pixels adds autoregressive steps, not qubits.

The circuit is used as a sampler, not as a feature extractor. It is executed k
times per block and the block is decoded from the resulting k-shot average of
Born measurement records. That finite-k average *is* the model output, so the
sampling fluctuation is the only source of randomness in a generated shower.
A trained fraction of the records is shared between the pixels of a block,
which is how measurement noise is shaped into physical inter-pixel correlation
rather than suppressed.

Training minimizes a characteristic-function maximum mean discrepancy for
which exact analytic gradients are available, so there is no bilevel
optimization, no stochastic-perturbation estimator, and no autodiff framework.
Gradients are hand-derived and checked against finite differences in the test
suite.

## Setup

Python 3.12 is the tested version.

```bash
pip install -r requirements.txt
```

`numpy`, `scipy`, `scikit-learn` and `matplotlib` are required. `mplhep` is
also listed: the figures use `hep.style.ROOT` when it is present, and fall
back to a built-in imitation when it is not. The figures in the paper were
made with `mplhep` installed.

Check the data is present:

```bash
ls data/
#  cal_shower_img_12q.npy   cal_shower_img_25q.npy
```

Both are CLIC electromagnetic shower images downsampled to 12 and 25 pixels,
47682 events each, split 4800 train and 1200 test with a fixed seed. If one is
missing, training at that size will fail.

## The short version

```bash
python scripts/train.py --d 12 --loop
python scripts/train.py --d 25 --loop --budget-s 900
python scripts/evaluate.py
python scripts/plot_paper_figures.py --d 12
python scripts/plot_paper_figures.py --d 25
python scripts/paper_numbers.py \
    --files outputs/model_d12.npz \
            outputs/model_d25.npz \
    --latex tables/
python tests/test_born.py
```

The rest of this file explains each step and where it can go wrong.

## Training

```bash
python scripts/train.py --d 12 --loop
python scripts/train.py --d 25 --loop
```

Same algorithm at both sizes. Only the circuit settings differ:

|      | depth | angle dim | theta | blocks |
|------|-------|-----------|-------|--------|
| d=12 | 2     | 8         | 12    | 6      |
| d=25 | 3     | 16        | 18    | 13     |

Shared by both: three qubits, block size b=2, 64 records per block, three
tensor-product measurement settings giving 18 features, 300 epochs, and one
trained record share per block.

Training runs in chunks. Each chunk trains until `--budget-s` expires and
checkpoints. With `--loop` the script starts the next chunk itself and keeps
going until it prints `TRAINING COMPLETE`, so one command is enough. Without
`--loop` it stops after one chunk and prints `CHUNK DONE`, and you re-run the
same command until it completes.

The default budget is 600 s per chunk. On a slower machine a larger value
means fewer checkpoint round-trips, for instance `--budget-s 900`.
Progress lives in `outputs/ckpt_<tag>.pkl`, and deleting that file restarts
from scratch.

Useful flags:

```bash
python scripts/train.py --d 25 --budget-s 600        # longer chunks
python scripts/train.py --d 25 --epochs 150          # shorter run
python scripts/train.py --d 25 --loop --max-hours 4  # ceiling on --loop
```

`--epochs` is fixed at the start of a run and stored in the checkpoint, so it
cannot be changed part way through without deleting the checkpoint.

Two things to know before you start:

- **Checkpoints are tied to an image size.** A checkpoint holding a different
  block count than the run expects is ignored, and training starts fresh with
  a message. Without that guard, resuming across image sizes fails deep inside
  generation with an unhelpful shape error.
- **Training overwrites the results file without asking.** Copy anything you
  care about out of `outputs/` first. Running `--epochs 3` to check something
  works will happily replace a finished 300-epoch run.

Training writes:

```
outputs/model_d12.npz          d=12 model, samples, ablations
outputs/loss_d12.npz       loss history
outputs/model_d25.npz      d=25 equivalents
outputs/loss_d25.npz
```

Each results file holds more than the trained parameters. It also carries the
generated samples and every ablation variant used in the paper:

| array | what it is |
|---|---|
| `Y_te` | held-out MC truth |
| `Y_gen` | the trained model |
| `Y_ini` | circuit frozen at its untrained initialization, classical stages refitted |
| `Y_mean` | sampled records replaced by their conditional expectations |
| `Y_cls` | quantum feature map replaced by a classical one of equal width |
| `Y_scr` | sketch scrambled, destroying the prefix conditioning |
| `Y_cop` | the calibrated variant |

Those are what the paper's component-removal table compares.

## Evaluation

```bash
python scripts/evaluate.py
```

Generates from the trained d=25 model with no post-processing of any kind: no
correlation calibration, no monotone marginal map, no copula. It reports
`off=0.1068` and writes `outputs/samples_d25.npz`, which the
d=25 plots consume.

**That number is not the one in the paper's Table I.** The table reports
0.1282 for d=25, which comes straight from training using native features and
k=64. This script uses quadratic record features and k=32, a different
configuration. Both are legitimate results; they are simply not comparable to
each other.

## Plotting

```bash
python scripts/plot_paper_figures.py --d 12    # -> plots_d12/
python scripts/plot_paper_figures.py --d 25    # -> plots_d25/
```

Each run produces, as both PDF and PNG: the Pearson correlation matrices for
MC truth, the trained model and the untrained reference; the per-pixel
marginal intensity distributions with model-over-MC ratio panels; and the
total deposited energy spectrum.

One asymmetry to be aware of. `--d 12` reads the training artifact directly,
so it only needs step 1. `--d 25` reads the raw evaluation output, so it needs
`evaluate.py` to have run since the last training.

If `mplhep` is installed the figures use the genuine ROOT style; otherwise a
built-in fallback that looks close. Either is fine.

## Numbers and tables

```bash
python scripts/paper_numbers.py \
    --files outputs/model_d12.npz \
            outputs/model_d25.npz \
    --latex tables/
```

Recomputes every metric in the paper from the artifacts: per-pixel Wasserstein
distances, correlation errors split into within-block and cross-block,
Spearman rank errors, total-energy statistics, and bootstrap confidence
intervals. Writes `tab_certificates.tex` and `tab_w1_pixels.tex` ready to
paste.

**Pass `--files` explicitly.** With `--outputs` it scans the directory and
chooses one artifact per image size, and an older variant can win. The choice
is printed and there is a `--prefer` flag, but naming the files removes the
ambiguity.

Two behaviours worth understanding:

- Rows where the generator has collapsed print `DEGENERATE` instead of a
  correlation number. That is the conditional-means ablation: with the
  sampling removed the rollout is deterministic, every generated sample is
  identical, and the correlation of a constant is undefined. Any finite number
  quoted for it is an artifact of how the singularity was filled.
- Bootstrap intervals default to 200 resamples. Use `--boot 500` for final
  numbers and `--boot 0` while iterating, since resampling dominates runtime.

## Other image sizes

`--data` replaces `--d` and trains on any `(N, d)` array of non-negative
values. Outputs are named from the data file unless `--tag` is given.

```bash
python scripts/train.py --data data/ds1_53.npy --nq 4 --block-size 2 --depth 3 \
       --decoder-intercept --seed 1 --loop --budget-s 900
```

| option | meaning |
|---|---|
| `--nq` | register width |
| `--block-size` | pixels per autoregressive step |
| `--depth`, `--angle-dim` | circuit depth and width of the sketch encoding |
| `--seed` | train/test split, initialization and sketch together; outputs get `_s<seed>` |
| `--decoder-intercept` | per-pixel decoder intercept. The shot-noise-aware ridge shrinks predictions toward zero; the intercept restores the training mean. Needed whenever a pixel's typical value is far from zero |
| `--tag` | name for the outputs |

**Keep blocks at two pixels.** The loss compares each block's joint
distribution through 1024 random frequencies. In two dimensions they resolve
the correlations; in ten they are spread so thinly that training finds almost
no structure. Every run with blocks of 4, 10 or 19 pixels learnt almost
nothing, and every run with 2-pixel blocks worked, so a larger image means a
longer chain, not larger blocks.

Training uses a factorised form of the loss that never enumerates the
2^(3 n_q) joint measurement records: identical to the original to about 1e-17
on every code path, and under 100 MB where the original would need 1.4 TB at
n_q = 10. One loss-and-gradient evaluation on a 64-sample batch takes roughly
3 s at n_q = 8, 17 s at n_q = 10 and 155 s at n_q = 12, dominated by the exact
parameter-shift gradients.

## CaloChallenge Dataset 1

`ds1/` applies the same model to the CaloChallenge Dataset 1 photons (ATLAS
geometry, 368 voxels in five layers) at 65 GeV, covering preparation,
training, generation and evaluation, with nothing fitted to generated output.
The preprocessing writes each shower as its total energy followed by the
energy fractions, so the circuit generates the energy spectrum itself and every
generated shower is consistent with its own total energy. `ds1/README.md` has
the commands and explains how to read the results.

The trained 368-voxel model of the paper and the showers behind its Dataset 1
table and figures are attached to the GitHub release of this repository, not
stored in it. Put them in `outputs/` and run
`python ds1/paper_figures.py outputs/model_ds1_368_nq4_b2_s1_showers.npz`.

## Register-width sweep

The width study of the paper's Sec. VIII trains the same pipeline at several
register widths:

```bash
python scripts/sweep_nq.py --verify --nq 2 3 4 5          # engine check, seconds
python scripts/sweep_nq.py --nq 2 3 4 5 --epochs 150 --budget-s 280   # repeat until complete
```

Results go to `outputs/sweep_nq<n>_result.json`; the paper's runs are included.

## Supplementary analyses

`supplementary/` holds the analyses behind the paper's comparisons and
ablations, run from the project root:

| script | what it measures |
|---|---|
| `eval_d25_classical_baselines.py` | QFAN against classical generative baselines at d=25, same data and protocol |
| `quantum_necessity_tests.py` | measurable bounds on where a quantum device becomes necessary |
| `plot_k_sweep.py` | records per block as a model parameter: fidelity against k |
| `eval_d25_raw_trainside_select.py` | re-selects the raw d=25 configuration on training data only, removing any test-set selection |

## Checking it works

```bash
python tests/test_born.py          # 15/15 checks passed
python tests/test_factorized.py    # two [PASS] lines
python tests/test_ds1.py           # data preparation and export self-tests
python tests/test_ds1.py --full    # plus the whole DS1 chain on synthetic showers
```

`test_born.py` covers the circuit engine against an independently constructed
dense unitary, every analytic gradient against central finite differences, the
noise-aware ridge correction and the record sharing. `test_factorized.py`
checks the factorised loss against the original on 36 code-path combinations
and on larger blocks. Run both first if you change anything in `src/qfan/`:
the gradients are hand-derived, so an error there does not raise an exception,
it just trains to a worse optimum.

## Reproducing a specific number

The paper's component-removal table comes entirely from the two training
artifacts, so any row can be checked directly:

```python
import numpy as np, sys
sys.path.insert(0, "src")
from qfan.utils import corr_nan_safe
from scipy.stats import wasserstein_distance

r = np.load("outputs/model_d12.npz", allow_pickle=True)
Y_te, Y = r["Y_te"], r["Y_gen"]        # or Y_ini, Y_cls, Y_scr, Y_mean, Y_cop
d = Y_te.shape[1]
off = ~np.eye(d, dtype=bool)

w1 = np.mean([wasserstein_distance(Y_te[:, j], Y[:, j]) for j in range(d)])
err = np.abs(corr_nan_safe(Y) - corr_nan_safe(Y_te))[off].mean()
print(w1, err)          # 0.00922  0.0925
```

## Layout

```
src/qfan/        the model: circuit engine, sketch, decoder, losses
scripts/         the paper: training, evaluation, figures, tables, width sweep, QASM export
ds1/             CaloChallenge Dataset 1 (see ds1/README.md)
supplementary/   analyses behind the paper's comparisons and ablations
tests/           correctness suites
data/            the 12- and 25-pixel CLIC images
outputs/         trained models, samples, metrics
plots_d12/       figures, d=12 (generated, not tracked)
plots_d25/       figures, d=25 (generated, not tracked)
tables/          generated LaTeX (generated, not tracked)
```

| script | role |
|---|---|
| `train.py` | training: `--d 12`, `--d 25`, or `--data` for any image |
| `plot_paper_figures.py` | figures, `--d 12` or `--d 25` |
| `evaluate.py` | d=25 evaluation, feeds the d=25 plots |
| `paper_numbers.py` | metrics and LaTeX tables |
| `sweep_nq.py` | register-width sweep |
| `export_qasm.py` | the trained circuits as OpenQASM 2.0, for hardware |
| `quad_features.py` | `ExtFeatures`, imported by the evaluation |
| `calibration.py` | imported by `quad_features` |

The last two are never run directly and look obsolete. They are not. The
evaluation imports through all of them:

```
evaluate -> quad_features -> calibration
                                          -> train
```

Delete any one of them and `evaluate.py` stops working, taking the
d=25 figures with it.

The core modules are `born.py`, which holds the circuit engine, the Born
record sampling, the noise-aware ridge fit and the CF-MMD gradients;
`sketch.py`, the streaming count-sketch; `ridge.py`, the closed-form decoder;
and `train.py`, the optimizer loop.

## If something goes wrong

**Training exits immediately saying complete.** A finished checkpoint or
results file already exists. Delete `outputs/ckpt_<tag>.pkl`.

**A shape error deep inside generation.** Almost always a checkpoint from the
other image size. The guard catches the common case; deleting the checkpoint
is the reliable fix.

**Plots are empty or stale.** The d=25 figures need `evaluate.py` to
have run since the last training.

**Numbers differ in the fourth decimal.** Every script is seeded, so
re-running training or evaluation with the same settings reproduces it exactly,
bit for bit. Differences come from changed settings: another `--seed`, another
number of epochs, or another package version.

**`ModuleNotFoundError: qfan`.** Every entry point walks up from its own
location looking for `src/qfan`, so `python scripts/train.py` works from
anywhere inside the tree. If it still fails, you are outside the project
directory.

## Citation

If you use this code, please cite the article, *Quantum Feature Amplification
Network (QFAN) as An Autoregressive Quantum Generative Model*, J. Slim,
S. Monaco, F. Rehm, D. Krücker and K. Borras, arXiv:2605.16044 (2026).
`CITATION.cff` has the details in machine-readable form.

## License

MIT, see `LICENSE`.
