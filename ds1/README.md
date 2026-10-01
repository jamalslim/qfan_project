# QFAN on CaloChallenge Dataset 1

GEANT4 photon showers in the ATLAS calorimeter geometry, public on Zenodo
(DOI 10.5281/zenodo.8099322): 368 voxels in five layers. Everything here uses
the 10,000 showers at 65 GeV in `dataset_1_photons_1.hdf5`.

| layer | angular x radial bins | voxels |
|---|---|---|
| 0 | 1 x 8 | 8 |
| 1 | 10 x 16 | 160 |
| 2 | 10 x 19 | 190 |
| 3 | 1 x 5 | 5 |
| 12 | 1 x 5 | 5 |

## Reproducing the paper

All 368 voxels, seed 1. Download `dataset_1_photons_1.hdf5` from Zenodo
(doi 10.5281/zenodo.8099322), then

    # 1. prepare (once)
    python ds1/prepare_ds1.py --src dataset_1_photons_1.hdf5 --energy 65536 --collapse none --order ring --fractions --out data/ds1_368.npy
    # 2. train (about 8 hours, resumable, run the same command again if it stops)
    python scripts/train.py --data data/ds1_368.npy --nq 4 --block-size 2 --depth 3 --decoder-intercept --seed 1 --loop --budget-s 900 --max-hours 72
    # 3. generate, with the records per block fitted after training
    python ds1/generate.py outputs/model_ds1_368_nq4_b2_s1.npz --data data/ds1_368.npy --fit-k
    # 4. results table and figures
    python ds1/aggregate.py outputs/model_ds1_368_nq4_b2_s1_showers.npz --name "QFAN, 368 voxels" --table outputs/results.md
    python ds1/paper_figures.py outputs/model_ds1_368_nq4_b2_s1_showers.npz --data data/ds1_368.npy --out figures

`bash ds1/run.sh dataset_1_photons_1.hdf5` runs the same steps. To skip
training, download the trained model and its showers from the GitHub release,
put them in `outputs/`, and run step 4.

## Pipeline

**prepare, train, generate, evaluate.** Nothing else touches the showers: no
calibration, and nothing fitted to generated output.

```bash
bash ds1/run.sh ~/data/calochallenge/dataset_1_photons_1.hdf5
```

or step by step, for one seed:

```bash
H5=~/data/calochallenge/dataset_1_photons_1.hdf5
python ds1/prepare_ds1.py --src $H5 --check                                   # voxel order: must print [PASS]
python ds1/prepare_ds1.py --src $H5 --energy 65536 --collapse angle --fractions --out data/ds1_53.npy
python scripts/train.py --data data/ds1_53.npy --nq 4 --block-size 2 --depth 3 \
       --decoder-intercept --learn-k --seed 1 --loop --budget-s 900
python ds1/generate.py outputs/model_ds1_53_nq4_b2_lk_s1.npz --data data/ds1_53.npy
python ds1/aggregate.py outputs/model_ds1_53_nq4_b2_lk_s1_showers.npz --name "QFAN"
python ds1/eval_geometry.py outputs/model_ds1_53_nq4_b2_lk_s1_showers.npz --data data/ds1_53.npy
python ds1/plot_polar.py    outputs/model_ds1_53_nq4_b2_lk_s1_showers.npz --data data/ds1_53.npy
```

`run.sh` does the same for seeds 1, 2 and 3 and writes `outputs/results.md` with
mean and spread across seeds. `SEEDS="1 2 3 4 5"` sets the seeds; `COLLAPSE=none`
models all 368 voxels instead of the 53 rings, with a longer chain and much
longer training. It is restartable: finished steps are skipped.

## Preprocessing

The only transformation besides QFAN itself. Deterministic, fixed before
training, and inverted exactly after generation.

1. **Units.** Each voxel is divided by the incident energy, and every value is
   multiplied by one global constant.
2. **Rings.** With `--collapse angle`, the ten angular bins of each ring in
   layers 1 and 2 are summed: one value per ring, 53 in all. Showers are close to
   azimuthally symmetric, and this is where their correlation structure lives.
   `--collapse none` keeps all 368 voxels.
3. **Total and fractions** (`--fractions`). Each shower is written as its total
   deposited energy followed by the fraction of that energy in every pixel. QFAN
   generates the total as its first value, so the energy spectrum is learnt by
   the circuit, and the fractions after it, each conditioned on everything
   before. After generation the fractions are normalized and multiplied by the
   generated total, so every generated shower is consistent with its own total energy. Without this step, a chain
   of 27 blocks drawing fresh shot noise at each one cannot hold the total to
   the 1.3% that GEANT4 does.

## Compressing the tails

The Born noise of a ridge-decoded value cannot exceed half its mean (the
dispersion cap). Measured on the 368-voxel training showers, 255 of the 368
energy fractions have a larger relative spread than that, so a model trained on
fractions cannot generate their tails. With `--power 0.3` each fraction is
prepared as fraction^0.3 instead: only 86 voxels then exceed the cap, largely
those that are often exactly zero, which a power leaves at zero. The inverse
raises the generated values back to the power 1/0.3 before normalising, so the
map stays exact.

    POWER=0.3 COLLAPSE=none SEEDS="1" bash ds1/run.sh <hdf5>

## The decoder

Each block is decoded from its measurement records by the closed-form ridge
regression of QFAN (arXiv:2605.16044), fitted on training showers under their
true history; the intercept restores each pixel's training mean. Nothing else
is fitted, and nothing is fitted to generated output. The untrained circuit,
decoded the same way, reproduces neither the correlations nor the energy
spectrum nor the voxel distributions, so what the model reproduces is the
circuit's work. The generated distributions are narrower than GEANT4's: the
ridge is the best predictor of each value's conditional mean and minimizes the
shot noise of its own output, and shot noise is the model's only source of
randomness where the data's spread is not predictable from the history.

## Learned records per block

With `--learn-k` (used by `run.sh`) each block learns its own number of
measurement records k_b, trained by the same loss as the circuit, like the
record share of each block. At generation each block averages round(k_b)
records, and its decoder is fitted for that count: the randomness stays purely
Born, and the model gains one number per block. Blocks need very different
noise levels: the total energy, generated first with no history, and the sparse
outer rings need more Born noise; blocks whose fluctuation follows from the
history need less. With one fixed k for all blocks the circuit cannot serve
both. On the 53 rings, at equal training, it raised the total-energy spread
from 0.35 to 0.60 of GEANT4's and the captured correlation structure from 36%
to 48%, while the untrained circuit, given the same learned budgets, still
produced an almost deterministic shower (energy spread 0.03, no correlations).

For models already trained with a fixed number of records, `generate.py
--fit-k` chooses each block's count after training instead: the count that
minimizes the training loss on training showers, with the circuit fixed. The
trained and the untrained circuit each get their own. On the 53 rings it gave
nearly the same gain as learning the counts during training (captured
structure 44% against 49%, energy spread 0.67 against 0.60), while the
untrained circuit with its own fitted counts still produced an almost
deterministic shower (energy spread 0.04, no correlations). The fitted counts
are more extreme than the learned ones, some blocks near 1,000 records and a
few near 2, which costs more measurement shots per shower.

## What the results show

`aggregate.py` reports each metric for QFAN next to three references on the
same test showers: the statistical floor (training showers against test
showers), the **untrained circuit** with its decoders fitted the same way, and a
**no-correlation model** (test showers with every pixel shuffled independently,
sent through the same inverse preprocessing).

| column | meaning |
|---|---|
| Pearson, Spearman | error of the full correlation matrix: **the main evidence**. Compare QFAN with the untrained circuit and the no-correlation model |
| captured | share of the achievable Spearman improvement, (null - model) / (null - floor) |
| W1 dense, W1 sparse, zeros | per-pixel distributions, straight from the circuit |
| E spread | spread of the total energy, generated by the circuit |
| L1-L2 | correlation between the energies of layers 1 and 2. Requiring each shower's voxels to sum to its total produces part of it for any model; the no-correlation row shows how much |

## Figures of the Dataset 1 paper

One script makes every figure, each plot a separate figure (PDF and PNG), in the
style of the first QFAN paper, with statistical error bars:

    python ds1/paper_figures.py outputs/model_ds1_368_nq4_b2_s1_showers.npz --data data/ds1_368.npy

    figures/correlation/  MC data, QFAN, QFAN untrained
    figures/energy/       total and per-layer energy, for QFAN and for the untrained
                          circuit: spectrum on top, ratio to MC data underneath
    figures/voxels/       single voxels, the same two figures each
    figures/polar/        average and single showers per layer: MC data, QFAN, untrained

`--voxels all` plots all 368 voxels (default: the first angular bin of every
ring); `--data` gives the polar plots in MeV.

## Files

| script | role |
|---|---|
| `prepare_ds1.py` | HDF5 to prepared data, and the exact inverse of `--fractions` |
| `generate.py` | showers from a trained model, with the untrained circuit and the no-correlation model |
| `aggregate.py` | metrics against the controls, with error bars across seeds |
| `run.sh` | the whole pipeline over several seeds |
| `eval_geometry.py` | per-layer tables; correlation, energy and marginal figures |
| `plot_polar.py` | showers drawn in the detector geometry, labeled with what is drawn |
| `sign_check.py` | layer-to-layer correlation signs, model against data |
| `export_calochallenge.py` | showers in the official CaloChallenge HDF5 format |
| `paper_figures.py` | the figures of the Dataset 1 paper |

## Notes

- **Keep blocks at two pixels.** Blocks of 4, 10 or 19 pixels learnt almost
  nothing: in ten dimensions the loss's random frequencies are too sparse to see
  the correlations.
- **Keep the shot-noise-aware decoder.** A plain ridge decoder made the
  total-energy spread 44 times too wide.
- **One global scale, no cap.** Scaling each pixel separately amplified the
  shot noise of the sparsest pixels; capping values at the scale target trimmed
  the shower core.
