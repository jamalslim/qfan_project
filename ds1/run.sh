#!/usr/bin/env bash
# CaloChallenge Dataset 1: prepare, train, generate, evaluate and plot.
#
# The defaults reproduce the paper: all 368 voxels, training seed 1, a fixed
# 64 measurement records per block during training, and the records per block
# fitted after training (generate.py --fit-k).
#
#   bash ds1/run.sh <path to dataset_1_photons_1.hdf5>
#
# Options, as environment variables:
#   SEEDS="1 2 3"   more training seeds (default 1)
#   COLLAPSE=angle  the 53-ring version instead of all 368 voxels
#   LEARN_K=1       learn the records per block during training instead of
#                   fitting them afterwards (output names get _lk)
#   POWER=0.3       prepare the fractions as fraction^0.3
#   EPOCHS=N        number of training epochs (default 300)
#
# Training resumes from its checkpoint, so if it stops (for example at the
# time limit) run the same command again. One seed at 368 voxels takes about
# eight hours on a single CPU, and fitting the records per block about half an
# hour more.
set -euo pipefail
HDF5="${1:?usage: bash ds1/run.sh <path to dataset_1_photons_1.hdf5>}"
SEEDS="${SEEDS:-1}"
COLLAPSE="${COLLAPSE:-none}"
LEARN_K="${LEARN_K:-0}"
POWER="${POWER:-1}"
EPOCHS="${EPOCHS:-}"

if [ "$COLLAPSE" = "none" ]; then NAME="ds1_368"; ORDER="--order ring"; else NAME="ds1_53"; ORDER=""; fi
if [ "$POWER" != "1" ]; then NAME="${NAME}_p${POWER}"; fi
DATA="data/${NAME}.npy"
if [ "$LEARN_K" = "1" ]; then LK="_lk"; TRAIN_K="--learn-k"; GEN_K=""; else LK=""; TRAIN_K=""; GEN_K="--fit-k"; fi
EP=""; if [ -n "$EPOCHS" ]; then EP="--epochs $EPOCHS"; fi
mkdir -p data outputs

[ -f "$DATA" ] || python ds1/prepare_ds1.py --src "$HDF5" --energy 65536 \
    --collapse "$COLLAPSE" $ORDER --fractions --power "$POWER" --out "$DATA"

FILES=""
for s in $SEEDS; do
  M="outputs/model_${NAME}_nq4_b2${LK}_s${s}"
  echo "=== seed ${s} ==="
  [ -f "${M}.npz" ] || python scripts/train.py --data "$DATA" --nq 4 --block-size 2 \
      --depth 3 --decoder-intercept $TRAIN_K $EP --seed "$s" --loop --budget-s 900 --max-hours 72
  if [ ! -f "${M}.npz" ]; then
    echo "training of seed ${s} is not finished (checkpoint saved); run this command again to resume"
    exit 1
  fi
  [ -f "${M}_showers.npz" ] || python ds1/generate.py "${M}.npz" --data "$DATA" $GEN_K
  FILES="$FILES ${M}_showers.npz"
done

rm -f outputs/results.md
python ds1/aggregate.py $FILES --name "QFAN, ${NAME}" --table outputs/results.md
FIRST="outputs/model_${NAME}_nq4_b2${LK}_s$(echo $SEEDS | cut -d' ' -f1)_showers.npz"
if [ "$COLLAPSE" = "none" ]; then
  python ds1/paper_figures.py "$FIRST" --data "$DATA" --out figures
else
  python ds1/eval_geometry.py "$FIRST" --data "$DATA"
fi
echo "table: outputs/results.md"
