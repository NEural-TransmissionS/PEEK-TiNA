#!/usr/bin/env bash
# Run from INSIDE the package dir:
#   cd block_pruning && WSD_ROOT=/path/to/satellite_components-71 \
#     WSD_WEIGHTS_N=/path/to/yolo26n_best.pt WSD_WEIGHTS_S=/path/to/yolo26s_best.pt \
#     VENV=/path/to/venv bash slurm/submit_wsd.sh
# WSD_ROOT = the Roboflow export (train/ valid/ test/, each images/ + labels/);
# WSD_WEIGHTS_{N,S} = scratch baselines from ../scripts/train_wsd_baselines.slurm.
# Links them into the shapes layout, then chains
# rank (GPU, once per size) -> the 20-task wsd.slurm array -> summarize.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm/logs data/wsd/images
: "${WSD_ROOT:?set WSD_ROOT}" "${WSD_WEIGHTS_N:?set WSD_WEIGHTS_N}" "${WSD_WEIGHTS_S:?set WSD_WEIGHTS_S}"
VENV="${VENV:-.venv}"; export VENV

cat > data/wsd/data.yaml <<YAML
path: $WSD_ROOT
train: train/images
val: valid/images
nc: 4
names: [antenna, body, solar, thruster]
YAML
ln -sfn "$WSD_ROOT/valid/images" data/wsd/images/val
ln -sfn "$WSD_ROOT/train/images" data/wsd/images/train  # ranking reads train only
ln -sfn "$WSD_WEIGHTS_N" weights/wsd_n.pt
ln -sfn "$WSD_WEIGHTS_S" weights/wsd_s.pt

RANK=$(sbatch --parsable --job-name=tp-wsd-rank --partition=gpu2 --gres=gpu:1 --time=00:30:00 \
  --mem=16G --output=slurm/logs/wsd_rank.out \
  --wrap=". $VENV/bin/activate; for s in n s; do python run_experiment.py --dataset wsd --size \$s --ks --rank-split train; done")
echo "rank      $RANK"
SWEEP=$(sbatch --parsable --dependency=afterok:"$RANK" slurm/wsd.slurm)
echo "sweep     $SWEEP"
SUM=$(sbatch --parsable --dependency=afterany:"$SWEEP" --job-name=tp-sum --time=00:20:00 \
  --mem=8G --output=slurm/logs/summarize_wsd.out \
  --wrap=". $VENV/bin/activate; python summarize.py")
echo "summarize $SUM"
