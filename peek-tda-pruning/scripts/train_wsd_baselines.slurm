#!/bin/bash
# Scratch YOLO26 baselines on WSD -- the checkpoints both experiments start from.
#   sbatch --export=ALL,SIZE=n,WSD_DATA=/path/to/data.yaml,OUT=/path/to/runs scripts/train_wsd_baselines.slurm
#   sbatch --export=ALL,SIZE=s,...
# -> $OUT/yolo26${SIZE}_scratch_seed42/weights/best.pt
#SBATCH --job-name=y26-wsd-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
set -euo pipefail
: "${SIZE:?n or s}" "${WSD_DATA:?path to WSD data.yaml}" "${OUT:?output dir}"
source "${VENV:-.venv}/bin/activate"
export WANDB_MODE=disabled
yolo detect train model="yolo26${SIZE}.yaml" data="$WSD_DATA" \
  epochs=1000 patience=100 imgsz=640 batch=16 workers=8 device=0 \
  seed=42 deterministic=True project="$OUT" name="yolo26${SIZE}_scratch_seed42" exist_ok=False
