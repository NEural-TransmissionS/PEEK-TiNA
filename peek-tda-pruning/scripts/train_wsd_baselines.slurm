#!/bin/bash
# Scratch YOLO26 baselines on WSD -- the checkpoints both experiments start from.
#   sbatch --export=ALL,SIZE=n,WSD_DATA=/path/to/data.yaml,OUT=/path/to/runs scripts/train_wsd_baselines.slurm
#   sbatch --export=ALL,SIZE=s,...
# ARCH=yolov5 trains the YOLOv5u models instead (same recipe; added 2026-11-02).
# -> $OUT/${ARCH}${SIZE}_scratch_seed42/weights/best.pt
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
ARCH=${ARCH:-yolo26}
yolo detect train model="${ARCH}${SIZE}.yaml" data="$WSD_DATA" \
  epochs=1000 patience=100 imgsz=640 batch=16 workers=8 device=0 \
  seed=42 deterministic=True project="$OUT" name="${ARCH}${SIZE}_scratch_seed42" exist_ok=False
