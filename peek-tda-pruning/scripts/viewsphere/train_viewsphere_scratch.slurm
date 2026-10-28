#!/bin/bash
# Train YOLO26 from scratch on viewsphere v1_wsd4, same settings as WSD's
# slurm/train_yolo26{n,s}_scratch.slurm (1000 ep, patience 100, batch 16, seed 42, deterministic).
# Usage: sbatch --job-name=y26n-vs4 --export=ALL,MODEL=yolo26n slurm/train_viewsphere_scratch.slurm
#SBATCH --partition=gpu1,gpu2
#SBATCH --exclude=gpu06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared/projects/tina/PEEK-TiNA/logs/%x-%j.out
#SBATCH --error=/shared/projects/tina/PEEK-TiNA/logs/%x-%j.err

set -euo pipefail

REPO=/shared/projects/tina/PEEK-TiNA
DATA=/shared/projects/nets/viewsphere-dataset/v1_wsd4/data.yaml
: "${MODEL:?set MODEL=yolo26n or yolo26s}"

cd "$REPO"
source "$REPO/tina_venv/bin/activate"

export WANDB_MODE=disabled

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi

srun yolo detect train \
    model="$MODEL.yaml" \
    data="$DATA" \
    epochs=1000 \
    patience=100 \
    imgsz=640 \
    batch=16 \
    workers=8 \
    device=0 \
    seed=42 \
    deterministic=True \
    project="$REPO/runs/viewsphere_wsd4_scratch" \
    name="${MODEL}_scratch_seed42" \
    exist_ok=False

echo "Finished: $(date)"
