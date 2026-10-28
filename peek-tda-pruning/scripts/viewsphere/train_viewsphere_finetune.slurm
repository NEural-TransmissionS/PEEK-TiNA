#!/bin/bash
# Fine-tune the WSD scratch models (runs/wsd71_scratch/<MODEL>_scratch_seed42/weights/best.pt) on
# viewsphere v1_wsd4. Same schedule as slurm/train_viewsphere_scratch.slurm (1000 ep, patience 100,
# batch 16, seed 42, deterministic) but with the project fine-tune optimizer: AdamW lr 1.25e-3,
# momentum 0.9, matched warmup bias lr (= lr0).
# Usage: sbatch --job-name=26n-vs4ft --export=ALL,MODEL=yolo26n slurm/train_viewsphere_finetune.slurm
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
INIT="$REPO/runs/wsd71_scratch/${MODEL}_scratch_seed42/weights/best.pt"

cd "$REPO"
source "$REPO/tina_venv/bin/activate"

export WANDB_MODE=disabled

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Init: $INIT ($(md5sum "$INIT" | cut -d' ' -f1))"
echo "Started: $(date)"
nvidia-smi

srun yolo detect train \
    model="$INIT" \
    data="$DATA" \
    epochs=1000 \
    patience=100 \
    imgsz=640 \
    batch=16 \
    workers=8 \
    device=0 \
    seed=42 \
    deterministic=True \
    optimizer=AdamW \
    lr0=0.00125 \
    momentum=0.9 \
    warmup_bias_lr=0.00125 \
    project="$REPO/runs/viewsphere_wsd4_finetune_wsd" \
    name="${MODEL}_ft_seed42" \
    exist_ok=False

echo "Finished: $(date)"
