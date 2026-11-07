#!/bin/bash
# Fine-tune the WSD scratch models (runs/wsd71_scratch/<MODEL>_scratch_seed42/weights/best.pt) on
# viewsphere v1_wsd4. Same schedule as slurm/train_viewsphere_scratch.slurm (1000 ep, patience 100,
# batch 16, seed 42, deterministic) but with the project fine-tune optimizer: AdamW lr 1.25e-3,
# momentum 0.9, matched warmup bias lr (= lr0).
# Usage: sbatch --job-name=26n-vs4ft --export=ALL,MODEL=yolo26n slurm/train_viewsphere_finetune.slurm
# DATA / PROJECT (added 2026-11-03) train on another split, e.g. v2_wsd4 into runs/viewsphere2_wsd4_finetune_wsd.
# NOISE (added 2026-11-06): max sigma in gray levels of the training-only pixel noise (see below).
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
DATA=${DATA:-/shared/projects/nets/viewsphere-dataset/v1_wsd4/data.yaml}
PROJECT=${PROJECT:-$REPO/runs/viewsphere_wsd4_finetune_wsd}
: "${MODEL:?set MODEL=yolo26n or yolo26s}"
# INIT (added 2026-11-06): other starting weights, e.g. COCO yolo26n.pt (submit_coco_pilot.sh)
INIT="${INIT:-$REPO/runs/wsd71_scratch/${MODEL}_scratch_seed42/weights/best.pt}"

cd "$REPO"
source "$REPO/tina_venv/bin/activate"

export WANDB_MODE=disabled

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Init: $INIT ($(md5sum "$INIT" | cut -d' ' -f1))"
echo "Started: $(date)"
nvidia-smi

# NOISE (added 2026-11-06): train with pixel-noise augmentation (train_noise_aug.py; v2 renders are ~96%
# exactly-black pixels and models trained on them are hypersensitive to background noise)
if [ -n "${NOISE:-}" ]; then TRAIN=(python "$REPO/tina/scripts/viewsphere/train_noise_aug.py" "$NOISE"); else TRAIN=(yolo detect train); fi
echo "Train: ${TRAIN[*]}"

srun "${TRAIN[@]}" \
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
    project="$PROJECT" \
    name="${MODEL}_ft_seed42" \
    exist_ok=False

echo "Finished: $(date)"
