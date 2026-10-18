#!/bin/bash
# WSD (satellite_components v71) version of the sweep, under the WSD scratch protocol
# (1000 epochs, patience 100, batch 16, deterministic, AMP). One array task per run:
#   0-1   full model (k=0) for n, s
#   2-19  (size, criterion, k) for size in n s, criterion in combined h0 h1, k in 1 2 3
# Needs ranking/wsd_{n,s}.json first -- submit with slurm/submit_wsd.sh.
#SBATCH --job-name=tp-wsd
#SBATCH --array=0-19
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

SIZES=(n s)
CRITS=(combined h0 h1)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
if (( TASK_ID < 2 )); then
  S=${SIZES[$TASK_ID]}; C=combined; K=0
else
  i=$((TASK_ID - 2))
  S=${SIZES[$((i / 9))]}; C=${CRITS[$((i % 9 / 3))]}; K=$((i % 3 + 1))
fi

source "${VENV:-.venv}/bin/activate"
echo "task ${TASK_ID}  ->  wsd ${S} ${C} k=${K}   ($(python -V 2>&1), $(hostname))"
python run_experiment.py --dataset wsd --size "$S" --criterion "$C" --ks "$K" --device 0 \
  --epochs 1000 --batch 16 --patience 100 --deterministic --amp
