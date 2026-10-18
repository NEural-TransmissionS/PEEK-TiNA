#!/bin/bash
# One array task per (dataset, size). Each ranks the candidate blocks once, then
# trains full + k=1/2/3 pruned models (each in its own subprocess).
#SBATCH --job-name=tp-sweep
#SBATCH --array=0-7
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%a.out
# ai-panther GPU partitions: gpu1 (A100-SXM4) gpu2 (A100-PCIe) h200 h200_mig vdi-{short,med,long} (L40S).
# Override the partition without editing:  SBATCH_PARTITION=gpu1 bash slurm/submit.sh
# Partitions/limits:  sinfo -o "%P %.5D %.14G %.11l"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

DATASETS=(circles triangles squares mixed)
SIZES=(n s)
# array job -> $SLURM_ARRAY_TASK_ID; run directly -> first arg, else 0
TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
D=${DATASETS[$((TASK_ID / 2))]}
S=${SIZES[$((TASK_ID % 2))]}

module load cuda 2>/dev/null || true
[ -n "${PY_MODULE:-}" ] && module load "$PY_MODULE"
# shellcheck disable=SC1090
source "${VENV:-.venv}/bin/activate" 2>/dev/null || true
PY="${PYTHON:-python3}"

# fail loudly rather than run on a stale system python
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' || {
  echo "ERROR: '$PY' is $("$PY" -V 2>&1) -- need >= 3.10." >&2
  echo "  Build the venv (README) or pass VENV=/path/to/env and/or PY_MODULE=python/3.11." >&2
  exit 1
}

echo "task ${TASK_ID}  ->  ${D} ${S} ${CRITERION:-combined}   ($($PY -V 2>&1))"
"$PY" run_experiment.py --dataset "$D" --size "$S" \
  --epochs "${EPOCHS:-60}" --n-rank "${N_RANK:-40}" --device 0 \
  --criterion "${CRITERION:-combined}" --ks ${KS:-0 1 2 3}
