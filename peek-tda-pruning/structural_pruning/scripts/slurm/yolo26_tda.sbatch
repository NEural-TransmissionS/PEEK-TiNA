#!/usr/bin/env bash
# One stage of the YOLO26 TDA pruning experiment. Launched by
# submit_yolo26_tda.sh; the stage is the first argument:
#   score      per-channel persistence scores (once)
#   prune      array job, one (criterion, ratio) per SLURM_ARRAY_TASK_ID
#   aggregate  results.csv + map_vs_ratio.png
#
# Partition/account/QOS: pass them to sbatch or export SBATCH_PARTITION,
# SBATCH_ACCOUNT, SBATCH_QOS before submitting (sbatch reads those env vars).
#SBATCH --job-name=yolo26-tda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
source "${REPO_ROOT:?submit via scripts/slurm/submit_yolo26_tda.sh}/scripts/slurm/yolo26_tda_env.sh"
activate_env
STAGE="$1"
echo "[$(date)] stage=$STAGE task=${SLURM_ARRAY_TASK_ID:-} seed=${SEED:-0} host=$(hostname)"
nvidia-smi -L || true

case "$STAGE" in
  score|score-field)
    run_py "$STAGE" --calib-images "$CALIB_IMAGES" --max-side "$MAX_SIDE" --basis "${BASIS:-peek}" \
      --workers "${SLURM_CPUS_PER_TASK:-8}" --device 0
    ;;
  prune)
    # shellcheck disable=SC2086
    run_py prune --grid-index "$SLURM_ARRAY_TASK_ID" --ratios $RATIOS --criteria $CRITERIA --seed "${SEED:-0}" \
      --calib-images "$CALIB_IMAGES" --protect-layers "$PROTECT_LAYERS" \
      --epochs "$EPOCHS" --warmup-epochs "${WARMUP_EPOCHS:-3}" --optimizer "${OPTIMIZER:-auto}" \
      ${LR0:+--lr0 "$LR0"} ${MOMENTUM:+--momentum "$MOMENTUM"} ${FT_DATA:+--finetune-data "$FT_DATA"} --batch "$BATCH" --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" --device 0
    ;;
  aggregate)
    run_py aggregate
    ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
echo "[$(date)] done"
