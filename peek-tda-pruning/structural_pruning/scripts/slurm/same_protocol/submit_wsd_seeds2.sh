#!/usr/bin/env bash
# WSD only, with the fine-tune seed fix (seed now reaches Ultralytics' train-loader stream):
# PEEK field H1, L2, random and the unpruned control on seeds 43/44, same 20-epoch setup.
# Seed 42 is the *_peek_ft20 grid (unchanged by the fix). Fresh OUT so the earlier seed runs,
# which were not independent fine-tunes, are never mixed in.
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
TINA=/shared/projects/tina/PEEK-TiNA
export REPO_ROOT=$PWD EPOCHS=20 WARMUP_EPOCHS=1 BATCH=16 PROTECT_LAYERS=6 WORKERS=16
export RATIOS="0.1 0.2 0.3 0.4 0.5" CRITERIA="field_h1 l2 random"   # 0 none, 1-5 field_h1, 6-10 l2, 11-15 random
export DATA=$TINA/data/data.yaml
SB=scripts/slurm/yolo26_tda.sbatch
LOG=outputs/wsd_seeds2_jobs.txt; : > $LOG
for m in n s; do
  export WEIGHTS=$TINA/runs/wsd71_scratch/yolo26${m}_scratch_seed42/weights/best.pt
  export OUT=$PWD/outputs/yolo26_tda/wsd_${m}_protect6_seeds_ft20
  mkdir -p $OUT && ln -sfn ../wsd_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
  ids=""
  for s in 43 44; do
    ids="$ids:$(sbatch --parsable --partition=gpu2 --exclude=gpu06 --export=ALL,SEED=$s \
      --job-name=ws2-$m-s$s --array=0-15 $SB prune)"
  done
  ag=$(sbatch --parsable --export=ALL --job-name=ws2-$m-agg --gres=none --cpus-per-task=2 --mem=8G \
    --time=00:30:00 --dependency=afterany$ids $SB aggregate)
  echo "wsd_$m prune=${ids#:} agg=$ag" | tee -a $LOG
done
