#!/usr/bin/env bash
# WSD only: PEEK field H1 and L2 on seeds 43/44 (seed 42 is in the *_peek_ft20 grid), same
# 20-epoch setup, so the key criteria have the same three seeds as random. Outputs go next to
# random's seeds 43/44 in wsd_{n,s}_protect6_ft20; PEEK field scores come from the seed-42 grid.
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
TINA=/shared/projects/tina/PEEK-TiNA
export REPO_ROOT=$PWD EPOCHS=20 WARMUP_EPOCHS=1 BATCH=16 PROTECT_LAYERS=6 WORKERS=16
export RATIOS="0.1 0.2 0.3 0.4 0.5" CRITERIA="field_h1 l2"   # grid: 0 none, 1-5 field_h1, 6-10 l2
export DATA=$TINA/data/data.yaml
SB=scripts/slurm/yolo26_tda.sbatch
LOG=outputs/wsd_seeds_jobs.txt; : > $LOG
for m in n s; do
  export WEIGHTS=$TINA/runs/wsd71_scratch/yolo26${m}_scratch_seed42/weights/best.pt
  export OUT=$PWD/outputs/yolo26_tda/wsd_${m}_protect6_ft20
  ln -sfn ../wsd_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
  ids=""
  for s in 43 44; do
    ids="$ids:$(sbatch --parsable --partition=gpu2 --exclude=gpu06 --export=ALL,SEED=$s \
      --job-name=ws-$m-s$s --array=1-10 $SB prune)"
  done
  ag=$(sbatch --parsable --export=ALL --job-name=ws-$m-agg --gres=none --cpus-per-task=2 --mem=8G \
    --time=00:30:00 --dependency=afterany$ids $SB aggregate)
  echo "wsd_$m prune=${ids#:} agg=$ag" | tee -a $LOG
done
