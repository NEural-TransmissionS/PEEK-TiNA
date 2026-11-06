#!/usr/bin/env bash
# WSD follow-ups, held until the shapes structural-pruning study finishes:
#   (1) oneshot: 10 random-mask seeds, BN recal only (EPOCHS=0)
#   (2) ft{5,10,20}: short fine-tune budgets, warmup 1 epoch;
#       field_total/l2/random at seed 42 + random seeds 43,44
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
export REPO_ROOT=$PWD SBATCH_PARTITION=gpu2
export DATA=/shared/projects/tina/PEEK-TiNA/data/data.yaml BATCH=16 PROTECT_LAYERS=6 RATIOS="0.1 0.2 0.3 0.4 0.5"
DEP=afterany:89412:89414:89416:89418:89420:89422:89424:89426   # shapes aggregate jobs
SB=scripts/slurm/yolo26_tda.sbatch
LOG=outputs/wsd_followup_jobs.txt; : > $LOG

arr() {  # name ntasks seed
  sbatch --parsable --export=ALL,SEED=$3 --exclude=gpu06 --job-name=$1 --array=0-$(($2 - 1)) --dependency=$DEP $SB prune
}
agg() {  # name deps
  sbatch --parsable --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate
}

for m in n s; do
  export WEIGHTS=/shared/projects/tina/PEEK-TiNA/runs/wsd71_scratch/yolo26${m}_scratch_seed42/weights/best.pt

  export OUT=$PWD/outputs/yolo26_tda/wsd_${m}_protect6_oneshot EPOCHS=0 CRITERIA=random
  mkdir -p $OUT; ids=""
  for seed in 0 1 2 3 4 5 6 7 8 9; do ids="$ids:$(arr wsd1-$m-s$seed 6 $seed)"; done
  echo "oneshot_$m prune=${ids#:} agg=$(agg wsd1-$m-agg $ids)" | tee -a $LOG

  for ep in 5 10 20; do
    export OUT=$PWD/outputs/yolo26_tda/wsd_${m}_protect6_ft$ep EPOCHS=$ep WARMUP_EPOCHS=1
    mkdir -p $OUT; ln -sf ../wsd_${m}_protect6/field_scores.pt $OUT/field_scores.pt
    a=$(CRITERIA="field_total l2 random" arr wsd2-$m-ft$ep-s42 16 42)
    b=$(CRITERIA=random arr wsd2-$m-ft$ep-s43 6 43)
    c=$(CRITERIA=random arr wsd2-$m-ft$ep-s44 6 44)
    echo "ft${ep}_$m prune=$a,$b,$c agg=$(agg wsd2-$m-ft$ep-agg :$a:$b:$c)" | tee -a $LOG
  done
done
