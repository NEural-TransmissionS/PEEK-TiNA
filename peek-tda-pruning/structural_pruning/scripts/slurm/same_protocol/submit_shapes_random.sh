#!/usr/bin/env bash
# Random pruning on the shapes with the same seeds as WSD: 10 one-shot masks (seeds 0-9,
# BN re-estimation only) and 20-epoch fine-tunes for seeds 43/44 (seed 42 is in the grid).
# Random-only grid: index 0 = none (not rerun), 1-5 = random at 10-50%.
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
SH=/shared/projects/tina/PEEK-TiNA/topo-prune-shapes
export REPO_ROOT=$PWD BATCH=16 PROTECT_LAYERS=6 WORKERS=16 RATIOS="0.1 0.2 0.3 0.4 0.5" CRITERIA=random
SB=scripts/slurm/yolo26_tda.sbatch
LOG=outputs/shapes_random_jobs.txt; : > $LOG

submit() {  # out epochs warmup name seeds...
  local out=$1 ep=$2 wu=$3 name=$4; shift 4
  export OUT=$out EPOCHS=$ep WARMUP_EPOCHS=$wu; mkdir -p $OUT
  local ids=""
  for s in "$@"; do
    ids="$ids:$(sbatch --parsable --partition=gpu2 --exclude=gpu06 --cpus-per-task=12 \
      --export=ALL,SEED=$s --job-name=$name --array=1-5 $SB prune)"
  done
  local ag=$(sbatch --parsable --export=ALL --job-name=$name-agg --gres=none --cpus-per-task=2 \
    --mem=8G --time=00:30:00 --dependency=afterany$ids $SB aggregate)
  echo "$(basename $out) prune=${ids#:} agg=$ag" | tee -a $LOG
}

for ds in circles triangles squares mixed; do
  for m in n s; do
    export WEIGHTS=$SH/weights/${ds}_$m.pt DATA=$SH/data/$ds/data.yaml
    base=$PWD/outputs/yolo26_tda/shapes_${ds}_${m}_protect6
    submit ${base}_oneshot 0 3 rs-oneshot 0 1 2 3 4 5 6 7 8 9
    submit ${base}_ft20 20 1 rs-ft20 43 44
  done
done
