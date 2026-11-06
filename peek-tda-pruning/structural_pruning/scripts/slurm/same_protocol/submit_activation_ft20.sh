#!/usr/bin/env bash
# Activation-basis TDA criteria on the exact PEEK setup (same models, 20-epoch fine-tune,
# warmup 1, batch 16, seed 42, calibration images), so both bases share one table.
# Only the 6 TDA criteria run (array 1-30); l2/random/none are basis-independent and come
# from the *_peek_ft20 grids. Plus activation-basis block re-ranking (current image selection).
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
TINA=/shared/projects/tina/PEEK-TiNA; SH=$TINA/topo-prune-shapes; ARCH=$TINA/activation_tda/structural_pruning
export REPO_ROOT=$PWD BASIS=activation EPOCHS=20 WARMUP_EPOCHS=1 BATCH=16 PROTECT_LAYERS=6 WORKERS=16
export RATIOS="0.1 0.2 0.3 0.4 0.5" CRITERIA="h0 h1 total field_h0 field_h1 field_total"
SB=scripts/slurm/yolo26_tda.sbatch
LOG=outputs/activation_ft20_jobs.txt; : > $LOG

chain() {  # tag weights data cpus
  export WEIGHTS=$2 DATA=$3 OUT=$PWD/outputs/yolo26_tda/$1_protect6_activation_ft20; mkdir -p $OUT
  # activation scores already computed (same model/seed/calibration) are reused as-is
  for f in tda_scores.pt field_scores.pt; do
    [ -f $ARCH/$1_protect6/$f ] && cp -n $ARCH/$1_protect6/$f $OUT/
  done
  local sc sf pr ag
  sc=$(sbatch --parsable --partition=gpu1,gpu2 --exclude=gpu06 --export=ALL --job-name=ac-score --time=04:00:00 $SB score)
  sf=$(sbatch --parsable --partition=gpu1,gpu2 --exclude=gpu06 --export=ALL --job-name=ac-score --time=04:00:00 $SB score-field)
  pr=$(sbatch --parsable --partition=gpu2 --exclude=gpu06 --export=ALL,SEED=42 --cpus-per-task=$4 \
    --job-name=ac-prune --array=1-30 --dependency=afterok:$sc:$sf $SB prune)
  ag=$(sbatch --parsable --export=ALL --job-name=ac-agg --gres=none --cpus-per-task=2 --mem=8G \
    --time=00:30:00 --dependency=afterany:$pr $SB aggregate)
  echo "$1 score=$sc field=$sf prune=$pr agg=$ag" | tee -a $LOG
}

for m in n s; do
  chain wsd_$m $TINA/runs/wsd71_scratch/yolo26${m}_scratch_seed42/weights/best.pt $TINA/data/data.yaml 16
done
for ds in circles triangles squares mixed; do
  for m in n s; do chain shapes_${ds}_$m $SH/weights/${ds}_$m.pt $SH/data/$ds/data.yaml 12; done
done

RANK=$(cd $SH && sbatch --parsable --job-name=ac-rank --partition=gpu2 --exclude=gpu06 --gres=gpu:1 \
  --time=01:00:00 --mem=16G --output=slurm/logs/activation_rank.out \
  --wrap=". /shared/projects/tina/PEEK-TiNA/tina_venv/bin/activate; for s in n s; do python run_experiment.py --dataset wsd --size \$s --ks --basis activation --rank-split train; for d in circles triangles squares mixed; do python run_experiment.py --dataset \$d --size \$s --ks --basis activation; done; done")
echo "block rank (activation) $RANK -> $SH/ranking/" | tee -a $LOG
