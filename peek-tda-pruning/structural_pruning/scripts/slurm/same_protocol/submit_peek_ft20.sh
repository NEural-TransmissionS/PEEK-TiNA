#!/usr/bin/env bash
# PEEK-basis redo of the structural (Torch-Pruning) channel-pruning grids -- WSD n/s and
# shapes {circles,triangles,squares,mixed} x {n,s} -- with a 20-epoch fine-tune (warmup 1),
# full grid incl. l2/random/none, plus PEEK-basis block re-ranking for the block-cut study.
# Everything is held until the shapes structural-pruning study finishes.
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
TINA=/shared/projects/tina/PEEK-TiNA; SH=$TINA/topo-prune-shapes
export REPO_ROOT=$PWD BASIS=peek EPOCHS=20 WARMUP_EPOCHS=1 BATCH=16 PROTECT_LAYERS=6 WORKERS=16
export RATIOS="0.1 0.2 0.3 0.4 0.5" CRITERIA="h0 h1 total l2 random field_h0 field_h1 field_total"
DEP=afterany:89412:89414:89416:89418:89420:89422:89424:89426   # shapes aggregate jobs
SB=scripts/slurm/yolo26_tda.sbatch
LOG=outputs/peek_ft20_jobs.txt; : > $LOG

chain() {  # tag weights data cpus
  export WEIGHTS=$2 DATA=$3 OUT=$PWD/outputs/yolo26_tda/$1_protect6_peek_ft20; mkdir -p $OUT
  local sc sf pr ag
  sc=$(sbatch --parsable --partition=gpu1,gpu2 --exclude=gpu06 --export=ALL --job-name=pk-score \
    --time=04:00:00 --dependency=$DEP $SB score)
  sf=$(sbatch --parsable --partition=gpu1,gpu2 --exclude=gpu06 --export=ALL --job-name=pk-score \
    --time=04:00:00 --dependency=$DEP $SB score-field)
  pr=$(sbatch --parsable --partition=gpu2 --exclude=gpu06 --export=ALL,SEED=42 --cpus-per-task=$4 \
    --job-name=pk-prune --array=0-40 --dependency=afterok:$sc:$sf $SB prune)
  ag=$(sbatch --parsable --export=ALL --job-name=pk-agg --gres=none --cpus-per-task=2 --mem=8G \
    --time=00:30:00 --dependency=afterany:$pr $SB aggregate)
  echo "$1 score=$sc field=$sf prune=$pr agg=$ag" | tee -a $LOG
}

for m in n s; do
  chain wsd_$m $TINA/runs/wsd71_scratch/yolo26${m}_scratch_seed42/weights/best.pt $TINA/data/data.yaml 16
done
for ds in circles triangles squares mixed; do
  for m in n s; do chain shapes_${ds}_$m $SH/weights/${ds}_$m.pt $SH/data/$ds/data.yaml 12; done
done

# block-cut study: PEEK-basis re-ranking only (same split as the originals: WSD train, shapes val)
RANK=$(cd $SH && sbatch --parsable --job-name=pk-rank --partition=gpu2 --exclude=gpu06 --gres=gpu:1 \
  --time=01:00:00 --mem=16G --output=slurm/logs/peek_rank.out --dependency=$DEP \
  --wrap=". /shared/projects/tina/PEEK-TiNA/tina_venv/bin/activate; for s in n s; do python run_experiment.py --dataset wsd --size \$s --ks --basis peek --rank-split train; for d in circles triangles squares mixed; do python run_experiment.py --dataset \$d --size \$s --ks --basis peek; done; done")
echo "block rank (peek) $RANK -> $SH/ranking_peek/" | tee -a $LOG
