#!/usr/bin/env bash
# COCO step 1 (one-shot only, no fine-tune), same setup as the WSD one-shot comparisons:
#   (a) activation-basis TDA criteria, seed 42  -> coco_n_protect6_activation_oneshot
#   (b) 10 random masks (seeds 0-9)              -> coco_n_protect6_oneshot
# PEEK-basis one-shot numbers come from the existing COCO grid (mv_exp/outputs/coco_outputs).
# Reads the COCO images/labels in mv_exp/datasets/coco (read-only) via configs/coco/coco.yaml.
set -euo pipefail
cd /shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
T=/shared/projects/tina/PEEK-TiNA
export REPO_ROOT=$PWD EPOCHS=0 BATCH=16 PROTECT_LAYERS=6 WORKERS=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export WEIGHTS=$T/yolo26n.pt DATA=$PWD/configs/coco/coco.yaml
SB=scripts/slurm/yolo26_tda.sbatch; P="--partition=gpu1,gpu2 --exclude=gpu06"
LOG=outputs/coco_oneshot_jobs.txt; : > $LOG

# (a) activation basis
export BASIS=activation CRITERIA="h0 h1 total field_h0 field_h1 field_total" OUT=$PWD/outputs/yolo26_tda/coco_n_protect6_activation_oneshot
mkdir -p $OUT
sc=$(sbatch --parsable $P --export=ALL --job-name=co-score --time=04:00:00 $SB score)
sf=$(sbatch --parsable $P --export=ALL --job-name=co-score --time=04:00:00 $SB score-field)
pr=$(sbatch --parsable $P --export=ALL,SEED=42 --job-name=co-act --time=01:00:00 --array=1-30 --dependency=afterok:$sc:$sf $SB prune)
ag=$(sbatch --parsable --export=ALL --job-name=co-act-agg --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 --dependency=afterany:$pr $SB aggregate)
echo "activation: score=$sc field=$sf prune=$pr agg=$ag" | tee -a $LOG

# (b) random masks, seeds 0-9
export CRITERIA=random OUT=$PWD/outputs/yolo26_tda/coco_n_protect6_oneshot
mkdir -p $OUT; ids=""
for s in 0 1 2 3 4 5 6 7 8 9; do
  ids="$ids:$(sbatch --parsable $P --export=ALL,SEED=$s --job-name=co-rnd-s$s --time=01:00:00 --array=1-5 $SB prune)"
done
ag=$(sbatch --parsable --export=ALL --job-name=co-rnd-agg --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 --dependency=afterany$ids $SB aggregate)
echo "random: prune=${ids#:} agg=$ag" | tee -a $LOG
