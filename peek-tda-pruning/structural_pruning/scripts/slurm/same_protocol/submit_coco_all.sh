#!/usr/bin/env bash
# All COCO channel-pruning work on our own COCO copy (coco_own), same recipe as WSD:
# AdamW lr 1.25e-3 momentum 0.9, 20 epochs, warmup 1, batch 16, seed 42 (+43/44); fine-tune on the
# fixed 10k train2017 subset (coco_ft10k.yaml), BN re-estimation + eval on full COCO (coco.yaml).
#   one-shot : n activation-basis criteria; 10 random masks for n and s (EPOCHS=0)
#   grids    : PEEK (n reuses the teammate's PEEK scores) and activation, n and s
#   seeds    : 43/44 for none/field_h1/l2/random, n and s
#   blocks   : PEEK block rankings (head + all scopes, grid/null norms); cut+FT submitted once ranked
# Partitions: every GPU partition; COCO s fine-tunes skip the 12 GB vdi GPUs, vdi-short (1 h) only for one-shot.
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes
cd $P
export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW LR0=0.00125 MOMENTUM=0.9 EPOCHS=20 WARMUP_EPOCHS=1
export DATA=$P/configs/coco/coco.yaml FT_DATA=$P/configs/coco/coco_ft10k.yaml
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PN="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
PS="--partition=gpu1,gpu2,h200,h200_mig $X"
P1="--partition=gpu1,gpu2,h200,h200_mig,vdi-short,vdi-med,vdi-long $X"
TDA="h0 h1 total field_h0 field_h1 field_total"; ALL="h0 h1 total l2 random field_h0 field_h1 field_total"
LOG=outputs/coco_all_jobs.txt; : > $LOG
O=$P/outputs/yolo26_tda

score() {  # name stage basis -> job id
  sbatch --parsable $PN --export=ALL,BASIS=$3 --job-name=$1 --time=04:00:00 $SB $2; }
prune() {  # part name array seed time dep -> job id
  sbatch --parsable $1 --export=ALL,SEED=$4 --job-name=$2 --array=$3 --time=$5 ${6:+--dependency=$6} $SB prune; }
agg() {  # name deps(:a:b) -> job id
  sbatch --parsable --partition=med --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate; }

# n one-shot: activation scores (also feed the n activation grid) + 30 one-shot runs
export WEIGHTS=$T/yolo26n.pt BASIS=activation CRITERIA="$TDA" EPOCHS=0 OUT=$O/coco_n_protect6_activation_oneshot
a=$(score ca-n-asc score activation); b=$(score ca-n-asf score-field activation); nact="afterok:$a:$b"
pr=$(prune "$P1" ca-n-act1 1-30 42 01:00:00 $nact)
echo "coco_n activation one-shot: scores=$a,$b prune=$pr agg=$(agg ca-n-act1-agg :$pr)" | tee -a $LOG
# 10 random one-shot masks, n and s
export BASIS=peek CRITERIA=random
for m in n s; do
  export WEIGHTS=$T/yolo26$m.pt OUT=$O/coco_${m}_protect6_oneshot; ro=""
  for s in 0 1 2 3 4 5 6 7 8 9; do ro="$ro:$(prune "$P1" ca-$m-rnd$s 1-5 $s 01:00:00)"; done
  echo "coco_$m random one-shot: prune=${ro#:} agg=$(agg ca-$m-rnd-agg $ro)" | tee -a $LOG
done
export EPOCHS=20

for m in n s; do
  export WEIGHTS=$T/yolo26$m.pt
  if [ $m = n ]; then PP=$PN; tl=04:00:00; else PP=$PS; tl=06:00:00; fi
  # PEEK grid
  export BASIS=peek OUT=$O/coco_${m}_protect6_peek_ft20 CRITERIA="$ALL"; mkdir -p $OUT
  if [ $m = n ]; then pkdep=""; else
    a=$(score ca-$m-sc score peek); b=$(score ca-$m-sf score-field peek); pkdep="afterok:$a:$b"; fi
  pk=$(prune "$PP" ca-$m-peek 0-40 42 $tl $pkdep)
  echo "coco_$m peek grid: prune=$pk agg=$(agg ca-$m-peek-agg :$pk) ${pkdep:+scores=${pkdep#afterok:}}" | tee -a $LOG
  # activation grid (TDA criteria only; none/l2/random shared with the PEEK grid)
  export BASIS=activation OUT=$O/coco_${m}_protect6_activation_ft20 CRITERIA="$TDA"; mkdir -p $OUT
  if [ $m = n ]; then
    ln -sfn ../coco_n_protect6_activation_oneshot/tda_scores.pt $OUT/tda_scores.pt
    ln -sfn ../coco_n_protect6_activation_oneshot/field_scores.pt $OUT/field_scores.pt
    acdep=$nact
  else
    a=$(score ca-$m-asc score activation); b=$(score ca-$m-asf score-field activation); acdep="afterok:$a:$b"; fi
  ac=$(prune "$PP" ca-$m-act 1-30 42 $tl $acdep)
  echo "coco_$m activation grid: prune=$ac agg=$(agg ca-$m-act-agg :$ac)" | tee -a $LOG
  # seeds 43/44 (train-loader seed fix)
  export BASIS=peek OUT=$O/coco_${m}_protect6_seeds_ft20 CRITERIA="field_h1 l2 random"; mkdir -p $OUT
  ln -sfn ../coco_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
  sd=""; for s in 43 44; do sd="$sd:$(prune "$PP" ca-$m-s$s 0-15 $s $tl $pkdep)"; done
  echo "coco_$m seeds: prune=${sd#:} agg=$(agg ca-$m-seeds-agg $sd)" | tee -a $LOG
done

# PEEK block rankings (cut + fine-tune runs are submitted once these exist)
cd $S
for m in n s; do
  r=$(sbatch --parsable $PN --gres=gpu:1 --cpus-per-task=4 --time=04:00:00 --mem=16G --job-name=ca-rank-$m \
    --output=slurm/logs/coco_rank_norm_$m.out \
    --wrap=". ../tina_venv/bin/activate; for sc in head all; do python run_experiment.py --dataset coco --size $m --ks --basis peek --scope \$sc --norms --rank-split train; done")
  echo "coco_$m block ranking: $r" | tee -a $P/$LOG
done
