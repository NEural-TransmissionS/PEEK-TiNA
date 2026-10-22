#!/usr/bin/env bash
# Bring COCO onto the WSD setup (and WSD blocks onto the cut + fine-tune protocol COCO needs).
# Shared recipe: AdamW lr 1.25e-3 momentum 0.9, 20 epochs, warmup 1, batch 16, seed 42 (+43/44).
# COCO fine-tunes on a fixed 10k train2017 subset (configs/coco/coco_ft10k.yaml); scoring, BN
# re-estimation and evaluation use full COCO (configs/coco/coco.yaml, val2017).
#   COCO n : PEEK grid (scores reused from the teammate's PEEK run), activation grid (scores from
#            the one-shot step, jobs 124196/7), seeds 43/44
#   COCO s : PEEK + activation scoring and grids, seeds 43/44, 10 one-shot random masks, block ranking
#   WSD    : every block cut set already studied, redone as cut + fine-tune from the scratch models
# COCO block cuts are submitted separately once the COCO block rankings exist.
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes
cd $P
export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW LR0=0.00125 MOMENTUM=0.9 EPOCHS=20 WARMUP_EPOCHS=1
export DATA=$P/configs/coco/coco.yaml FT_DATA=$P/configs/coco/coco_ft10k.yaml
SB=scripts/slurm/yolo26_tda.sbatch; PART="--partition=gpu1,gpu2 --exclude=gpu06"
TDA="h0 h1 total field_h0 field_h1 field_total"; ALL="h0 h1 total l2 random field_h0 field_h1 field_total"
LOG=outputs/sync_wsd_coco_jobs.txt; : > $LOG
O=$P/outputs/yolo26_tda

score() {  # name stage basis -> job id
  BASIS=$3 sbatch --parsable $PART --export=ALL,BASIS=$3 --job-name=$1 --time=04:00:00 $SB $2; }
prune() {  # name array seed time dep -> job id
  sbatch --parsable $PART --export=ALL,SEED=$3 --job-name=$1 --array=$2 --time=$4 ${5:+--dependency=$5} $SB prune; }
agg() {  # name deps(:a:b) -> job id
  sbatch --parsable --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate; }

for m in n s; do
  export WEIGHTS=$T/yolo26$m.pt; tl=$([ $m = n ] && echo 04:00:00 || echo 06:00:00)
  # PEEK grid
  export OUT=$O/coco_${m}_protect6_peek_ft20 CRITERIA="$ALL"; mkdir -p $OUT
  if [ $m = n ]; then
    cp -n $T/mv_exp/outputs/coco_outputs/{tda_scores.pt,field_scores.pt,layer_persistence.csv,layer_field_drift.csv} $OUT/
    pk=$(prune cg-$m-peek 0-40 42 $tl); pkdep=""
  else
    a=$(score cs-$m-sc score peek); b=$(score cs-$m-sf score-field peek); pkdep="afterok:$a:$b"
    pk=$(prune cg-$m-peek 0-40 42 $tl $pkdep)
  fi
  echo "coco_$m peek grid: prune=$pk agg=$(agg cg-$m-peek-agg :$pk) ${pkdep:+scores=${pkdep#afterok:}}" | tee -a $LOG
  # activation grid (TDA criteria only; l2/random/none shared with the PEEK grid)
  export OUT=$O/coco_${m}_protect6_activation_ft20 CRITERIA="$TDA"; mkdir -p $OUT
  if [ $m = n ]; then
    ln -sfn ../coco_n_protect6_activation_oneshot/tda_scores.pt $OUT/tda_scores.pt
    ln -sfn ../coco_n_protect6_activation_oneshot/field_scores.pt $OUT/field_scores.pt
    acdep="afterok:124196:124197"
  else
    a=$(score cs-$m-asc score activation); b=$(score cs-$m-asf score-field activation); acdep="afterok:$a:$b"
  fi
  ac=$(prune cg-$m-act 1-30 42 $tl $acdep)
  echo "coco_$m activation grid: prune=$ac agg=$(agg cg-$m-act-agg :$ac)" | tee -a $LOG
  # seeds 43/44 with the train-loader seed fix
  export OUT=$O/coco_${m}_protect6_seeds_ft20 CRITERIA="field_h1 l2 random"; mkdir -p $OUT
  ln -sfn ../coco_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
  sd=""; for s in 43 44; do sd="$sd:$(prune cg-$m-s$s 0-15 $s $tl $pkdep)"; done
  echo "coco_$m seeds: prune=${sd#:} agg=$(agg cg-$m-seeds-agg $sd)" | tee -a $LOG
done

# COCO s: 10 one-shot random masks (n is the step-1 run)
export WEIGHTS=$T/yolo26s.pt OUT=$O/coco_s_protect6_oneshot CRITERIA=random EPOCHS=0; mkdir -p $OUT
ro=""; for s in 0 1 2 3 4 5 6 7 8 9; do ro="$ro:$(prune cg-s-rnd-s$s 1-5 $s 01:00:00)"; done
echo "coco_s random one-shot: prune=${ro#:} agg=$(agg cg-s-rnd-agg $ro)" | tee -a $LOG

# COCO s block ranking (n: job 124211)
cd $S
r=$(sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=4 --time=04:00:00 --mem=16G --job-name=co-rank-s \
  --output=slurm/logs/coco_rank_norm_s.out \
  --wrap=". ../tina_venv/bin/activate; for sc in head all; do python run_experiment.py --dataset coco --size s --ks --basis peek --scope \$sc --norms --rank-split train; done")
echo "coco_s block ranking: $r" | tee -a $P/$LOG

# WSD block cuts as cut + fine-tune (same recipe), every cut set already studied + full-model reference
ft="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --lr0 0.00125 --momentum 0.9 --warmup-epochs 1 --device 0"
wsd_block() {  # tag args...
  local tag=$1; shift
  sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name=wb-$tag \
    --output=slurm/logs/wsd_blockft_$tag.out \
    --wrap=". ../tina_venv/bin/activate; python _train_one.py --dataset wsd $* --basis peek --init-weights weights/wsd_\$WSD_SIZE.pt $ft"
}
for m in n s; do
  export WSD_SIZE=$m
  ids="$(wsd_block $m-k0 --size $m --k 0)"
  for k in 1 2 3; do
    ids="$ids $(wsd_block $m-head-k$k --size $m --k $k --scope head)"
    for v in grid null; do ids="$ids $(wsd_block $m-$v-k$k --size $m --k $k --scope all --norm $v)"; done
    for s in 0 1 2; do
      { [ $m = n ] && [ $s = 1 ] && [ $k = 1 ]; } && continue   # {16} = null k1
      { [ $m = s ] && [ $s = 0 ] && [ $k = 1 ]; } && continue   # {6}  = grid k1
      ids="$ids $(wsd_block $m-r$s-k$k --size $m --k $k --scope all --random-order $s)"
    done
  done
  echo "wsd_$m block cut+fine-tune: $ids" | tee -a $P/$LOG
done
