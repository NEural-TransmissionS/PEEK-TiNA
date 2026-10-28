#!/usr/bin/env bash
# Viewsphere (v1_wsd4: 4 satellites, WSD's 4 classes, split by view) phase 1: everything that does not
# need a block ranking. Base models = WSD scratch -> viewsphere fine-tunes
# (runs/viewsphere_wsd4_finetune_wsd/yolo26{n,s}_ft_seed42/weights/best.pt).
# Same recipe as WSD/COCO: AdamW lr 1.25e-3 momentum 0.9, 20 epochs, warmup 1, batch 16, seed 42 (+43/44),
# with the matched warmup bias lr (= lr0), the standard since 2026-10-22. Eval = v1_wsd4 valid (views 3, 14).
#   one-shot : 10 random masks (seeds 0-9), n and s
#   grids    : PEEK (all 8 criteria + unpruned, 0-40) and activation (TDA criteria, 1-30), n and s
#   seeds    : 43/44 for none/field_h1/l2/random, n and s
#   rankings : PEEK block rankings (head + all, --norms) and activation head ranking, on 40 train images
# Phase 2 (submit_viewsphere_phase2.sh) submits block cuts and the recovery study once rankings exist.
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=$P/outputs/yolo26_tda
V=/shared/projects/nets/viewsphere-dataset/v1_wsd4
cd $P

# block-study wiring: data/viewsphere + weights/viewsphere_{n,s}.pt
mkdir -p $S/data/viewsphere/images
ln -sfn $V/data.yaml $S/data/viewsphere/data.yaml
ln -sfn $V/train/images $S/data/viewsphere/images/train
ln -sfn $V/valid/images $S/data/viewsphere/images/val
for m in n s; do ln -sfn $T/runs/viewsphere_wsd4_finetune_wsd/yolo26${m}_ft_seed42/weights/best.pt $S/weights/viewsphere_$m.pt; done

export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW LR0=0.00125 MOMENTUM=0.9 EPOCHS=20 WARMUP_EPOCHS=1 WARMUP_BIAS_LR=0.00125
export DATA=$V/data.yaml; unset FT_DATA
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
TDA="h0 h1 total field_h0 field_h1 field_total"; ALL="h0 h1 total l2 random field_h0 field_h1 field_total"
LOG=outputs/viewsphere_jobs.txt; : > $LOG

score() {  # name stage basis -> job id
  sbatch --parsable $PW --export=ALL,BASIS=$3 --job-name=$1 --time=02:00:00 $SB $2; }
prune() {  # name array seed dep -> job id
  sbatch --parsable $PW --export=ALL,SEED=$3 --job-name=$1 --array=$2 --time=01:30:00 ${4:+--dependency=$4} $SB prune; }
agg() {  # name deps(:a:b) -> job id
  sbatch --parsable --partition=med --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate; }

for m in n s; do
  export WEIGHTS=$S/weights/viewsphere_$m.pt
  # one-shot random masks
  export EPOCHS=0 BASIS=peek CRITERIA=random OUT=$O/viewsphere_${m}_protect6_oneshot; mkdir -p $OUT; ro=""
  for s in 0 1 2 3 4 5 6 7 8 9; do ro="$ro:$(prune vs-$m-rnd$s 1-5 $s)"; done
  echo "viewsphere_$m random one-shot: prune=${ro#:} agg=$(agg vs-$m-rnd-agg $ro)" | tee -a $LOG
  export EPOCHS=20
  # PEEK grid
  export BASIS=peek OUT=$O/viewsphere_${m}_protect6_peek_ft20 CRITERIA="$ALL"; mkdir -p $OUT
  a=$(score vs-$m-sc score peek); b=$(score vs-$m-sf score-field peek); pkdep="afterok:$a:$b"
  pk=$(prune vs-$m-peek 0-40 42 $pkdep)
  echo "viewsphere_$m peek grid: scores=$a,$b prune=$pk agg=$(agg vs-$m-peek-agg :$pk)" | tee -a $LOG
  # activation grid (TDA criteria only; none/l2/random shared with the PEEK grid)
  export BASIS=activation OUT=$O/viewsphere_${m}_protect6_activation_ft20 CRITERIA="$TDA"; mkdir -p $OUT
  a=$(score vs-$m-asc score activation); b=$(score vs-$m-asf score-field activation)
  ac=$(prune vs-$m-act 1-30 42 afterok:$a:$b)
  echo "viewsphere_$m activation grid: scores=$a,$b prune=$ac agg=$(agg vs-$m-act-agg :$ac)" | tee -a $LOG
  # seeds 43/44
  export BASIS=peek OUT=$O/viewsphere_${m}_protect6_seeds_ft20 CRITERIA="field_h1 l2 random"; mkdir -p $OUT
  ln -sfn ../viewsphere_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
  sd=""; for s in 43 44; do sd="$sd:$(prune vs-$m-s$s 0-15 $s $pkdep)"; done
  echo "viewsphere_$m seeds: prune=${sd#:} agg=$(agg vs-$m-seeds-agg $sd)" | tee -a $LOG
done

# block rankings (PEEK head + all with norms; activation head), 40 train images
cd $S
for m in n s; do
  r=$(sbatch --parsable $PW --gres=gpu:1 --cpus-per-task=4 --time=02:00:00 --mem=16G --job-name=vs-rank-$m \
    --output=slurm/logs/viewsphere_rank_$m.out \
    --wrap=". ../tina_venv/bin/activate; for sc in head all; do python run_experiment.py --dataset viewsphere --size $m --ks --basis peek --scope \$sc --norms --rank-split train; done; python run_experiment.py --dataset viewsphere --size $m --ks --basis activation --scope head --rank-split train")
  echo "viewsphere_$m block rankings: $r" | tee -a $P/$LOG
done
echo "submitted: $(grep -c . $P/$LOG) lines in $LOG"
