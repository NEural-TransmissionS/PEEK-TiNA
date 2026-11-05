#!/usr/bin/env bash
# H0, H1 and H0+H1 through every test (2026-10-29). Until now the other TDA criteria ran once (seed 42); this
# repeats every same-protocol and recovery test for them, each dataset on its own protocol (see
# submit_noise_seeds.sh), only the criterion changes:
#   channel: PEEK field H0 and field total (H0 + H1), next to field H1
#     ft       20-epoch fine-tune seeds 43-46 (5 seeds with the seed-42 grid) -> <ds>_<m>_protect6_seeds_ft20
#     oneshot  BN re-estimation seeds 43-51 (masks from the seed-42 scores)  -> <ds>_<m>_protect6_oneshot_seeds
#     recovery submit_recovery.sh with CRITERIA / ARRAY: probes (WSD, COCO: both warmups; viewsphere: matched),
#              budget sweep (WSD, viewsphere: seeds 42-44; COCO: seed 42 incl. 25k / 50k), COCO full train2017
#   blocks : H0-only and H1-only rankings (raw head, grid, null; k = 1-3), next to the combined blend. A cut
#            model depends only on which blocks are cut, so only cut sets no earlier run trained are trained
#            (plan_h0h1.py -> outputs/h0h1_blocks_{ft,rec,full}.tsv): cut + fine-tune for the missing seeds
#            of 42-44, then the same recovery tests (COCO full train2017: null on n, grid on s, k = 1 and 3).
# STUDIES (default all): ft oneshot blocks recovery. Log: outputs/h0h1_jobs.txt
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=$P/outputs/yolo26_tda
V=/shared/projects/nets/viewsphere-dataset/v1_wsd4
STUDIES=${STUDIES:-ft oneshot blocks recovery}
cd $P
export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW MOMENTUM=0.9 WARMUP_EPOCHS=1 CRITERIA="field_h0 field_total"   # array 1-5 field_h0, 6-10 field_total
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
PS="--partition=gpu1,gpu2,h200,h200_mig $X"   # COCO s: no 12 GB vdi GPUs
LOG=outputs/h0h1_jobs.txt; : > $LOG
BCOMMON="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --momentum 0.9 --warmup-epochs 1 --device 0"

setds() {  # ds m -> exports + PART, TL, BINIT, BFT (as submit_noise_seeds.sh)
  unset FT_DATA WARMUP_BIAS_LR
  case $1 in
    wsd) export WEIGHTS=$T/runs/wsd71_scratch/yolo26${2}_scratch_seed42/weights/best.pt DATA=$T/data/data.yaml LR0=0.00125
         BFT="--lr0 0.00125"; PART=$PW; TL=01:30:00 ;;
    coco) export WEIGHTS=$T/yolo26$2.pt DATA=$P/configs/coco/coco.yaml FT_DATA=$P/configs/coco/coco_ft10k.yaml LR0=0.000125
          BFT="--lr0 0.000125 --data $P/configs/coco/coco_ft10k.yaml"
          if [ $2 = s ]; then PART=$PS; TL=04:00:00; else PART=$PW; TL=03:00:00; fi ;;
    viewsphere) export WEIGHTS=$S/weights/viewsphere_$2.pt DATA=$V/data.yaml LR0=0.00125 WARMUP_BIAS_LR=0.00125
                BFT="--lr0 0.00125 --warmup-bias-lr 0.00125"; PART=$PW; TL=01:30:00 ;;
  esac
  BINIT=weights/${1}_$2.pt
}
prune() {  # name array seed time -> job id
  sbatch --parsable $PART --export=ALL,SEED=$3 --job-name=$1 --array=$2 --time=$4 $SB prune; }
agg() {  # name deps(:a:b)
  sbatch --parsable --partition=med --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate; }

(cd $S && ../tina_venv/bin/python $P/scripts/slurm/plan_h0h1.py $P/outputs) | tee -a $LOG

for ds in wsd coco viewsphere; do for m in n s; do
  setds $ds $m
  src=$O/${ds}_${m}_protect6_peek_ft20
  test -e $src/field_scores.pt || { echo "missing $src/field_scores.pt" >&2; exit 1; }
  if [[ " $STUDIES " == *" ft "* ]]; then
    export EPOCHS=20 BASIS=peek OUT=$O/${ds}_${m}_protect6_seeds_ft20
    ids=""; for s in 43 44 45 46; do ids="$ids:$(prune hc-$ds-$m-s$s 1-10 $s $TL)"; done
    echo "$ds $m channel field_h0/field_total ft seeds 43-46: ${ids#:} agg=$(agg hc-$ds-$m-agg $ids)" | tee -a $LOG
  fi
  if [[ " $STUDIES " == *" oneshot "* ]]; then
    export EPOCHS=0 BASIS=peek OUT=$O/${ds}_${m}_protect6_oneshot_seeds
    ids=""; for s in 43 44 45 46 47 48 49 50 51; do ids="$ids:$(prune ho-$ds-$m-s$s 1-10 $s 01:00:00)"; done
    echo "$ds $m channel field_h0/field_total one-shot seeds 43-51: ${ids#:} agg=$(agg ho-$ds-$m-agg $ids)" | tee -a $LOG
  fi
done; done

if [[ " $STUDIES " == *" blocks "* ]]; then
  cd $S
  while IFS=$'\t' read -r ds m lab cut args seeds; do
    setds $ds $m
    for s in ${seeds//,/ }; do
      j=$(sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$TL --job-name=hb-$ds-$m-$lab-s$s \
        --output=slurm/logs/h0h1_blocks_${ds}_${m}_${lab}_s$s.out \
        --wrap=". ../tina_venv/bin/activate; python _train_one.py --dataset $ds --size $m $args --basis peek --init-weights $BINIT $BCOMMON $BFT --seed $s --skip-existing")
      echo "block $ds $m $lab $cut s$s $j" >> $P/$LOG
    done
  done < $P/outputs/h0h1_blocks_ft.tsv
  echo "block cut + fine-tune: $(grep -c '^block ' $P/$LOG) jobs" | tee -a $P/$LOG
  cd $P
fi

if [[ " $STUDIES " == *" recovery "* ]]; then
  rec() { env -u FT_DATA -u WARMUP_BIAS_LR -u OUT -u EPOCHS CRITERIA="field_h0 field_total" ARRAY=1-6 \
            CUTS_FILE=$P/outputs/h0h1_blocks_rec.tsv CUTS_FULL_FILE=$P/outputs/h0h1_blocks_full.tsv "$@" bash scripts/slurm/submit_recovery.sh; }
  rec STUDIES=probe DATASETS="wsd coco" WBLR=0
  rec STUDIES=probe DATASETS="wsd coco viewsphere" WBLR=1
  for s in 42 43 44; do rec SEED=$s STUDIES=budget DATASETS="wsd viewsphere" WBLR=1; done
  rec STUDIES=budget DATASETS=coco WBLR=1 BUDGETS="250 500 1000 2500 5000 25000 50000"
  rec STUDIES=full DATASETS=coco
  echo "recovery: see outputs/recovery_jobs.txt (from $(date))" | tee -a $LOG
fi
echo "done; $(grep -c . $LOG) lines in $LOG"
