#!/usr/bin/env bash
# The whole same-protocol + recovery study again on YOLOv5u (Ultralytics' anchor-free YOLOv5n/s), 2026-11-02.
# Model ids v5n / v5s (YOLO26 = n / s): outputs <ds>_v5<m>_protect6_*, topo-prune-shapes weights/<ds>_v5<m>.pt,
# ranking_peek/<ds>_v5<m>*.json, results/<ds>_v5<m>_*. Base models: COCO = yolov5{n,s}u.pt; WSD = trained from
# scratch like YOLO26's (tina/scripts/train_wsd_baselines.slurm, ARCH=yolov5); viewsphere = those fine-tuned on
# v1_wsd4 (train_viewsphere_finetune.slurm, MODEL=yolov5{n,s}). Each dataset keeps its YOLO26 protocol
# (submit_noise_seeds.sh / submit_h0h1.sh); criteria are those the report uses: PEEK field H1 / H0 / H0+H1, L2,
# random. Not repeated: per-channel h0 / h1 / total, activation basis, lr pilot, viewsphere scratch block cuts.
#   bash submit_yolov5.sh channel <ds>   scoring, channel grid (seed 42) + seeds 43-46, one-shot random masks 0-9
#                                         and BN seeds 43-51, block rankings (raw head; all with grid / null);
#                                         then queues "blocks <ds>" to run once the rankings exist
#   bash submit_yolov5.sh blocks <ds>    cut + fine-tune: k0 and every ranking's cut (combined / h0 / h1 x head /
#                                         grid / null x k1-3) seeds 42-44, 23 random orders seed 42 (plan:
#                                         plan_yolov5.py); recovery (submit_recovery.sh with MODELS): probes,
#                                         budgets (WSD / viewsphere seeds 42-44; COCO seed 42 incl. 25k / 50k),
#                                         COCO full train2017
# DEP=<afterok:ids> holds the "channel" stage until the base models exist. Log: outputs/<TAG>_jobs.txt
# MODELS / TAG (default v5n v5s / yolov5; MODELS="n s" TAG=yolo26 runs YOLO26) and dataset viewsphere2
# (viewsphere v2: 17 satellites, v2_wsd4; protocol as viewsphere) added 2026-11-03 for submit_viewsphere2.sh.
# Added 2026-11-06 for submit_rerun_coco.sh: NOISE_VS = training-only pixel noise (gray levels, PIXEL_NOISE in
# the fine-tunes) on viewsphere / viewsphere2; base-weight links are only created when missing (the re-run sets
# them); O_ROOT / JOBLOG move the channel outputs and the job log (dry runs).
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/sbatch_retry.sh"   # retries when the controller is busy
PHASE=$1; DS=$2
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=${O_ROOT:-$P/outputs/yolo26_tda}
NOISE_VS=${NOISE_VS:-}
V=/shared/projects/nets/viewsphere-dataset/v1_wsd4; V2=/shared/projects/nets/viewsphere-dataset/v2_wsd4
MODELS=${MODELS:-v5n v5s}; TAG=${TAG:-yolov5}; CR="field_h1 field_h0 field_total l2 random"   # grid: 0 none, then 5 ratios per criterion
JP=$([ $TAG = yolov5 ] && echo v5 || echo y26)   # job-name prefix
cd $P
export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=8 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW MOMENTUM=0.9 WARMUP_EPOCHS=1
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
PS="--partition=gpu1,gpu2,h200,h200_mig $X"   # COCO s: no 12 GB vdi GPUs
LOG=${JOBLOG:-$P/outputs/${TAG}_jobs.txt}; PLAN=${PLAN_DIR:-$P/outputs}   # PLAN_DIR: block plans (dry runs)
BCOMMON="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --momentum 0.9 --warmup-epochs 1 --device 0"

# base weights, once: weights/<ds>_<m>.pt (WSD / viewsphere ones exist once their training finishes);
# an existing link is kept, so the COCO-start bases that submit_rerun_coco.sh links stay in place
link() { [ -L "$2" ] || [ -e "$2" ] || ln -sn "$1" "$2"; }
for m in n s; do
  link $T/yolov5${m}u.pt $S/weights/coco_v5$m.pt
  link $T/runs/wsd71_scratch/yolov5${m}_scratch_seed42/weights/best.pt $S/weights/wsd_v5$m.pt
  link $T/runs/viewsphere_wsd4_finetune_wsd/yolov5${m}_ft_seed42/weights/best.pt $S/weights/viewsphere_v5$m.pt
  link $T/runs/viewsphere2_wsd4_finetune_wsd/yolov5${m}_ft_seed42/weights/best.pt $S/weights/viewsphere2_v5$m.pt
  link $T/runs/viewsphere2_wsd4_finetune_wsd/yolo26${m}_ft_seed42/weights/best.pt $S/weights/viewsphere2_$m.pt
done

setds() {  # ds m -> exports + PART, TL, BINIT, BFT (each dataset's protocol, as submit_noise_seeds.sh)
  unset FT_DATA WARMUP_BIAS_LR PIXEL_NOISE
  [[ $1 == viewsphere* ]] && [ -n "$NOISE_VS" ] && export PIXEL_NOISE=$NOISE_VS
  export WEIGHTS=$S/weights/${1}_$2.pt
  case $1 in
    wsd) export DATA=$T/data/data.yaml LR0=0.00125; BFT="--lr0 0.00125"; PART=$PW; TL=01:30:00 ;;
    coco) export DATA=$P/configs/coco/coco.yaml FT_DATA=$P/configs/coco/coco_ft10k.yaml LR0=0.000125
          BFT="--lr0 0.000125 --data $P/configs/coco/coco_ft10k.yaml"
          if [[ $2 == *s ]]; then PART=$PS; TL=04:00:00; else PART=$PW; TL=03:00:00; fi ;;
    viewsphere) export DATA=$V/data.yaml LR0=0.00125 WARMUP_BIAS_LR=0.00125
                BFT="--lr0 0.00125 --warmup-bias-lr 0.00125"; PART=$PW; TL=01:30:00 ;;
    viewsphere2) export DATA=$V2/data.yaml LR0=0.00125 WARMUP_BIAS_LR=0.00125   # 255 batches / epoch: 4x v1
                 BFT="--lr0 0.00125 --warmup-bias-lr 0.00125"; PART=$PW; TL=03:00:00 ;;
  esac
  BINIT=weights/${1}_$2.pt
}
prune() {  # name array seed time dep -> job id
  sbatch --parsable $PART --export=ALL,SEED=$3 --job-name=$1 --array=$2 --time=$4 ${5:+--dependency=$5} $SB prune; }
agg() {  # name deps(:a:b)
  sbatch --parsable --partition=short,med --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate; }
later() {  # dep phase -> job id of a small job that runs "submit_yolov5.sh <phase> $DS" once dep is met
  # on the GPU partitions (1 GPU, minutes): med is often full for hours; sbatch works inside jobs here
  sbatch --parsable $PW --gres=gpu:1 --cpus-per-task=2 --mem=4G --time=01:00:00 --export=NONE \
    --dependency=$1 --job-name=$JP-$2-$DS --output=$P/slurm_logs/$JP-submit-$2-$DS-%j.out \
    --wrap="bash -lc 'cd $P && MODELS=\"$MODELS\" TAG=$TAG NOISE_VS=$NOISE_VS bash scripts/slurm/submit_yolov5.sh $2 $DS'"; }

if [ $PHASE = channel ]; then
  if [ -n "${DEP:-}" ]; then echo "$DS: channel stage waits for $DEP: $(later $DEP channel)" | tee -a $LOG; exit 0; fi
  ranks=""
  for m in $MODELS; do
    setds $DS $m
    test -e $WEIGHTS || { echo "missing $WEIGHTS" >&2; exit 1; }
    G=$O/${DS}_${m}_protect6_peek_ft20; mkdir -p $G
    sf=$(sbatch --parsable $PART --export=ALL,BASIS=peek,OUT=$G --job-name=${JP}sf-$DS-$m --time=04:00:00 $SB score-field)
    export BASIS=peek EPOCHS=20 CRITERIA="$CR" OUT=$G
    g=$(prune ${JP}g-$DS-$m 0-25 42 $TL afterok:$sf)
    echo "$DS $m field scores $sf; grid seed 42: $g agg=$(agg ${JP}g-$DS-$m-agg :$g)" | tee -a $LOG
    export OUT=$O/${DS}_${m}_protect6_seeds_ft20; mkdir -p $OUT; ln -sfn ../${DS}_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
    ids=""; for s in 43 44 45 46; do ids="$ids:$(prune ${JP}s-$DS-$m-s$s 0-25 $s $TL afterok:$sf)"; done
    echo "$DS $m grid seeds 43-46: ${ids#:} agg=$(agg ${JP}s-$DS-$m-agg $ids)" | tee -a $LOG
    export EPOCHS=0 CRITERIA=random OUT=$O/${DS}_${m}_protect6_oneshot; mkdir -p $OUT
    ids=""; for s in 0 1 2 3 4 5 6 7 8 9; do ids="$ids:$(prune ${JP}r-$DS-$m-$s 1-5 $s 01:00:00)"; done
    echo "$DS $m one-shot random masks 0-9: ${ids#:} agg=$(agg ${JP}r-$DS-$m-agg $ids)" | tee -a $LOG
    export CRITERIA="field_h1 field_h0 field_total l2" OUT=$O/${DS}_${m}_protect6_oneshot_seeds; mkdir -p $OUT
    ln -sfn ../${DS}_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
    ids=""; for s in 43 44 45 46 47 48 49 50 51; do ids="$ids:$(prune ${JP}o-$DS-$m-s$s 1-20 $s 01:00:00 afterok:$sf)"; done
    echo "$DS $m one-shot BN seeds 43-51: ${ids#:} agg=$(agg ${JP}o-$DS-$m-agg $ids)" | tee -a $LOG
    # block rankings on 40 train images: raw head (ranking_peek/<ds>_<m>.json), all 8 C3 with grid / null (_all_norm)
    r=$(cd $S && sbatch --parsable $PW --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --job-name=${JP}rank-$DS-$m \
      --output=slurm/logs/v5rank_${DS}_$m.out \
      --wrap=". /shared/projects/tina/PEEK-TiNA/tina_venv/bin/activate; python run_experiment.py --dataset $DS --size $m --ks --basis peek --scope head --rank-split train; python run_experiment.py --dataset $DS --size $m --ks --basis peek --scope all --norms --rank-split train")
    ranks="$ranks:$r"
    echo "$DS $m block rankings: $r" | tee -a $LOG
  done
  echo "$DS: blocks stage queued after the rankings: $(later afterok$ranks blocks)" | tee -a $LOG
fi

if [ $PHASE = blocks ]; then
  cd $S
  /shared/projects/tina/PEEK-TiNA/tina_venv/bin/python $P/scripts/slurm/plan_yolov5.py $DS $PLAN $TAG $MODELS | tee -a $LOG
  while IFS=$'\t' read -r ds m lab cut args seeds; do
    setds $ds $m
    for s in ${seeds//,/ }; do
      j=$(sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=$TL --job-name=${JP}b-$ds-$m-$lab-s$s \
        --output=slurm/logs/v5blocks_${ds}_${m}_${lab}_s$s.out \
        --wrap=". /shared/projects/tina/PEEK-TiNA/tina_venv/bin/activate; python _train_one.py --dataset $ds --size $m $args --basis peek --init-weights $BINIT $BCOMMON $BFT --seed $s --skip-existing")
      echo "block $ds $m $lab $cut s$s $j" >> $LOG
    done
  done < $PLAN/${TAG}_${DS}_blocks_ft.tsv
  echo "$DS block cut + fine-tune: $(grep -c "^block $DS " $LOG) jobs" | tee -a $LOG
  cd $P
  rec() { env -u FT_DATA -u WARMUP_BIAS_LR -u OUT -u EPOCHS -u CRITERIA MODELS="$MODELS" CRITERIA="$CR" ARRAY=0-15 \
            CUTS_FILE=$PLAN/${TAG}_${DS}_blocks_rec.tsv CUTS_FULL_FILE=$PLAN/${TAG}_${DS}_blocks_full.tsv \
            DATASETS=$DS "$@" bash scripts/slurm/submit_recovery.sh; }
  case $DS in
    wsd) rec STUDIES=probe WBLR=0; rec STUDIES=probe WBLR=1
         for s in 42 43 44; do rec SEED=$s STUDIES=budget WBLR=1; done ;;
    viewsphere|viewsphere2) rec STUDIES=probe WBLR=1
                for s in 42 43 44; do rec SEED=$s STUDIES=budget WBLR=1; done ;;
    coco) rec STUDIES=probe WBLR=0; rec STUDIES=probe WBLR=1
          rec STUDIES=budget WBLR=1 BUDGETS="250 500 1000 2500 5000 25000 50000"
          rec STUDIES=full ;;
  esac
  echo "$DS recovery: see outputs/recovery_jobs.txt ($(date))" | tee -a $LOG
fi
