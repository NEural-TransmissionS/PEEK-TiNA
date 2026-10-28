#!/usr/bin/env bash
# Recovery studies on WSD + COCO, same recipe as the same-protocol runs (AdamW momentum 0.9, batch 16,
# lr 1.25e-3 WSD / 1.25e-4 COCO, seed 42). Subset: unpruned + PEEK field H1 / L2 / random at 10/30/50%
# (channel), uncut + raw-head / grid / null rankings at k=1-3 (blocks).
#   probe  : the 20-epoch schedule, evaluated every 50 training batches (iter_probe.py). WSD: whole run,
#            full val. COCO: first 2,500 batches (4 epochs), fixed 500-image val2017 slice.
#   WBLR=1 : the warmup's bias lr starts at lr0 instead of Ultralytics' 0.1 (outputs / tags get _wblr);
#            the standard for COCO from 2026-10-22 (0.1 crashes every COCO model in the first ~50 batches).
#   budget : separate fine-tunes of B batches (63 per WSD epoch, 625 per COCO epoch); whole epochs,
#            COCO budgets under 2 epochs train on the first 40% / 80% of the 10k subset (= Ultralytics
#            fraction, as own datasets coco_ft10k_f0.4/_f0.8 so each has its own label cache); warmup 5% of the run
#            (Ultralytics floor: 100 batches); mosaic off for the last half.
#   full   : COCO only, matched warmup: the same fine-tune on all of train2017 (118,287 images, 7,393 batches
#            per epoch) for 3 and 7 epochs (22k / 52k batches, vs the 10k subset's 25k / 50k budgets): is the
#            subset the ceiling? Unpruned + the channel subset, and two block cuts per model
#            (n: null [16], null [13 16 19]; s: grid [16], grid [2 8 16]). Outputs coco_<m>_protect6_full_e<E>_wblr.
# What runs is set by the environment (defaults = the first submission):
#   viewsphere: like WSD (whole run, full valid), base = weights/viewsphere_{n,s}.pt (WSD -> viewsphere
#            fine-tunes), 58 batches per epoch; added 2026-10-27, submitted via submit_viewsphere_phase2.sh.
#   STUDIES="probe budget"  DATASETS="wsd coco"  BUDGETS="250 500 1000 2500 5000"  WBLR=0
# e.g. WBLR=1 STUDIES=probe bash submit_recovery.sh;  STUDIES=budget DATASETS=coco BUDGETS="25000 50000" bash ...
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=$P/outputs/yolo26_tda
STUDIES=${STUDIES:-probe budget}; DATASETS=${DATASETS:-wsd coco}; BUDGETS=${BUDGETS:-250 500 1000 2500 5000}; WBLR=${WBLR:-0}
cd $P
export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 SEED=42 OPTIMIZER=AdamW MOMENTUM=0.9
export RATIOS="0.1 0.3 0.5" CRITERIA="field_h1 l2 random"   # array 0 none, 1-3 field_h1, 4-6 l2, 7-9 random
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"   # wide
PL="--partition=gpu1,gpu2,h200,h200_mig,vdi-long $X"           # wide, > 8 h
PS="--partition=gpu1,gpu2,h200,h200_mig $X"                    # COCO s fine-tunes: no 12 GB vdi
LOG=outputs/recovery_jobs.txt
# B -> "epochs fraction"
declare -A WSD_B=([250]="4 1" [500]="8 1" [1000]="16 1" [2500]="40 1" [5000]="79 1")
declare -A VS_B=([250]="4 1" [500]="9 1" [1000]="17 1" [2500]="43 1" [5000]="86 1")
declare -A COCO_B=([250]="1 0.4" [500]="1 0.8" [1000]="2 0.8" [2500]="4 1" [5000]="8 1" [25000]="40 1" [50000]="80 1")

setup() {  # ds m -> exports for that model
  if [ $1 = wsd ]; then
    export WEIGHTS=$T/runs/wsd71_scratch/yolo26${2}_scratch_seed42/weights/best.pt DATA=$T/data/data.yaml LR0=0.00125
    unset FT_DATA; BW=weights/wsd_$2.pt; BLR=0.00125; BDATA=""; PART=$PW
  elif [ $1 = viewsphere ]; then
    export WEIGHTS=$S/weights/viewsphere_$2.pt DATA=/shared/projects/nets/viewsphere-dataset/v1_wsd4/data.yaml LR0=0.00125
    unset FT_DATA; BW=weights/viewsphere_$2.pt; BLR=0.00125; BDATA=""; PART=$PW
  else
    export WEIGHTS=$T/yolo26$2.pt DATA=$P/configs/coco/coco.yaml LR0=0.000125 FT_DATA=$P/configs/coco/coco_ft10k.yaml
    BW=weights/coco_$2.pt; BLR=0.000125; BDATA="--data $P/configs/coco/coco_ft10k.yaml"
    PART=$([ $2 = s ] && echo "$PS" || echo "$PW")
  fi
}
block() {  # name time args...
  local n=$1 tl=$2; shift 2
  (cd $S && sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$tl --job-name=$n \
    --output=slurm/logs/recovery_$n.out \
    --wrap=". ../tina_venv/bin/activate; python _train_one.py $* --basis peek --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --momentum 0.9 --device 0 --skip-existing")
}
cuts() {  # ds m -> one "args" line per cut set (uncut, raw head, grid, null; COCO s null k1 = grid k1)
  echo "--k 0"
  for k in 1 2 3; do
    echo "--k $k --scope head"; echo "--k $k --scope all --norm grid"
    [ "$1$2$k" = cocos1 ] || echo "--k $k --scope all --norm null"
  done
}

for ds in $DATASETS; do for m in n s; do
  setup $ds $m
  src=$O/${ds}_${m}_protect6_peek_ft20
  if [[ " $STUDIES " == *" probe "* ]]; then
    v=""; wb=""; unset WARMUP_BIAS_LR
    if [ "$WBLR" = 1 ]; then v=_wblr; wb="--warmup-bias-lr $BLR"; export WARMUP_BIAS_LR=$LR0; fi
    export OUT=$O/${ds}_${m}_protect6_probe$v EPOCHS=20 WARMUP_EPOCHS=1 PROBE_EVERY=50; unset FRACTION CLOSE_MOSAIC
    mkdir -p $OUT; for f in tda_scores.pt field_scores.pt; do ln -sfn $src/$f $OUT/$f; done
    if [ $ds = coco ]; then export PROBE_MAX=2500 FT_DATA=$P/configs/coco/coco_ft10k_val500.yaml; else unset PROBE_MAX; fi
    j=$(sbatch --parsable $PART --export=ALL --job-name=rp$v-$ds-$m --array=0-9 --time=03:00:00 $SB prune)
    echo "$ds $m probe$v channel: $j" | tee -a $LOG
    pv="--probe-every 50"; [ $ds = coco ] && pv="$pv --probe-max-batches 2500 --data $P/configs/coco/coco_ft10k_val500.yaml" || pv="$pv $BDATA"
    ids=""; while read -r c; do
      ids="$ids $(block rpb$v-$ds-$m-$(echo $c | tr -d ' -') 03:00:00 --dataset $ds --size $m $c --init-weights $BW --epochs 20 --lr0 $BLR --warmup-epochs 1 $wb --tag _probe$v $pv)"
    done < <(cuts $ds $m)
    echo "$ds $m probe$v blocks:$ids" | tee -a $LOG
    unset PROBE_EVERY PROBE_MAX WARMUP_BIAS_LR
    [ $ds = coco ] && export FT_DATA=$P/configs/coco/coco_ft10k.yaml
  fi
  if [[ " $STUDIES " == *" budget "* ]]; then
    v=""; wb=""; unset WARMUP_BIAS_LR
    if [ "$WBLR" = 1 ]; then v=_wblr; wb="--warmup-bias-lr $BLR"; export WARMUP_BIAS_LR=$LR0; fi
    for b in $BUDGETS; do
      if [ $ds = wsd ]; then read -r e f <<< "${WSD_B[$b]}"; elif [ $ds = viewsphere ]; then read -r e f <<< "${VS_B[$b]}"; else read -r e f <<< "${COCO_B[$b]}"; fi
      w=$(python3 -c "print(round(0.05 * $e, 3))"); cm=$(( (e + 1) / 2 ))
      # time limit: COCO ~3 min/epoch (train + full-val) for n, 6 for s, plus 2 h; WSD is quick
      if [ $ds = coco ]; then mpe=$([ $m = s ] && echo 6 || echo 3); h=$(( (e * mpe + 59) / 60 + 2 )); else h=2; fi
      tl=$(printf "%02d:00:00" $h); bp=$PART
      [ $h -gt 8 ] && [ "$PART" = "$PW" ] && bp=$PL
      export OUT=$O/${ds}_${m}_protect6_budget$b$v EPOCHS=$e WARMUP_EPOCHS=$w CLOSE_MOSAIC=$cm
      bd=$BDATA; [ $ds = coco ] && export FT_DATA=$P/configs/coco/coco_ft10k.yaml
      if [ "$f" != 1 ]; then export FT_DATA=$P/configs/coco/coco_ft10k_f$f.yaml; bd="--data $FT_DATA"; fi
      mkdir -p $OUT; for x in tda_scores.pt field_scores.pt; do ln -sfn $src/$x $OUT/$x; done
      j=$(sbatch --parsable $bp --export=ALL --job-name=rb$b$v-$ds-$m --array=0-9 --time=$tl $SB prune)
      ids=""; while read -r c; do
        ids="$ids $(PART=$bp block rbb$b$v-$ds-$m-$(echo $c | tr -d ' -') $tl --dataset $ds --size $m $c --init-weights $BW --epochs $e --lr0 $BLR --warmup-epochs $w --close-mosaic $cm $wb $bd --tag _b$b$v)"
      done < <(cuts $ds $m)
      echo "$ds $m budget$v $b (epochs $e, fraction $f, warmup $w, close_mosaic $cm, $tl): channel $j blocks:$ids" | tee -a $LOG
    done
    unset CLOSE_MOSAIC WARMUP_BIAS_LR
    [ $ds = coco ] && export FT_DATA=$P/configs/coco/coco_ft10k.yaml
  fi
  if [[ " $STUDIES " == *" full "* ]] && [ $ds = coco ]; then
    export WARMUP_BIAS_LR=$LR0 FT_DATA=$P/configs/coco/coco.yaml; unset PROBE_EVERY PROBE_MAX
    for e in ${FULL_EPOCHS:-3 7}; do
      w=$(python3 -c "print(round(0.05 * $e, 3))"); cm=$(( (e + 1) / 2 )); tl=$(printf "%02d:00:00" $(( e + 3 )))
      export OUT=$O/coco_${m}_protect6_full_e${e}_wblr EPOCHS=$e WARMUP_EPOCHS=$w CLOSE_MOSAIC=$cm
      mkdir -p $OUT; for x in tda_scores.pt field_scores.pt; do ln -sfn $src/$x $OUT/$x; done
      j=$(sbatch --parsable $PART --export=ALL --job-name=rf$e-$ds-$m --array=0-9 --time=$tl $SB prune)
      if [ $m = n ]; then cuts_full="--k 1 --scope all --norm null|--k 3 --scope all --norm null"
      else cuts_full="--k 1 --scope all --norm grid|--k 3 --scope all --norm grid"; fi
      ids=""; while IFS= read -r c; do
        ids="$ids $(block rfb$e-$ds-$m-$(echo $c | tr -d ' -') $tl --dataset $ds --size $m $c --init-weights $BW --epochs $e --lr0 $BLR --warmup-epochs $w --close-mosaic $cm --warmup-bias-lr $BLR --data $P/configs/coco/coco.yaml --tag _full_e${e}_wblr)"
      done < <(echo "$cuts_full" | tr '|' '\n')
      echo "$ds $m full train2017 $e epochs (warmup $w, close_mosaic $cm, $tl): channel $j blocks:$ids" | tee -a $LOG
    done
    unset WARMUP_BIAS_LR CLOSE_MOSAIC
  fi
done; done
