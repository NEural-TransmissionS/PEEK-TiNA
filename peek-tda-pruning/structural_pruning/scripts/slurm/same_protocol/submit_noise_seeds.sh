#!/usr/bin/env bash
# Extra seeds to check which WSD / COCO / viewsphere differences are inside seed noise. Every run copies
# its dataset's existing protocol exactly (only the seed changes):
#   WSD        lr 1.25e-3, Ultralytics warmup bias lr      base runs/wsd71_scratch
#   COCO       lr 1.25e-4, 10k subset, Ultralytics warmup  base yolo26{n,s}.pt
#   viewsphere lr 1.25e-3, matched warmup (bias lr = lr0)  base weights/viewsphere_{n,s}.pt
# Studies (STUDIES env, default all four):
#   channel : seeds 45/46 for none / field_h1 / l2 / random at 10-50% (-> 5 seeds with 42/43/44),
#             into the existing <ds>_<m>_protect6_seeds_ft20
#   oneshot : field_h1 and l2 one-shot with BN re-estimation seeds 43-51 (scores fixed: they come from
#             the seed-42 scoring pass) -> <ds>_<m>_protect6_oneshot_seeds
#   blocks  : cut + fine-tune seeds 43/44 for the uncut model and every ranked cut (raw head, grid,
#             null; k1-3; a cut set two rankings share is trained once) -> results/*_ft_s<seed>.json
#   budgets : recovery budget sweep seeds 43/44, WSD + viewsphere (submit_recovery.sh, SEED env)
# Log: outputs/noise_seeds_jobs.txt; block plan: outputs/noise_seeds_blocks.tsv
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=$P/outputs/yolo26_tda
V=/shared/projects/nets/viewsphere-dataset/v1_wsd4
STUDIES=${STUDIES:-channel oneshot blocks budgets}
cd $P
export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW MOMENTUM=0.9 WARMUP_EPOCHS=1
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
PS="--partition=gpu1,gpu2,h200,h200_mig $X"   # COCO s: no 12 GB vdi GPUs
LOG=outputs/noise_seeds_jobs.txt; : > $LOG
BCOMMON="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --momentum 0.9 --warmup-epochs 1 --device 0"

setds() {  # ds m -> exports + PART, TL, BINIT, BFT
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

for ds in wsd coco viewsphere; do for m in n s; do
  setds $ds $m
  src=$O/${ds}_${m}_protect6_peek_ft20
  test -e $src/field_scores.pt || { echo "missing $src/field_scores.pt" >&2; exit 1; }
  if [[ " $STUDIES " == *" channel "* ]]; then
    export EPOCHS=20 BASIS=peek CRITERIA="field_h1 l2 random" OUT=$O/${ds}_${m}_protect6_seeds_ft20   # 0 none, 1-5, 6-10, 11-15
    mkdir -p $OUT; ln -sfn ../${ds}_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
    ids=""; for s in 45 46; do ids="$ids:$(prune ns-$ds-$m-s$s 0-15 $s $TL)"; done
    echo "$ds $m channel seeds 45/46: ${ids#:} agg=$(agg ns-$ds-$m-agg $ids)" | tee -a $LOG
  fi
  if [[ " $STUDIES " == *" oneshot "* ]]; then
    export EPOCHS=0 BASIS=peek CRITERIA="field_h1 l2" OUT=$O/${ds}_${m}_protect6_oneshot_seeds   # 1-5 field_h1, 6-10 l2
    mkdir -p $OUT; ln -sfn ../${ds}_${m}_protect6_peek_ft20/field_scores.pt $OUT/field_scores.pt
    ids=""; for s in 43 44 45 46 47 48 49 50 51; do ids="$ids:$(prune os-$ds-$m-s$s 1-10 $s 01:00:00)"; done
    echo "$ds $m one-shot seeds 43-51: ${ids#:} agg=$(agg os-$ds-$m-agg $ids)" | tee -a $LOG
  fi
done; done

if [[ " $STUDIES " == *" blocks "* ]]; then
  cd $S
  PLAN=$P/outputs/noise_seeds_blocks.tsv
  ../tina_venv/bin/python - > $PLAN <<'PY'
import json, sys
sys.path.insert(0, ".")
from topo_prune import cut_order, ranking_path
for ds in ("wsd", "coco", "viewsphere"):
    for m in "ns":
        have = {frozenset()}; print(f"{ds}\t{m}\tk0\t[]\t--k 0")
        head = json.loads(ranking_path(ds, m, "peek", "head", "").read_text())
        allr = json.loads(ranking_path(ds, m, "peek", "all", "norm").read_text())
        for k in (1, 2, 3):
            for lab, r, args in (("head", head, f"--k {k} --scope head"),
                                 ("grid", {**allr, **allr["grid"]}, f"--k {k} --scope all --norm grid"),
                                 ("null", {**allr, **allr["null"]}, f"--k {k} --scope all --norm null")):
                cut = frozenset(cut_order(r, "combined")[0][:k])
                if cut not in have:
                    have.add(cut); print(f"{ds}\t{m}\t{lab}-k{k}\t{sorted(cut)}\t{args}")
PY
  echo "block seeds 43/44: $(wc -l < $PLAN) cut sets x 2 seeds (plan $PLAN)" | tee -a $P/$LOG
  while IFS=$'\t' read ds m what cut args; do
    setds $ds $m
    for s in 43 44; do
      j=$(sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$TL --job-name=nb-$ds-$m-$what-s$s \
        --output=slurm/logs/noise_blocks_${ds}_${m}_${what}_s$s.out \
        --wrap=". ../tina_venv/bin/activate; python _train_one.py --dataset $ds --size $m $args --basis peek --init-weights $BINIT $BCOMMON $BFT --seed $s --skip-existing")
      echo "block $ds $m $what $cut s$s $j" >> $P/$LOG
    done
  done < $PLAN
  cd $P
fi

if [[ " $STUDIES " == *" budgets "* ]]; then
  for s in 43 44; do
    SEED=$s DATASETS="wsd viewsphere" STUDIES=budget WBLR=1 bash scripts/slurm/submit_recovery.sh
  done
  echo "budget seeds 43/44: see outputs/recovery_jobs.txt" | tee -a $LOG
fi
echo "done; $(grep -c . $LOG) lines in $LOG"
