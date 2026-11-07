#!/usr/bin/env bash
# Re-run of the whole same-protocol + recovery study on COCO-pretrained base models, queued at once (drafted
# 2026-11-06). Why: bases fine-tuned from Ultralytics' COCO weights beat the current ones (WSD from scratch,
# viewsphere fine-tuned from those) on val and test (results/base_models/coco_pilot_test.csv). Viewsphere renders
# are ~96% exactly-black pixels, so viewsphere bases AND their pruning fine-tunes train with pixel noise
# (NOISE / PIXEL_NOISE = 3 gray levels; structural_pruning/peek_rvc_pruning/pixel_noise.py; noise pilot checked by
# scripts/viewsphere/check_v2_stability.py). COCO is not re-run: its bases already are the COCO weights.
# For YOLO26 n/s and YOLOv5u n/s on WSD, viewsphere (v1) and viewsphere2:
#   bases   WSD: runs/wsd71_coco_ft (COCO pilot, done); v2: runs/viewsphere2_wsd4_coco_noise3 (noise pilot, done);
#           v1: 4 new fine-tunes from COCO weights with NOISE=3 -> runs/viewsphere_wsd4_coco_noise3
#   study   submit_yolov5.sh channel <ds> per architecture: channel grid seed 42 + seeds 43-46, one-shot random
#           masks 0-9 and BN seeds 43-51, block rankings (now with the L2 block baseline); it queues "blocks"
#           itself (cuts + 20-epoch fine-tunes incl. L2 cuts, recovery probes and budgets). v1 waits for its bases.
# With SUBMIT=1, before anything is queued, the old-base outputs are MOVED (same filesystem, instant) into a
# _base_scratch/ folder next to them, so the collector and the report read only the re-run:
#   PEEK-rvc-pruning/outputs/yolo26_tda/<ds>_*   topo-prune-shapes/{results,runs,ranking_peek}/<ds>_*
#   PEEK-rvc-pruning/outputs/{yolo26,yolov5}_<ds>_blocks_*.tsv
# and topo-prune-shapes/weights/<ds>_<m>.pt are pointed at the new bases (old targets go to the log).
#   bash submit_rerun_coco.sh            # dry run: fake sbatch into /tmp, nothing moved or linked; job counts
#   SUBMIT=1 bash submit_rerun_coco.sh   # moves, relinks, submits; log PEEK-rvc-pruning/outputs/rerun_coco_jobs.txt
# GPU packing (2026-11-07): every GPU job may run on gpu1, gpu2, h200, h200_mig (v1 bases) and the L40S vdi
# partitions (study jobs, as before); prune jobs 8 CPUs / 48G, block fine-tunes 32G, 8 dataloader workers, so all
# 4 GPUs per gpu2 node, all 8 per vdi node and 16 of 24 MIG slices can be in use. Aggregation: short or med.
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=$P/outputs/yolo26_tda
SB=$T/tina/scripts/viewsphere/train_viewsphere_finetune.slurm; V1=/shared/projects/nets/viewsphere-dataset/v1_wsd4
DSS="wsd viewsphere2 viewsphere"; NOISE=3; SUBMIT=${SUBMIT:-0}
declare -A BASE=([wsd]=$T/runs/wsd71_coco_ft [viewsphere2]=$T/runs/viewsphere2_wsd4_coco_noise3
                 [viewsphere]=$T/runs/viewsphere_wsd4_coco_noise3)
mid() { [ $1 = yolov5 ] && echo v5$2 || echo $2; }   # yolov5 n -> v5n, yolo26 n -> n

# 0. checks: the finished bases load (one v2 pilot checkpoint was once written corrupt), COCO inits exist
for ds in wsd viewsphere2; do for a in yolo26 yolov5; do for m in n s; do
  w=${BASE[$ds]}/$a${m}_ft_seed42/weights/best.pt
  if ! $T/tina_venv/bin/python -c "import sys,torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" \
       $w 2>/dev/null; then
    if [ "$SUBMIT" = 1 ]; then echo "base $w missing or unreadable, stopping"; exit 1; fi
    echo "WARNING (dry run goes on): base $w missing or unreadable"
  fi
done; done; done
for w in yolo26n yolo26s yolov5nu yolov5su; do [ -f $T/$w.pt ] || { echo "missing $T/$w.pt"; exit 1; }; done
[ -e ${BASE[viewsphere]} ] && { echo "${BASE[viewsphere]} exists: v1 bases already started? stopping"; exit 1; }
echo "checks done: WSD + v2 bases, COCO weights"

if [ "$SUBMIT" = 1 ]; then
  LOG=$P/outputs/rerun_coco_jobs.txt; echo "== $(date) submit_rerun_coco.sh" | tee -a $LOG
  # 1. move the old-base outputs aside
  for d in $O $S/results $S/runs $S/ranking_peek; do
    mkdir -p $d/_base_scratch
    for ds in $DSS; do for f in $d/${ds}_*; do [ -e "$f" ] && mv "$f" $d/_base_scratch/; done; done
    echo "moved $(ls $d/_base_scratch | wc -l) entries to $d/_base_scratch" | tee -a $LOG
  done
  mkdir -p $P/outputs/_base_scratch
  for ds in $DSS; do for f in $P/outputs/{yolo26,yolov5}_${ds}_blocks_*.tsv; do [ -e "$f" ] && mv "$f" $P/outputs/_base_scratch/; done; done
  # 2. weights/<ds>_<m>.pt -> the new bases (v1's exist once its base jobs finish)
  for ds in $DSS; do for a in yolo26 yolov5; do for m in n s; do
    l=$S/weights/${ds}_$(mid $a $m).pt
    echo "relink $l: was $(readlink $l || echo none)" >> $LOG
    ln -sfn ${BASE[$ds]}/$a${m}_ft_seed42/weights/best.pt $l
  done; done; done
  run() { "$@"; }
else
  # dry run: a fake sbatch counts tasks; outputs, plans and logs go to a scratch dir
  DRY=$(mktemp -d /tmp/rerun_dry.XXXX); LOG=$DRY/rerun.txt; mkdir -p $DRY/bin $DRY/out
  cat > $DRY/bin/sbatch <<'EOF'
#!/bin/bash
n=1; for a in "$@"; do case $a in --array=*) r=${a#--array=}; n=$(( ${r#*-} - ${r%-*} + 1 ));; esac; done
echo "$n noise=${PIXEL_NOISE:-0} $*" >> $DRY_DIR/sbatch.log; echo $(( 900000 + $(wc -l < $DRY_DIR/sbatch.log) ))
EOF
  chmod +x $DRY/bin/sbatch
  export PATH=$DRY/bin:$PATH DRY_DIR=$DRY O_ROOT=$DRY/out JOBLOG=$DRY/jobs.txt RECLOG=$DRY/rec.txt PLAN_DIR=$DRY
  echo "DRY RUN in $DRY (old-base rankings / weights stand in for the new ones)" | tee -a $LOG
  run() { "$@"; }
fi

# 3. viewsphere v1 bases: COCO weights -> v1_wsd4 with pixel noise (same recipe as the noise pilot)
v1_26=""; v1_5=""
for a in yolo26 yolov5; do for m in n s; do
  w=$T/$a$m.pt; [ $a = yolov5 ] && w=$T/$a${m}u.pt
  j=$(cd $T && sbatch --parsable --partition=gpu1,gpu2,h200,h200_mig --exclude=gpu06 --mem=48G --time=2-00:00:00 --job-name=rr-base-$(mid $a $m)-vs1 \
      --export=ALL,MODEL=$a$m,INIT=$w,DATA=$V1/data.yaml,PROJECT=${BASE[viewsphere]},NOISE=$NOISE $SB)
  [ $a = yolo26 ] && v1_26="$v1_26:$j" || v1_5="$v1_5:$j"
  echo "v1 base $a$m (COCO, noise $NOISE): $j" | tee -a $LOG
done; done

# 4. the study per dataset and architecture (dry run: v1 without its wait, so its stage is counted)
cd $P
for ds in $DSS; do
  d26=""; d5=""
  if [ $ds = viewsphere ] && [ "$SUBMIT" = 1 ]; then d26=afterok$v1_26; d5=afterok$v1_5; fi
  DEP=$d26 MODELS="n s" TAG=yolo26 NOISE_VS=$NOISE bash scripts/slurm/submit_yolov5.sh channel $ds | tee -a $LOG
  DEP=$d5 NOISE_VS=$NOISE bash scripts/slurm/submit_yolov5.sh channel $ds | tee -a $LOG
done

if [ "$SUBMIT" != 1 ]; then
  # the blocks stages the channel stages would queue, planned on the current (old-base) rankings
  for ds in $DSS; do
    MODELS="n s" TAG=yolo26 NOISE_VS=$NOISE bash scripts/slurm/submit_yolov5.sh blocks $ds > /dev/null
    NOISE_VS=$NOISE bash scripts/slurm/submit_yolov5.sh blocks $ds > /dev/null
  done
  grep -h "cut sets\|block fine-tune jobs" $DRY/jobs.txt | sed 's/^/  plan: /'
  python3 - $DRY/sbatch.log <<'EOF'
import sys, collections
c = collections.Counter(); nz = collections.Counter(); jobs = 0
for line in open(sys.argv[1]):
    n, noise, rest = line.split(" ", 2); n = int(n); jobs += 1
    name = next((a.split("=", 1)[1] for a in rest.split() if a.startswith("--job-name=")), "?")
    ds = next((d for d in ("viewsphere2", "viewsphere", "wsd", "vs1") if f"-{d}" in name), "?")
    nz[(ds, noise.split("=")[1])] += n
    kind = ("base" if name.startswith("rr-base") else "hand-off" if "-channel-" in name or "-blocks-" in name
            else "aggregate" if name.endswith("-agg") else "block rankings" if "rank-" in name
            else "recovery" if name.startswith(("rp", "rb", "rf")) else "block fine-tunes" if "b-" in name[:5]
            else "channel prune / scores")
    c[kind] += n
print("GPU / CPU tasks by kind:", dict(c)); print(f"total: {sum(c.values())} tasks in {jobs} submissions")
print("tasks by (dataset, PIXEL_NOISE at submit):", dict(nz))
EOF
fi
