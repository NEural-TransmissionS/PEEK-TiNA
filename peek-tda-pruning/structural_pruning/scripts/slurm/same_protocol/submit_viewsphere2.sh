#!/usr/bin/env bash
# Viewsphere v2 (all 17 approved satellites, rwhite 2026-11-02) through the whole study, YOLO26 and YOLOv5u
# (drafted 2026-11-03). Dataset id "viewsphere2" = v2 relabelled to WSD's 4 classes (v2_wsd4), split and
# filtered like v1_wsd4: valid = views 3, 14; test = views 6, 11; train = the rest minus frames under 10% lit
# (4,079 / 680 / 680 images, 255 train batches per epoch).
#   1. dataset   remap_wsd4.py v2 + make_splits.py v2 -> /shared/projects/nets/viewsphere-dataset/v2_wsd4 (no jobs)
#   2. wiring    topo-prune-shapes/data/viewsphere2 (rankings read data/viewsphere2/images/train)
#   3. bases     the WSD scratch models fine-tuned on v2_wsd4 like v1 (train_viewsphere_finetune.slurm, DATA /
#                PROJECT): yolo26n/s now; yolov5n/s after their WSD base trainings (V5_WSD_JOBS) if still running
#   4. study     submit_yolov5.sh channel viewsphere2 for each architecture, held until its bases finish; it
#                queues the block + recovery stage itself (protocol = viewsphere's: lr 1.25e-3, matched warmup)
# Log: outputs/viewsphere2_jobs.txt (+ outputs/{yolo26,yolov5}_jobs.txt)
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes
VD=/shared/projects/nets/viewsphere-dataset; V2=$VD/v2_wsd4
V5_WSD_JOBS=${V5_WSD_JOBS:-160891:160893}   # yolov5n / yolov5s WSD scratch trainings
LOG=$P/outputs/viewsphere2_jobs.txt; : > $LOG

# 1. dataset
cd $T/tina/scripts/viewsphere
python3 remap_wsd4.py v2 && python3 make_splits.py v2
for s in train valid test; do echo "v2_wsd4 $s: $(ls $V2/$s/images | wc -l) images" | tee -a $LOG; done

# 2. topo-prune-shapes wiring
mkdir -p $S/data/viewsphere2/images
ln -sfn $V2/data.yaml $S/data/viewsphere2/data.yaml
ln -sfn $V2/train/images $S/data/viewsphere2/images/train
ln -sfn $V2/valid/images $S/data/viewsphere2/images/val

# 3. base fine-tunes
cd $T/tina
dep_v5=""
for j in ${V5_WSD_JOBS//:/ }; do
  st=$(sacct -j $j -n -X -o State | tr -d ' ')
  [ "$st" = COMPLETED ] || dep_v5="$dep_v5:$j"
done
ids26=""; ids5=""
for a in yolo26 yolov5; do for m in n s; do
  d=""; [ $a = yolov5 ] && [ -n "$dep_v5" ] && d="--dependency=afterok$dep_v5"
  j=$(sbatch --parsable $d --job-name=${a#yolo}$m-vs2ft \
      --export=ALL,MODEL=$a$m,DATA=$V2/data.yaml,PROJECT=$T/runs/viewsphere2_wsd4_finetune_wsd \
      scripts/viewsphere/train_viewsphere_finetune.slurm)
  [ $a = yolo26 ] && ids26="$ids26:$j" || ids5="$ids5:$j"
  echo "base $a$m on v2_wsd4: $j ${d}" | tee -a $LOG
done; done

# 4. the study, per architecture, once its bases exist
cd $P
DEP=afterok$ids26 MODELS="n s" TAG=yolo26 bash scripts/slurm/submit_yolov5.sh channel viewsphere2 | tee -a $LOG
DEP=afterok$ids5 bash scripts/slurm/submit_yolov5.sh channel viewsphere2 | tee -a $LOG
