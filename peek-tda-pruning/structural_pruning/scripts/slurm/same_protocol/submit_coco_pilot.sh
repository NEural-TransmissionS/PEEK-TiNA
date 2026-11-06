#!/bin/bash
# Pilot: do COCO-pretrained starting points give better base models than the current ones?
# Current bases: WSD = trained from scratch (wsd71_scratch), viewsphere v1/v2 = those fine-tuned.
# Pilot = 8 fine-tunes from Ultralytics' COCO weights (yolo26n/s.pt, yolov5n/su.pt), same recipe as the
# viewsphere fine-tunes (train_viewsphere_finetune.slurm: AdamW lr 1.25e-3, matched warmup bias lr,
# 1000 epochs, patience 100, batch 16, seed 42):
#   4 x WSD v71        -> runs/wsd71_coco_ft/<model>_ft_seed42           (compare wsd71_scratch)
#   4 x viewsphere v2  -> runs/viewsphere2_wsd4_coco_ft/<model>_ft_seed42 (compare viewsphere2_wsd4_finetune_wsd)
# Nothing downstream is queued: pruning runs keep using the current bases until we decide.
#   bash scripts/slurm/submit_coco_pilot.sh          # dry run: prints the sbatch lines
#   SUBMIT=1 bash scripts/slurm/submit_coco_pilot.sh # submits; job ids -> outputs/coco_pilot_jobs.txt
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning
SB=$T/tina/scripts/viewsphere/train_viewsphere_finetune.slurm
WSD=$T/data/data.yaml; V2=/shared/projects/nets/viewsphere-dataset/v2_wsd4/data.yaml
LOG=$P/outputs/coco_pilot_jobs.txt
for f in $SB $WSD $V2; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
cd $T
for ds in wsd vs2; do
  case $ds in wsd) DATA=$WSD; PROJ=$T/runs/wsd71_coco_ft;;
              vs2) DATA=$V2;  PROJ=$T/runs/viewsphere2_wsd4_coco_ft;; esac
  for m in yolo26n yolo26s yolov5n yolov5s; do
    case $m in yolov5*) w=$T/${m}u.pt;; *) w=$T/$m.pt;; esac  # YOLOv5u weights are yolov5{n,s}u.pt
    [ -f $w ] || { echo "missing weights $w"; exit 1; }
    [ -e $PROJ/${m}_ft_seed42 ] && { echo "exists: $PROJ/${m}_ft_seed42, skipping"; continue; }
    cmd=(sbatch --parsable --time=2-00:00:00 --job-name=coco-${m#yolo}-$ds
         --export=ALL,MODEL=$m,INIT=$w,DATA=$DATA,PROJECT=$PROJ $SB)
    if [ "${SUBMIT:-0}" = 1 ]; then
      j=$("${cmd[@]}"); echo "$(date +%F) coco pilot $ds $m init $(basename $w): $j" | tee -a $LOG
    else
      echo "DRY: ${cmd[*]}"
    fi
  done
done
