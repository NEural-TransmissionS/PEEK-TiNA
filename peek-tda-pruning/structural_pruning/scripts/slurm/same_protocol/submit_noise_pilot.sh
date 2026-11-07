#!/bin/bash
# Pilot (2026-11-06): does pixel-noise augmentation fix the viewsphere v2 instability?
# v2 renders are ~96% exactly-black pixels; models trained on them swing with fp16/fp32, epoch and BN
# re-estimation (v2 YOLOv5n COCO start: 0.48 val in fp16, 0.06-0.11 in fp32). This re-runs the 4 v2
# COCO-start fine-tunes of submit_coco_pilot.sh with NOISE=3 (train_noise_aug.py: per-image Gaussian
# sigma ~ U(0, 3/255) on training batches only), same recipe otherwise:
#   4 x viewsphere v2 -> runs/viewsphere2_wsd4_coco_noise3/<model>_ft_seed42 (compare viewsphere2_wsd4_coco_ft)
#   bash .../submit_noise_pilot.sh          # dry run: prints the sbatch lines
#   SUBMIT=1 bash .../submit_noise_pilot.sh # submits; job ids -> outputs/noise_pilot_jobs.txt
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning
SB=$T/tina/scripts/viewsphere/train_viewsphere_finetune.slurm
V2=/shared/projects/nets/viewsphere-dataset/v2_wsd4/data.yaml
PROJ=$T/runs/viewsphere2_wsd4_coco_noise3; LOG=$P/outputs/noise_pilot_jobs.txt
for f in $SB $V2 $T/tina/scripts/viewsphere/train_noise_aug.py; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
cd $T
for m in yolo26n yolo26s yolov5n yolov5s; do
  case $m in yolov5*) w=$T/${m}u.pt;; *) w=$T/$m.pt;; esac
  [ -f $w ] || { echo "missing weights $w"; exit 1; }
  [ -e $PROJ/${m}_ft_seed42 ] && { echo "exists: $PROJ/${m}_ft_seed42, skipping"; continue; }
  cmd=(sbatch --parsable --time=2-00:00:00 --job-name=noise-${m#yolo}-vs2
       --export=ALL,MODEL=$m,INIT=$w,DATA=$V2,PROJECT=$PROJ,NOISE=3 $SB)
  if [ "${SUBMIT:-0}" = 1 ]; then
    j=$("${cmd[@]}"); echo "$(date +%F) noise pilot vs2 $m: $j" | tee -a $LOG
  else
    echo "DRY: ${cmd[*]}"
  fi
done
