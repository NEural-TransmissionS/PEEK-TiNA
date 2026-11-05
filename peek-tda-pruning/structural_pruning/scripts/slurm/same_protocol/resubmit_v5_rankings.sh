#!/usr/bin/env bash
# Re-run the YOLOv5u block rankings (2026-11-03). The first ones (161021, 161051, 162612, 162642, 163907, 163937)
# failed in xaikit/xai.py class_scores (KeyError 'one2many': YOLOv5u's plain Detect head has no one2many key;
# fixed), so the three "blocks" stages (161052, 162643, 163938) can never start. This cancels those, re-runs the
# rankings and queues each dataset's blocks stage after them, as submit_yolov5.sh channel would have.
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long --exclude=gpu06"
LOG=$P/outputs/yolov5_jobs.txt
scancel 161052 162643 163938 || true
for ds in coco wsd viewsphere; do
  ranks=""
  for m in v5n v5s; do
    r=$(cd $S && sbatch --parsable $PW --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --job-name=v5rank-$ds-$m \
      --output=slurm/logs/v5rank_${ds}_$m.out \
      --wrap=". ../tina_venv/bin/activate; python run_experiment.py --dataset $ds --size $m --ks --basis peek --scope head --rank-split train; python run_experiment.py --dataset $ds --size $m --ks --basis peek --scope all --norms --rank-split train")
    ranks="$ranks:$r"
  done
  b=$(sbatch --parsable $PW --gres=gpu:1 --cpus-per-task=2 --mem=4G --time=01:00:00 --export=NONE \
    --dependency=afterok$ranks --job-name=v5-blocks-$ds --output=$P/slurm_logs/v5-submit-blocks-$ds-%j.out \
    --wrap="bash -lc 'cd $P && bash scripts/slurm/submit_yolov5.sh blocks $ds'")
  echo "$ds: rankings re-run ${ranks#:}; blocks stage $b (resubmitted 2026-11-03 after the xai.py fix)" | tee -a $LOG
done
