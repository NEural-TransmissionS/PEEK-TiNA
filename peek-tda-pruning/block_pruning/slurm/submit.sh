#!/usr/bin/env bash
# Copy this directory to the server, create the venv (see README), then:
#   bash slurm/submit.sh
# Chains: data prep -> the 8-task sweep -> summarize.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

ACT='source ${VENV:-.venv}/bin/activate 2>/dev/null || true'

PREP=$(sbatch --parsable --job-name=tp-prep --time=00:20:00 --mem=8G \
  --output=slurm/logs/prep.out \
  --wrap="$ACT; python make_shapes.py")
echo "prep      $PREP"

SWEEP=$(sbatch --parsable --dependency=afterok:"$PREP" slurm/sweep.slurm)
echo "sweep     $SWEEP"

SUM=$(sbatch --parsable --dependency=afterany:"$SWEEP" --job-name=tp-sum \
  --time=00:20:00 --mem=8G --output=slurm/logs/summarize.out \
  --wrap="$ACT; python summarize.py")
echo "summarize $SUM"
