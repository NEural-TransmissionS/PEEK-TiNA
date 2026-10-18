#!/usr/bin/env bash
# Local sequential run (no SLURM). One process per (dataset, size); each spawns
# its own per-k training subprocess.
set -euo pipefail
cd "$(dirname "$0")"

python make_shapes.py
for d in circles triangles squares mixed; do
  for s in n s; do
    python run_experiment.py --dataset "$d" --size "$s" \
      --epochs "${EPOCHS:-60}" --device "${DEVICE:-0}"
  done
done
python summarize.py
