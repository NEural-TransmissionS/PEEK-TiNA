#!/usr/bin/env bash
# Submit the whole YOLO26 TDA pruning experiment as a dependency chain:
#   score (verifies the unpruned baseline first)  ->  prune array (criteria x ratios, per seed)  ->  aggregate
#
#   scripts/slurm/submit_yolo26_tda.sh
#   SBATCH_PARTITION=gpu SBATCH_ACCOUNT=mylab EPOCHS=20 SEEDS="0 1 2" scripts/slurm/submit_yolo26_tda.sh
#   DRY_RUN=1 scripts/slurm/submit_yolo26_tda.sh    # print the grid, submit nothing

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/yolo26_tda_env.sh"
activate_env
export REPO_ROOT
mkdir -p slurm_logs "$OUT"

# shellcheck disable=SC2086
N=$(python scripts/run_yolo26_tda_pruning.py grid --ratios $RATIOS --criteria $CRITERIA | tail -n 1)  # last line: Ultralytics may print a banner first
echo "model=$WEIGHTS data=$DATA epochs=$EPOCHS seeds=[$SEEDS] -> $N runs per seed, outputs in $OUT"
if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi

SB="scripts/slurm/yolo26_tda.sbatch"
score=""
for stage in $SCORE_STAGE; do  # score and/or score-field, in parallel; empty for l2/random-only grids
  score="$score:$(sbatch --parsable --export=ALL --job-name="yolo26-tda-$stage" --time=04:00:00 "$SB" "$stage")"
done
echo "score:     ${score#:}"
after=()
if [ -n "$score" ]; then after=(--dependency=afterok"$score"); fi

deps=""
for seed in $SEEDS; do
  arr=$(sbatch --parsable --export=ALL,SEED="$seed" --job-name="yolo26-tda-prune-s$seed" \
    --array=0-$((N - 1)) "${after[@]}" "$SB" prune)
  echo "prune s$seed: $arr (array 0-$((N - 1)))"
  deps="$deps:$arr"
done

agg=$(sbatch --parsable --export=ALL --job-name=yolo26-tda-aggregate --gres=none --cpus-per-task=2 \
  --mem=8G --time=00:30:00 --dependency=afterany"$deps" "$SB" aggregate)
echo "aggregate: $agg"
echo "Results -> $OUT/results.csv and $OUT/map_vs_ratio.png (squeue -u \$USER to watch)"
