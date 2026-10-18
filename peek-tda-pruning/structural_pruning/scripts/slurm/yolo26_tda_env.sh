# Shared settings for the YOLO26 TDA pruning SLURM jobs. Sourced by the other
# scripts in this directory; override any value by exporting it first, e.g.
#   EPOCHS=20 RATIOS="0.1 0.3 0.5" scripts/slurm/submit_yolo26_tda.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export WEIGHTS="${WEIGHTS:-yolo26n.pt}"          # yolo26{n,s,m,l,x}.pt, resolved in $REPO_ROOT
export DATA="${DATA:-coco.yaml}"                  # any Ultralytics detection dataset yaml
export IMGSZ="${IMGSZ:-640}"
export EPOCHS="${EPOCHS:-10}"                     # fine-tune epochs after pruning
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"        # Ultralytics LR warmup during the fine-tune
export BATCH="${BATCH:-32}"
export RATIOS="${RATIOS:-0.1 0.2 0.3 0.4 0.5}"    # per-layer channel pruning ratios
export CRITERIA="${CRITERIA:-h0 h1 total l2 random field_h0 field_h1 field_total}"  # grid criteria
export SCORE_STAGE="${SCORE_STAGE:-score score-field}"  # score: per-channel h0/h1/total; score-field: field_*
export BASIS="${BASIS:-peek}"                     # topology read from PEEK maps (activation = archived variant)
export SEEDS="${SEEDS:-0}"                        # e.g. "0 1 2" for error bars
export PROTECT_LAYERS="${PROTECT_LAYERS:-6}"         # never prune top-level layers 0..N-1 (stem..stride-16)
export CALIB_IMAGES="${CALIB_IMAGES:-128}"
export MAX_SIDE="${MAX_SIDE:-64}"
export OUT="${OUT:-$REPO_ROOT/outputs/yolo26_tda/${WEIGHTS%.pt}_protect${PROTECT_LAYERS}}"
export DATASETS_DIR="${DATASETS_DIR:-$(cd "$REPO_ROOT/.." && pwd)/datasets}"

# Keep Ultralytics settings/downloads inside the repo instead of ~/.config
export YOLO_CONFIG_DIR="$REPO_ROOT/.ultralytics"

# Cluster-specific environment (uncomment/edit for your site):
# module load cuda/12.8
# module load python/3.10

activate_env() {
  cd "$REPO_ROOT"
  # shellcheck disable=SC1091
  source "${VENV:-$REPO_ROOT/.venv}/bin/activate"
}

run_py() {
  python scripts/run_yolo26_tda_pruning.py "$@" \
    --weights "$WEIGHTS" --data "$DATA" --imgsz "$IMGSZ" --output-dir "$OUT"
}
