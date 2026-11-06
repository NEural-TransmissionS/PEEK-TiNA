#!/usr/bin/env bash
# One-time setup on the cluster LOGIN node (compute nodes are often offline):
# venv + deps, YOLO26 weights, and the detection dataset.
#
#   scripts/slurm/setup_yolo26_tda.sh              # everything (COCO is ~20 GB)
#   SKIP_DATA=1 scripts/slurm/setup_yolo26_tda.sh  # dataset already in $DATASETS_DIR

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/yolo26_tda_env.sh"
cd "$REPO_ROOT"

[ -d .venv ] || ./setup.sh --with-gpu
activate_env
python -m pip install -q -r requirements.txt

mkdir -p "$YOLO_CONFIG_DIR" "$DATASETS_DIR"
yolo settings datasets_dir="$DATASETS_DIR" sync=False >/dev/null

python - <<PY
from ultralytics import YOLO
YOLO("$WEIGHTS")          # downloads into the repo root
YOLO("yolo26n.pt")        # Ultralytics' AMP self-check loads this one
PY

if [ "${SKIP_DATA:-0}" != "1" ]; then
  python -c "from ultralytics.data.utils import check_det_dataset; print(check_det_dataset('$DATA')['path'])"
fi

python -c "import torch, gudhi, torch_pruning, ultralytics; from peek_rvc_pruning import yolo_tda; print('ok: torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| gudhi', gudhi.__version__)"
echo "Setup done. Submit with: scripts/slurm/submit_yolo26_tda.sh"
