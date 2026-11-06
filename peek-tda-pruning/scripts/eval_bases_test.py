"""Test-split mAP of the base models: the current ones vs the COCO-pretrained pilot (2026-11-06).

Starting points compared, per dataset and model (YOLO26n/s, YOLOv5n/s):
  WSD v71:        scratch       runs/wsd71_scratch/<m>_scratch_seed42        (current bases)
                  coco          runs/wsd71_coco_ft/<m>_ft_seed42             (pilot: from COCO weights)
  viewsphere v2:  wsd           runs/viewsphere2_wsd4_finetune_wsd/<m>_ft_seed42  (current: from WSD scratch)
                  coco          runs/viewsphere2_wsd4_coco_ft/<m>_ft_seed42       (pilot: from COCO weights)
Each best.pt (chosen on val) is evaluated once on the test split (WSD: 40 images; v2: 680).
Writes one row per model to the CSV given (default runs/base_models_test.csv).

    sbatch --gres=gpu:1 --wrap='tina_venv/bin/python tina/scripts/eval_bases_test.py'   (from PEEK-TiNA/)
"""
import csv
import sys
from pathlib import Path

from ultralytics import YOLO

T = Path("/shared/projects/tina/PEEK-TiNA")
DATA = {"wsd": T / "data/data.yaml", "viewsphere2": Path("/shared/projects/nets/viewsphere-dataset/v2_wsd4/data.yaml")}
RUNS = {("wsd", "scratch"): "wsd71_scratch/{m}_scratch_seed42", ("wsd", "coco"): "wsd71_coco_ft/{m}_ft_seed42",
        ("viewsphere2", "wsd"): "viewsphere2_wsd4_finetune_wsd/{m}_ft_seed42",
        ("viewsphere2", "coco"): "viewsphere2_wsd4_coco_ft/{m}_ft_seed42"}
MODELS = ("yolo26n", "yolo26s", "yolov5n", "yolov5s")

out = Path(sys.argv[1]) if len(sys.argv) > 1 else T / "runs/base_models_test.csv"
rows = []
for (ds, start), run in RUNS.items():
    for m in MODELS:
        w = T / "runs" / run.format(m=m) / "weights/best.pt"
        r = YOLO(str(w)).val(data=str(DATA[ds]), split="test", imgsz=640, batch=16, device=0, plots=False,
                             project=str(T / "runs/base_models_test_val"), name=f"{ds}_{start}_{m}", exist_ok=True)
        rows.append({"dataset": ds, "start": start, "model": m, "test_map50_95": round(r.box.map, 4),
                     "test_map50": round(r.box.map50, 4), "weights": str(w)})
        print(rows[-1], flush=True)
with open(out, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("wrote", out)
