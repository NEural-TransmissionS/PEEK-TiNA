"""Is a viewsphere v2 base model numerically stable? Compares the noise-augmented pilot with the plain one.

Background (2026-11-06): v2 renders are ~96% exactly-black pixels; plain-trained models react to tiny noise
on the background, so fp16 vs fp32, epoch-to-epoch val and BN re-estimation swing (see train_noise_aug.py).
Per model, for runs/viewsphere2_wsd4_coco_ft (plain) and runs/viewsphere2_wsd4_coco_noise3 (NOISE=3):
  map_{val,test}_{fp16,fp32}   best.pt mAP50-95 at each precision (stable model: fp16 ~ fp32)
  bg_noise                     max class-score change from 1e-3 noise on the black background (8 val images)
  bn_seed_min/max              val mAP50-95 (fp32) after BN re-estimation on 128 train images, calib seeds
                               43-47, as run_yolo26_tda_pruning does for one-shot pruning (bn_calibration_images:
                               noise3 models get the training noise on those images, plain ones clean images;
                               first run, clean for both: noise_pilot_check_cleanbn.csv)
  curve_jump                   median |change| of val mAP50-95 between epochs; win_min = lowest val within
                               +-10 epochs of the best epoch (spiky curve: far below the best)
Writes runs/v2_numerics/noise_pilot_check.csv and prints it. With argument "wsd": the same checks on the WSD
COCO-start bases (runs/wsd71_coco_ft, no noise) -> runs/v2_numerics/wsd_coco_check.csv.

    sbatch --partition=gpu1 --gres=gpu:1 --wrap='tina_venv/bin/python tina/scripts/viewsphere/check_v2_stability.py [wsd]'
"""
import copy
import csv
import glob
import os
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

T = Path("/shared/projects/tina/PEEK-TiNA")
sys.path.insert(0, str(T / "tina/structural_pruning/scripts"))
import run_yolo26_tda_pruning as rp  # noqa: E402  (calibration_images; puts peek_rvc_pruning on the path)
from peek_rvc_pruning import yolo_tda  # noqa: E402

SETS = {  # argv[1]: data.yaml, {run kind: run dir}, output csv
    "v2": ("/shared/projects/nets/viewsphere-dataset/v2_wsd4/data.yaml",
           {"plain": T / "runs/viewsphere2_wsd4_coco_ft", "noise3": T / "runs/viewsphere2_wsd4_coco_noise3"},
           T / "runs/v2_numerics/noise_pilot_check.csv"),
    "wsd": (str(T / "data/data.yaml"), {"plain": T / "runs/wsd71_coco_ft"}, T / "runs/v2_numerics/wsd_coco_check.csv"),
}
DATA, RUNS, OUT = SETS[sys.argv[1] if len(sys.argv) > 1 else "v2"]
MODELS = ("yolo26n", "yolo26s", "yolov5n", "yolov5s")


def val(w, model=None, split="val", half=False):
    y = YOLO(str(w))
    if model is not None:
        y.model = copy.deepcopy(model)
    r = y.val(data=DATA, split=split, imgsz=640, batch=16, device=0, half=half, plots=False, verbose=False,
              project=str(OUT.parent / "val"), name="check", exist_ok=True)
    return round(float(r.box.map), 4)


def bg_noise(w):
    fs = sorted(glob.glob(str(Path(DATA).parent / "valid/images/*")))[::85][:8]
    x = torch.tensor(np.stack([cv2.resize(cv2.imread(f), (640, 640), interpolation=cv2.INTER_NEAREST)[:, :, ::-1]
                               .transpose(2, 0, 1) / 255.0 for f in fs]), dtype=torch.float32).cuda()
    black = (x == 0).all(1, keepdim=True).float()
    m = YOLO(str(w)).model.float().cuda().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        a, b = m(x), m(x + torch.randn_like(x) * 1e-3 * black)
    a, b = (y[0] if isinstance(y, (list, tuple)) else y for y in (a, b))
    return round((a[:, 4:].max(1).values - b[:, 4:].max(1).values).abs().max().item(), 3)


def bn_seeds(w):
    out = []
    for seed in range(43, 48):
        net = YOLO(str(w)).model.float().cuda()
        yolo_tda.recalibrate_bn(net, rp.bn_calibration_images(DATA, 128, 640, seed))  # noise per PIXEL_NOISE
        out.append(val(w, net.cpu()))
    return min(out), max(out)


def curve(run):
    v = [float(r["metrics/mAP50-95(B)"]) for r in csv.DictReader(open(run / "results.csv"))]
    i = v.index(max(v))
    return round(st.median(abs(a - b) for a, b in zip(v, v[1:])), 4), round(min(v[max(0, i - 10):i + 11]), 4), i + 1


rows = []
for kind, root in RUNS.items():
    os.environ["PIXEL_NOISE"] = "3" if kind == "noise3" else "0"  # BN calibration as in the re-run
    for m in MODELS:
        run = root / f"{m}_ft_seed42"
        w = run / "weights/best.pt"
        try:
            torch.load(w, map_location="cpu", weights_only=False)
        except Exception:  # plain yolo26s best.pt was written corrupt; last.pt is the end of training
            w = run / "weights/last.pt"
        jump, win_min, best_ep = curve(run)
        bmin, bmax = bn_seeds(w)
        rows.append({"run": kind, "model": m, "weights": w.name, "best_epoch": best_ep,
                     "map_val_fp16": val(w, half=True), "map_val_fp32": val(w),
                     "map_test_fp16": val(w, split="test", half=True), "map_test_fp32": val(w, split="test"),
                     "bg_noise": bg_noise(w), "bn_seed_min": bmin, "bn_seed_max": bmax,
                     "curve_jump": jump, "win_min": win_min})
        print(rows[-1], flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("wrote", OUT)
