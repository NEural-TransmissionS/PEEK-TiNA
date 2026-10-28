"""Evaluate viewsphere models on the v1_wsd4 test split, overall and by satellite / Sun phase / lit fraction.

Builds symlinked subset datasets under outputs/viewsphere_eval/subsets, runs Ultralytics val on each
<runs_dir>/{yolo26n,yolo26s}_<run_suffix>/weights/best.pt, writes outputs/viewsphere_eval/<out_csv>.
Usage: python eval_viewsphere.py [runs_dir] [run_suffix] [out_csv]
  defaults: runs/viewsphere_wsd4_scratch scratch_seed42 test_breakdown.csv
"""
import csv
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

D = Path("/shared/projects/nets/viewsphere-dataset/v1_wsd4")
V1 = Path("/shared/projects/nets/viewsphere-dataset/v1")
OUT = Path("/shared/projects/tina/PEEK-TiNA/outputs/viewsphere_eval")
REPO = Path("/shared/projects/tina/PEEK-TiNA")
NAMES = ["antenna", "body", "solar", "thruster"]


def lit_bin(f):
    return "lit<10%" if f < 0.10 else "lit10-50%" if f < 0.50 else "lit>=50%"


def frame_info():
    info = {}
    for sat in ["Clementine", "ICESatB", "Jason-1", "Sentinel-6"]:
        for r in csv.DictReader(open(V1 / sat / "visibility.csv")):
            info[f"{sat}_frame_{int(r['frame']):04d}"] = {
                "sat": sat, "phase": f"phase{int(float(r['phase_deg']))}",
                "lit": lit_bin(float(r["obj_lit_frac"]))}
    return info


def make_subsets(info):
    stems = sorted(p.stem for p in (D / "test" / "images").iterdir())
    groups = {"all": stems}
    for key in ("sat", "phase", "lit"):
        for s in stems:
            groups.setdefault(info[s][key], []).append(s)
    root = OUT / "subsets"
    shutil.rmtree(root, ignore_errors=True)
    yamls = {}
    for name, members in groups.items():
        for sub in ("images", "labels"):
            (root / name / sub).mkdir(parents=True)
        for s in members:
            (root / name / "images" / f"{s}.png").symlink_to((D / "test/images" / f"{s}.png").resolve())
            (root / name / "labels" / f"{s}.txt").symlink_to((D / "test/labels" / f"{s}.txt").resolve())
        y = root / f"{name}.yaml"
        y.write_text(f"path: {root / name}\ntrain: images\nval: images\nnc: 4\nnames: {NAMES}\n")
        yamls[name] = (y, len(members))
    return yamls


def main():
    args = sys.argv[1:] + ["runs/viewsphere_wsd4_scratch", "scratch_seed42", "test_breakdown.csv"][len(sys.argv) - 1:]
    runs, suffix, out_csv = REPO / args[0], args[1], args[2]
    yamls = make_subsets(frame_info())
    rows = []
    for m in ("yolo26n", "yolo26s"):
        model = YOLO(runs / f"{m}_{suffix}/weights/best.pt")
        for name, (y, n) in yamls.items():
            r = model.val(data=str(y), split="val", batch=16, imgsz=640, plots=False,
                          verbose=False, project=str(OUT / "val_runs"), name=f"{m}_{suffix}_{name}", exist_ok=True)
            row = {"model": m, "subset": name, "images": n,
                   "mAP50": round(r.box.map50, 4), "mAP50-95": round(r.box.map, 4)}
            for i, c in enumerate(NAMES):
                if i in r.box.ap_class_index:
                    row[f"AP50-95_{c}"] = round(r.box.maps[i], 4)
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
