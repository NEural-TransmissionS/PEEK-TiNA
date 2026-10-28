"""Build viewsphere-dataset/v1_wsd4: the v1 part labels remapped to WSD's 4 classes.

WSD classes: 0 antenna, 1 body, 2 solar, 3 thruster.
- Laser, radiator, startracker, telescope and instrument parts become body.
- The whole-satellite class (old group 0) is dropped.
- Det labels: all body-class boxes in a frame merge into one union box (WSD has one
  body box per image); antenna/solar/thruster keep one box per part.
- Seg labels: one polygon per part, class number remapped.
Images, masks and per-frame metadata are symlinked to v1, which is never modified.
Usage: python3 remap_wsd4.py  (run from anywhere; paths are absolute)
"""
import csv
import os
from pathlib import Path

ROOT = Path("/shared/projects/nets/viewsphere-dataset")
SRC, DST = ROOT / "v1", ROOT / "v1_wsd4"
SATS = ["Clementine", "ICESatB", "Jason-1", "Sentinel-6"]
NAMES = ["antenna", "body", "solar", "thruster"]
GROUP_TO_NEW = {"antenna": 0, "body": 1, "laser": 1, "radiator": 1, "startracker": 1,
                "telescope": 1, "instrument": 1, "solar": 2, "thruster": 3}
LINKS = ["chaser", "masks", "mission_meta.json", "camera_intrinsics.json",
         "transforms_nerf.json", "transforms_opencv.json", "visibility.csv"]
BODY = 1


def read_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def union_box(rows):
    x0 = min(float(r[1]) - float(r[3]) / 2 for r in rows)
    y0 = min(float(r[2]) - float(r[4]) / 2 for r in rows)
    x1 = max(float(r[1]) + float(r[3]) / 2 for r in rows)
    y1 = max(float(r[2]) + float(r[4]) / 2 for r in rows)
    return [str(BODY), f"{(x0 + x1) / 2:.6f}", f"{(y0 + y1) / 2:.6f}",
            f"{x1 - x0:.6f}", f"{y1 - y0:.6f}"]


def remap(rows, old_to_new):
    return [[str(old_to_new[int(r[0])])] + r[1:] for r in rows if int(r[0]) in old_to_new]


def build(sat):
    src, dst = SRC / sat, DST / sat
    ann_src, ann_dst = src / "chaser_annotations", dst / "chaser_annotations"
    for sub in ("labels_det_grouped", "labels_seg_grouped"):
        (ann_dst / sub).mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        link = dst / name
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(src / name, dst))

    groups = list(csv.DictReader(open(ann_src / "class_groups.csv")))
    old_to_new = {int(g["group_index"]): GROUP_TO_NEW[g["group_name"]]
                  for g in groups if g["group_index"] != "0"}
    with open(ann_dst / "class_groups.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["category_id", "class_name", "old_group_name", "group_name", "group_index"])
        for g in groups:
            if g["group_index"] == "0":
                continue
            new = GROUP_TO_NEW[g["group_name"]]
            w.writerow([g["category_id"], g["class_name"], g["group_name"], NAMES[new], new])
    (ann_dst / "classes_grouped.txt").write_text("\n".join(NAMES) + "\n")

    for i in range(1, 361):
        fname = f"frame_{i:04d}.txt"
        det = remap(read_rows(ann_src / "labels_det_grouped" / fname), old_to_new)
        body = [r for r in det if r[0] == str(BODY)]
        det = ([union_box(body)] if body else []) + [r for r in det if r[0] != str(BODY)]
        seg = remap(read_rows(ann_src / "labels_seg_grouped" / fname), old_to_new)
        for sub, rows in (("labels_det_grouped", det), ("labels_seg_grouped", seg)):
            (ann_dst / sub / fname).write_text("".join(" ".join(r) + "\n" for r in rows))


if __name__ == "__main__":
    for sat in SATS:
        build(sat)
        print("built", sat)
