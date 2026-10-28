"""Lay out v1_wsd4 like WSD: train/valid/test with images/ + labels/, plus data.yaml.

Split is by view (same views for every satellite): valid = views 3, 14; test = views 6, 11;
train = the other 14. Dark frames (under 10% of the target lit, v1/<sat>/visibility.csv
obj_lit_frac) are left out of train only; valid and test keep them. Files are symlinks named <sat>_frame_NNNN: images point at v1/<sat>/chaser,
labels at v1_wsd4/<sat>/chaser_annotations/labels_det_grouped (built by remap_wsd4.py, run that first).
Usage: python3 make_splits.py
"""
import csv
import os
import shutil
from pathlib import Path

ROOT = Path("/shared/projects/nets/viewsphere-dataset")
SRC, DST = ROOT / "v1", ROOT / "v1_wsd4"
SATS = ["Clementine", "ICESatB", "Jason-1", "Sentinel-6"]
SPLIT_VIEWS = {"valid": {3, 14}, "test": {6, 11}}
NAMES = ["antenna", "body", "solar", "thruster"]
DARK_LIT_FRAC = 0.10


def dark_frames(sat):
    rows = csv.DictReader(open(SRC / sat / "visibility.csv"))
    return {int(r["frame"]) for r in rows if float(r["obj_lit_frac"]) < DARK_LIT_FRAC}


def split_of(view):
    return next((s for s, views in SPLIT_VIEWS.items() if view in views), "train")


def link(target, link_path):
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def main():
    for split in ("train", "valid", "test"):
        shutil.rmtree(DST / split, ignore_errors=True)
        for sub in ("images", "labels"):
            (DST / split / sub).mkdir(parents=True)
    for sat in SATS:
        dark = dark_frames(sat)
        for row in csv.DictReader(open(SRC / "inputs" / f"{sat}_manifest.csv")):
            split, stem = split_of(int(row["view"])), f"frame_{int(row['frame']):04d}"
            if split == "train" and int(row["frame"]) in dark:
                continue
            out = DST / split
            link(SRC / sat / "chaser" / f"{stem}.png", out / "images" / f"{sat}_{stem}.png")
            link(DST / sat / "chaser_annotations" / "labels_det_grouped" / f"{stem}.txt",
                 out / "labels" / f"{sat}_{stem}.txt")
    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n")


if __name__ == "__main__":
    main()
