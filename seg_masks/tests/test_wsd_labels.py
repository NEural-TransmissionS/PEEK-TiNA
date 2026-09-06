from __future__ import annotations

from pathlib import Path

import pytest
from seg_masks.wsd_labels import labels_dir_for, parse_label_file


def test_labels_dir_for_wsd_and_coco_layouts():
    assert labels_dir_for(Path("d/satellite_components-71/train/images")) == Path(
        "d/satellite_components-71/train/labels"
    )
    assert labels_dir_for(Path("d/coco128/images/train2017")) == Path("d/coco128/labels/train2017")
    with pytest.raises(ValueError):
        labels_dir_for(Path("d/pics/train"))


def test_parse_label_file_converts_to_pixels(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("2 0.5 0.5 0.4 0.2\n0 0.1 0.1 0.05 0.05\n")
    boxes = parse_label_file(p, image_width=100, image_height=100)
    assert len(boxes) == 2
    solar = boxes[0]
    assert solar.class_id == 2
    assert (solar.x0, solar.y0, solar.x1, solar.y1) == (30, 40, 70, 60)


def test_parse_label_file_missing_is_background(tmp_path):
    assert parse_label_file(tmp_path / "nope.txt", 640, 640) == []


def test_parse_label_file_drops_zero_area_and_clips(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("1 0.5 0.5 0.0 0.5\n1 1.0 1.0 0.4 0.4\n")
    boxes = parse_label_file(p, 100, 100)
    assert len(boxes) == 1  # first row zero-width dropped
    assert boxes[0].x1 == 100 and boxes[0].y1 == 100  # clipped


def test_parse_label_file_rejects_malformed_row(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("2 0.5 0.5 0.4\n")
    with pytest.raises(ValueError):
        parse_label_file(p, 100, 100)
