from pathlib import Path

from topoprun.datasets import audit_wsd


def test_audit_wsd_counts_matching_pairs(tmp_path: Path):
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        (tmp_path / split / "images" / "frame.jpg").write_bytes(b"image")
        (tmp_path / split / "labels" / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    result = audit_wsd(tmp_path)
    assert result["names"] == ["antenna", "body", "solar", "thruster"]
    assert result["counts"]["val"] == {"images": 1, "labels": 1}
    assert len(result["exact_duplicate_groups"]) == 1
    assert result["exact_duplicate_groups"][0] == [
        "train/frame.jpg",
        "val/frame.jpg",
        "test/frame.jpg",
    ]
