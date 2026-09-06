"""Read YOLO-format WSD labels into pixel-space boxes.

WSD exports (Roboflow "YOLOv5 PyTorch", parent ``README.md``) put one ``.txt``
per image next to it under ``labels/``, each row ``cls cx cy w h`` normalised to
``[0, 1]``. Classes: ``0 antenna, 1 body, 2 solar, 3 thruster``
(``src/topoprun/datasets.py:13``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WSD_CLASSES: tuple[str, ...] = ("antenna", "body", "solar", "thruster")


@dataclass(frozen=True)
class Box:
    class_id: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.width, self.height


def labels_dir_for(images_dir: Path) -> Path:
    """The YOLO labels tree that mirrors ``images_dir`` with the last ``images``
    path segment swapped for ``labels`` -- handles WSD's ``train/images`` +
    ``train/labels`` and COCO's ``images/<split>`` + ``labels/<split>``.
    """
    parts = list(Path(images_dir).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise ValueError(f"no 'images' segment in {images_dir}")


def parse_label_file(path: Path, image_width: int, image_height: int) -> list[Box]:
    """Parse one YOLO ``.txt`` into pixel-space :class:`Box` objects.

    Boxes are clipped to the image and zero-area rows are dropped. A missing file
    means a valid background image and yields ``[]``.
    """
    path = Path(path)
    if not path.is_file():
        return []
    boxes: list[Box] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        row = raw.split()
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(row)}")
        cls = int(float(row[0]))
        cx, cy, w, h = (float(v) for v in row[1:])
        x0 = round((cx - w / 2) * image_width)
        y0 = round((cy - h / 2) * image_height)
        x1 = round((cx + w / 2) * image_width)
        y1 = round((cy + h / 2) * image_height)
        x0, x1 = max(0, min(x0, image_width)), max(0, min(x1, image_width))
        y0, y1 = max(0, min(y0, image_height)), max(0, min(y1, image_height))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append(Box(class_id=cls, x0=x0, y0=y0, x1=x1, y1=y1))
    return boxes
