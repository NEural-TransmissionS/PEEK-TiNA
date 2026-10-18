#!/usr/bin/env python3
"""Visualize one PEEK-TiNA TDA JSON result with its source image."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


COLORS = {0: "#2369bd", 1: "#d06b22"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="One TDA JSON file, or one image directory containing module_*.json files",
    )
    parser.add_argument(
        "--peek-map",
        type=Path,
        help="Override the PEEK .npy path stored in the JSON",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Override automatic lookup of the original WSD image",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Override automatic lookup of the matching trained checkpoint",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum prediction confidence used by the correctness check",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="Minimum IoU for a prediction/ground-truth match",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.60,
        help="NMS IoU used during model prediction",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics inference device for the correctness check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output PNG (file mode) or output directory (directory mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Maximum finite barcode intervals shown per homology dimension",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution")
    return parser.parse_args()


def diagrams_by_dimension(data: dict) -> dict[int, dict]:
    diagrams = {int(item["dimension"]): item for item in data.get("diagrams", [])}
    missing = {0, 1} - diagrams.keys()
    if missing:
        raise ValueError(f"JSON is missing persistence diagram(s): {sorted(missing)}")
    return diagrams


def infer_peek_map(json_file: Path, data: dict, override: Path | None) -> Path:
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override.expanduser())
    if data.get("input"):
        candidates.append(Path(data["input"]).expanduser())

    # Fallback for a repository or artifact tree moved after TDA was run.
    try:
        parts = list(json_file.resolve().parts)
        tda_index = parts.index("tda")
        parts[tda_index] = "peek_maps"
        candidates.append(Path(*parts).with_suffix(".npy"))
    except ValueError:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the PEEK .npy map. Pass it explicitly with --peek-map."
        f"\nTried:\n  {attempted}"
    )


def artifact_context(json_file: Path) -> tuple[Path, str, str, str] | None:
    """Return repository root, model, split, and image stem from the artifact tree."""
    parts = list(json_file.resolve().parts)
    for index in range(len(parts) - 4):
        if parts[index : index + 2] != ["artifacts", "peek"]:
            continue
        try:
            tda_index = parts.index("tda", index + 2)
        except ValueError:
            return None
        if len(parts) <= tda_index + 2:
            return None
        repository = Path(*parts[:index])
        model = parts[index + 2]
        split = parts[tda_index + 1]
        image_stem = parts[tda_index + 2]
        return repository, model, split, image_stem
    return None


def infer_source_image(json_file: Path, override: Path | None) -> Path:
    if override is not None:
        source = override.expanduser().resolve(strict=True)
        if not source.is_file():
            raise FileNotFoundError(f"Source image is not a file: {source}")
        return source

    context = artifact_context(json_file)
    if context is None:
        raise FileNotFoundError(
            "Could not infer the source-image location from the artifact tree. "
            "Pass it explicitly with --source-image."
        )
    repository, _, split, image_stem = context
    split_directory = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}.get(
        split, split
    )
    image_directory = repository / "data" / split_directory / "images"
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    matches = sorted(
        path
        for path in image_directory.glob(f"{image_stem}.*")
        if path.is_file() and path.suffix.lower() in supported
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Could not find the original image for {image_stem} under {image_directory}. "
            "Pass it explicitly with --source-image."
        )
    raise RuntimeError(
        "Multiple source images matched; select one with --source-image:\n  "
        + "\n  ".join(str(path) for path in matches)
    )


def infer_checkpoint(json_file: Path, override: Path | None) -> Path:
    if override is not None:
        checkpoint = override.expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is not a file: {checkpoint}")
        return checkpoint

    context = artifact_context(json_file)
    if context is None:
        raise FileNotFoundError(
            "Could not infer the trained checkpoint from the artifact tree. "
            "Pass it explicitly with --model."
        )
    repository, model, _, _ = context
    checkpoint = repository / "runs" / "wsd71_scratch" / f"{model}_seed42" / "weights" / "best.pt"
    if checkpoint.is_file():
        return checkpoint.resolve()
    raise FileNotFoundError(
        f"Could not find the inferred checkpoint at {checkpoint}. Pass it with --model."
    )


def infer_label_file(json_file: Path) -> Path:
    context = artifact_context(json_file)
    if context is None:
        raise FileNotFoundError("Could not infer the WSD label path from the artifact tree")
    repository, _, split, image_stem = context
    split_directory = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}.get(
        split, split
    )
    label_file = repository / "data" / split_directory / "labels" / f"{image_stem}.txt"
    if not label_file.is_file():
        raise FileNotFoundError(f"Could not find the ground-truth label file: {label_file}")
    return label_file.resolve()


def load_ground_truth(label_file: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    for line_number, line in enumerate(label_file.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO fields in {label_file}:{line_number}")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        boxes.append(
            [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ]
        )
        classes.append(class_id)
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(classes, dtype=int)


def pairwise_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    upper_left = np.maximum(first[:, None, :2], second[None, :, :2])
    lower_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = np.clip(lower_right - upper_left, 0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0, None), axis=1)
    second_area = np.prod(np.clip(second[:, 2:] - second[:, :2], 0, None), axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def verify_strict_detection(json_file: Path, args: argparse.Namespace) -> None:
    from scipy.optimize import linear_sum_assignment
    from ultralytics import YOLO

    source_image = infer_source_image(json_file, args.source_image)
    checkpoint = infer_checkpoint(json_file, args.model)
    label_file = infer_label_file(json_file)
    with Image.open(source_image) as image_file:
        width, height = image_file.size
    truth_boxes, truth_classes = load_ground_truth(label_file, width, height)

    detector = YOLO(str(checkpoint))
    result = detector.predict(
        source=str(source_image),
        imgsz=640,
        conf=args.confidence,
        iou=args.nms_iou,
        device=args.device,
        verbose=False,
    )[0]
    if result.boxes is None:
        predicted_boxes = np.empty((0, 4), dtype=float)
        predicted_classes = np.empty(0, dtype=int)
    else:
        predicted_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        predicted_classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    truth_count, prediction_count = len(truth_boxes), len(predicted_boxes)
    if truth_count != prediction_count:
        raise SystemExit(
            "Strict detection check failed: "
            f"{truth_count} ground-truth boxes but {prediction_count} predictions at "
            f"confidence >= {args.confidence:.2f}. No figures were generated."
        )
    if truth_count == 0:
        print("Strict detection check passed: no ground-truth boxes and no predictions")
        return

    overlaps = pairwise_iou(truth_boxes, predicted_boxes)
    class_matches = truth_classes[:, None] == predicted_classes[None, :]
    cost = 1.0 - overlaps
    cost[~class_matches] = 1_000_000.0
    truth_indices, prediction_indices = linear_sum_assignment(cost)
    assigned_ious = overlaps[truth_indices, prediction_indices]
    assigned_classes_match = class_matches[truth_indices, prediction_indices]
    accepted = assigned_classes_match & (assigned_ious >= args.match_iou)
    if not accepted.all():
        failures = [
            (
                int(truth_classes[truth_index]),
                int(predicted_classes[prediction_index]),
                float(overlaps[truth_index, prediction_index]),
            )
            for truth_index, prediction_index, valid in zip(
                truth_indices, prediction_indices, accepted, strict=True
            )
            if not valid
        ]
        raise SystemExit(
            "Strict detection check failed: one or more boxes lacked a same-class match at "
            f"IoU >= {args.match_iou:.2f}. Failed matches (truth class, prediction class, IoU): "
            f"{failures}. No figures were generated."
        )

    print(
        f"Strict detection check passed: {truth_count} boxes matched one-to-one; "
        f"minimum IoU={assigned_ious.min():.3f}, confidence >= {args.confidence:.2f}"
    )


def finite_intervals(diagram: dict) -> np.ndarray:
    values = np.asarray(diagram.get("intervals", []), dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    values = values.reshape(-1, 2)
    return values[np.isfinite(values).all(axis=1)]


def draw_persistence_diagram(ax: plt.Axes, diagram: dict, dimension: int) -> None:
    intervals = finite_intervals(diagram)
    essential = np.asarray(diagram.get("essential_births", []), dtype=float)
    color = COLORS[dimension]

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.45", label="death = birth")
    if len(intervals):
        ax.scatter(
            intervals[:, 0],
            intervals[:, 1],
            s=20,
            alpha=0.72,
            color=color,
            edgecolors="none",
            label=f"finite features ({len(intervals)})",
        )
    if len(essential):
        ax.scatter(
            essential,
            np.ones_like(essential),
            marker="^",
            s=34,
            color=color,
            label=f"essential ({len(essential)})",
        )

    lifetimes = intervals[:, 1] - intervals[:, 0] if len(intervals) else np.empty(0)
    mean_lifetime = float(lifetimes.mean()) if len(lifetimes) else 0.0
    max_lifetime = float(lifetimes.max()) if len(lifetimes) else 0.0
    ax.set_title(
        f"$H_{dimension}$ persistence diagram\n"
        f"mean persistence={mean_lifetime:.3f}, max={max_lifetime:.3f}"
    )
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)


def draw_barcode(ax: plt.Axes, diagrams: dict[int, dict], top_k: int) -> None:
    rows: list[tuple[int, float, float, float]] = []
    for dimension in (0, 1):
        intervals = finite_intervals(diagrams[dimension])
        ranked = sorted(intervals.tolist(), key=lambda pair: pair[1] - pair[0], reverse=True)
        rows.extend((dimension, birth, death, death - birth) for birth, death in ranked[:top_k])

    rows.sort(key=lambda item: (item[0], -item[3]))
    if not rows:
        ax.text(0.5, 0.5, "No finite persistence intervals", ha="center", va="center")
        ax.set_axis_off()
        return

    for y, (dimension, birth, death, _) in enumerate(rows):
        ax.hlines(y, birth, death, color=COLORS[dimension], linewidth=2)
        ax.scatter([birth, death], [y, y], color=COLORS[dimension], s=8)

    handles = [
        plt.Line2D([0], [0], color=COLORS[0], linewidth=2, label="$H_0$"),
        plt.Line2D([0], [0], color=COLORS[1], linewidth=2, label="$H_1$"),
    ]
    ax.set_title(f"Top {top_k} finite intervals per dimension")
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Features ranked by persistence")
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)
    ax.legend(handles=handles, loc="lower right")


def module_label(json_file: Path, data: dict) -> str:
    if data.get("module") is not None:
        return f"module {data['module']}"
    match = re.search(r"module_(\d+)", json_file.stem)
    return f"module {match.group(1)}" if match else json_file.stem


def default_output_path(json_file: Path) -> Path:
    """Mirror artifacts/peek/<model>/tda/... beneath artifacts/viz/<model>/."""
    parts = list(json_file.parts)
    for index in range(len(parts) - 3):
        if parts[index : index + 2] != ["artifacts", "peek"]:
            continue
        try:
            tda_index = parts.index("tda", index + 2)
        except ValueError:
            break
        model = parts[index + 2]
        repository = Path(*parts[:index])
        relative = Path(*parts[tda_index + 1 :]).with_suffix(".png")
        return repository / "artifacts" / "viz" / model / relative
    return Path.cwd() / "artifacts" / "viz" / f"{json_file.stem}_visualization.png"


def module_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"module_(\d+)", path.stem)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def render_one(json_file: Path, args: argparse.Namespace, output: Path | None = None) -> Path:
    data = json.loads(json_file.read_text())
    diagrams = diagrams_by_dimension(data)
    peek_map_path = infer_peek_map(json_file, data, args.peek_map)
    source_image_path = infer_source_image(json_file, args.source_image)
    peek_map = np.load(peek_map_path)
    if peek_map.ndim != 2:
        raise ValueError(f"Expected a 2-D PEEK map, got shape {peek_map.shape}")
    with Image.open(source_image_path) as image_file:
        source_image = np.asarray(image_file.convert("RGB"))

    output = (output if output is not None else default_output_path(json_file)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplot_mosaic(
        [["source", "peek", "h0"], ["barcode", "barcode", "h1"]],
        figsize=(16, 10),
        constrained_layout=True,
    )
    axes["source"].imshow(source_image)
    axes["source"].set_title("Original WSD image")
    axes["source"].set_axis_off()

    peek_image = axes["peek"].imshow(peek_map, cmap="viridis", origin="upper", aspect="auto")
    axes["peek"].set_title("PEEK information map")
    axes["peek"].set_xlabel("Spatial x")
    axes["peek"].set_ylabel("Spatial y")
    fig.colorbar(peek_image, ax=axes["peek"], shrink=0.82, label="PEEK value")

    draw_persistence_diagram(axes["h0"], diagrams[0], 0)
    draw_persistence_diagram(axes["h1"], diagrams[1], 1)
    draw_barcode(axes["barcode"], diagrams, args.top_k)

    split = json_file.parent.parent.name if len(json_file.parents) >= 2 else "unknown split"
    image_name = json_file.parent.name
    fig.suptitle(
        f"{image_name} — {module_label(json_file, data)} — {split} split\n"
        "Superlevel filtration (high-information regions enter first)",
        fontsize=14,
    )
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output}")
    return output


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    for name in ("confidence", "match_iou", "nms_iou"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 1]")

    input_path = args.input_path.expanduser().resolve(strict=True)
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise SystemExit(f"Expected a JSON file, got: {input_path}")
        verify_strict_detection(input_path, args)
        output = args.output.expanduser() if args.output is not None else None
        render_one(input_path, args, output)
        return

    json_files = sorted(input_path.glob("module_*.json"), key=module_sort_key)
    if not json_files:
        raise SystemExit(f"No module_*.json files found in {input_path}")
    if args.peek_map is not None:
        raise SystemExit("--peek-map can only be used when visualizing one JSON file")
    verify_strict_detection(json_files[0], args)
    output_directory = args.output.expanduser().resolve() if args.output is not None else None
    for json_file in json_files:
        output = output_directory / f"{json_file.stem}.png" if output_directory else None
        render_one(json_file, args, output)
    print(f"Generated {len(json_files)} module visualizations for {input_path.name}")


if __name__ == "__main__":
    main()
