"""CLI for topology analysis of PEEK maps and canonical PEEK activations."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .peek_adapter import to_peek_map
from .topology import persistence_diagrams, summarize_diagrams
from .visualization import export_filtration


def _load_peek_map(input_path: Path, module: int | None) -> np.ndarray:
    if input_path.suffix == ".npy":
        return np.load(input_path)
    if input_path.suffix in {".pkl", ".pickle"}:
        with input_path.open("rb") as stream:
            activations = pickle.load(stream)
        if module is None:
            raise SystemExit(f"--module is required for an activation pickle ({input_path})")
        return to_peek_map(activations[module])
    raise SystemExit(f"Input must be .npy, .pkl, or .pickle ({input_path})")


def _run_one(
    input_path: Path,
    *,
    module: int | None,
    filtration: str,
    visualization_dir: Path | None,
) -> dict:
    peek_map = _load_peek_map(input_path, module)
    diagrams = persistence_diagrams(peek_map, filtration=filtration)
    if visualization_dir:
        export_filtration(peek_map, visualization_dir, filtration)
    return {
        "input": str(input_path),
        "module": module,
        "filtration": filtration,
        "diagrams": [x.as_dict() for x in diagrams],
        "persistence": [x.as_dict() for x in summarize_diagrams(diagrams)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path,
        help="A single .npy PEEK map or PEEK hook-generated .pkl, or a directory "
             "to recursively process every .npy PEEK map under it",
    )
    parser.add_argument("--module", type=int, help="Module key for a hook-generated pickle")
    parser.add_argument(
        "--output", type=Path,
        help="Output file (single-input mode) or output root directory (directory "
             "mode, required there) -- results mirror the input tree, .npy -> .json",
    )
    parser.add_argument("--filtration", choices=["superlevel", "sublevel"], default="superlevel")
    parser.add_argument(
        "--visualization-dir", type=Path,
        help="Directory for filtration frames/heatmap (single-input mode), or a root "
             "mirroring the input tree per file (directory mode)",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        if args.output is None:
            raise SystemExit("--output (a directory) is required when input is a directory")
        paths = sorted(args.input.rglob("*.npy"))
        if not paths:
            raise SystemExit(f"No .npy PEEK maps found under {args.input}")
        for path in paths:
            relative = path.relative_to(args.input)
            result = _run_one(
                path,
                module=args.module,
                filtration=args.filtration,
                visualization_dir=(args.visualization_dir / relative.with_suffix(""))
                if args.visualization_dir else None,
            )
            out_path = args.output / relative.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote TDA results for {len(paths)} PEEK maps to {args.output}")
        return

    result = _run_one(
        args.input, module=args.module, filtration=args.filtration,
        visualization_dir=args.visualization_dir,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
