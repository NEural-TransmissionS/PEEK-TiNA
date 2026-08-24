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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy PEEK map or PEEK hook-generated .pkl")
    parser.add_argument("--module", type=int, help="Module key for a hook-generated pickle")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--filtration", choices=["superlevel", "sublevel"], default="superlevel")
    parser.add_argument("--visualization-dir", type=Path)
    args = parser.parse_args()

    if args.input.suffix == ".npy":
        peek_map = np.load(args.input)
    elif args.input.suffix in {".pkl", ".pickle"}:
        with args.input.open("rb") as stream:
            activations = pickle.load(stream)
        if args.module is None:
            raise SystemExit("--module is required for an activation pickle")
        peek_map = to_peek_map(activations[args.module])
    else:
        raise SystemExit("Input must be .npy, .pkl, or .pickle")

    diagrams = persistence_diagrams(peek_map, filtration=args.filtration)
    if args.visualization_dir:
        export_filtration(peek_map, args.visualization_dir, args.filtration)
    result = {
        "input": str(args.input),
        "module": args.module,
        "filtration": args.filtration,
        "diagrams": [x.as_dict() for x in diagrams],
        "persistence": [x.as_dict() for x in summarize_diagrams(diagrams)],
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
