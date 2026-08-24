#!/usr/bin/env python3
"""Convert captured latent pickles into PEEK maps.

Reads <latents-root>/<split>/<image>.pkl (as written by capture_latents.py)
and writes <output-root>/<split>/<image>/module_{N}.npy per module, using
this project's to_peek_map (topoprun.peek_adapter). Every module present in
the pickle is converted; if one isn't a spatial BCHW activation (e.g. a
detection head's decoded predictions), this fails loudly rather than
silently dropping it -- exclude non-spatial modules explicitly at capture
time with capture_latents.py's --modules instead.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "third_party/PEEK"))
    sys.path.insert(0, str(ROOT / "src"))
    from topoprun.peek_adapter import to_peek_map

    pickles = sorted(args.latents_root.rglob("*.pkl"))
    if not pickles:
        raise SystemExit(f"No .pkl files under {args.latents_root}")

    converted = 0
    for pkl_path in pickles:
        split = pkl_path.relative_to(args.latents_root).parts[0]
        out_dir = args.output_root / split / pkl_path.stem
        with pkl_path.open("rb") as stream:
            cache = pickle.load(stream)
        out_dir.mkdir(parents=True, exist_ok=True)
        for module, activation in cache.items():
            # A genuine hook output is BCHW (ndim 4). Anything else (e.g. a
            # detection head's decoded (1, num_boxes, fields) predictions)
            # is not a spatial map, even though it may coincidentally be
            # 3D after a naive batch-squeeze -- reject by raw ndim instead
            # of guessing, and fail loudly: a captured module that can't
            # become a PEEK map means the capture request was wrong, not
            # something to quietly drop.
            if activation.ndim != 4:
                raise ValueError(
                    f"module {module} in {pkl_path} has shape {tuple(activation.shape)}, "
                    "not a spatial BCHW activation -- exclude it via capture_latents.py --modules"
                )
            peek_map = to_peek_map(activation)
            np.save(out_dir / f"module_{module}.npy", peek_map.astype(np.float32))
            converted += 1

    print(f"Wrote {converted} PEEK maps to {args.output_root}")


if __name__ == "__main__":
    main()
