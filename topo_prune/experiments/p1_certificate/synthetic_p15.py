#!/usr/bin/env python3
"""P1.5 -- validate certificate.audit on synthetic scenes with KNOWN homology.

No detector. We build ``n_scenes`` logit-map fields whose superlevel-set Betti
numbers are exact by construction (``topo_prune.synthetic``), perturb each one
the way pruning would (smooth low-frequency field at a target sup-norm), and
check:

  1. Soundness -- on every scene the certificate certifies at sup-norm ``delta``,
     the perturbed Betti numbers equal the originals. Expected violations: 0.
  2. Coverage falls off smoothly as ``delta`` grows.
  3. Targeted check -- a ``bridge_perturbation`` that deliberately merges two
     panels is never certified as topology-preserving (and the audit would flag
     it if it were).

Writes run.json + prints the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "topo_prune" / "src"))

from topo_prune.certificate import audit, certify
from topo_prune.filtrations import superlevel_diagram
from topo_prune.metrics import betti_number_error
from topo_prune.synthetic import (
    bridge_perturbation,
    render_scene,
    smooth_perturbation,
)

SCENE_GRID = [
    {"n_panels": n, "gap": g, "ring": r}
    for n in (1, 2, 3, 4)
    for g in (True, False)
    for r in (False, True)
]


def _scenes(n_scenes: int, size: int):
    out = []
    for i in range(n_scenes):
        cfg = SCENE_GRID[i % len(SCENE_GRID)]
        rng = np.random.default_rng(10_000 + i)
        # vary contrast so the per-scene margin spreads and coverage is a curve,
        # not a step (real logit maps have a range of peak heights)
        peak = float(rng.uniform(3.0, 8.0))
        background = float(rng.uniform(-7.0, -2.0))
        out.append(render_scene(size=size, seed=i, peak=peak, background=background, **cfg))
    return out


def run(n_scenes: int, size: int, deltas, perturb_seeds: int, out_dir: Path):
    scenes = _scenes(n_scenes, size)
    ref_diagrams = [superlevel_diagram(s.field) for s in scenes]
    # every scene's constructed Betti must match its diagram (sanity)
    from topo_prune.filtrations import betti_numbers_at

    mismatch = sum(
        1 for s, d in zip(scenes, ref_diagrams)
        if betti_numbers_at(d, s.tau) != {0: s.beta0, 1: s.beta1}
    )

    rows = []
    for delta in deltas:
        triples = []
        n_perturbed_betti_changed = 0
        for si, (scene, ref_d) in enumerate(zip(scenes, ref_diagrams)):
            for ps in range(perturb_seeds):
                pert = smooth_perturbation(scene.field.shape, linf=delta, seed=1000 * si + ps)
                pruned_field = scene.field + pert
                pruned_d = superlevel_diagram(pruned_field)
                if betti_number_error(pruned_d, ref_d, scene.tau)["total"]:
                    n_perturbed_betti_changed += 1
                triples.append((f"s{si}.p{ps}", ref_d, pruned_d, scene.tau, delta))
        report = audit(triples)
        rows.append({
            "delta": delta,
            "triples": len(triples),
            "certified": report.certified_count,
            "coverage": report.certified_count / len(triples),
            "violations": report.violation_count,
            "perturbed_betti_changed": n_perturbed_betti_changed,
            "violation_detail": report.as_dict()["violations"][:10],
        })
        print(f"delta={delta:5.2f}  coverage={rows[-1]['coverage']:.3f}  "
              f"violations={rows[-1]['violations']}  "
              f"(betti changed on {n_perturbed_betti_changed}/{len(triples)} perturbations)")

    # targeted: a bridge that merges two panels must never be certified
    targeted = {"cases": 0, "certified": 0, "betti_actually_changed": 0}
    for si, (scene, ref_d) in enumerate(zip(scenes, ref_diagrams)):
        if scene.beta0 < 2:
            continue
        targeted["cases"] += 1
        amp = 12.0
        bridged = scene.field + bridge_perturbation(scene.field.shape, amplitude=amp)
        bridged_d = superlevel_diagram(bridged)
        # certify against the sup-norm the bridge actually introduces
        realized = float(np.abs(bridged - scene.field).max())
        r = certify(ref_d, scene.tau, realized)
        if r.certified:
            targeted["certified"] += 1
        if betti_number_error(bridged_d, ref_d, scene.tau)["total"]:
            targeted["betti_actually_changed"] += 1

    manifest = {
        "experiment": "p1_5_synthetic_certificate_validation",
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_scenes": len(scenes),
        "scene_grid": SCENE_GRID,
        "grid_size": size,
        "perturbation": "smooth low-frequency field, exact target sup-norm",
        "constructed_vs_computed_betti_mismatches": mismatch,
        "deltas": rows,
        "targeted_bridge_check": targeted,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"synthetic_{stamp}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nconstructed-vs-computed Betti mismatches: {mismatch}/{len(scenes)}")
    print(f"targeted bridge check: {targeted}")
    print(f"wrote {out_dir / f'synthetic_{stamp}.json'}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-scenes", type=int, default=240)
    p.add_argument("--size", type=int, default=72)
    p.add_argument("--deltas", default="0.5,1,2,3,4,6,8")
    p.add_argument("--perturb-seeds", type=int, default=5)
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "topo_prune" / "experiments" / "p1_certificate" / "runs")
    a = p.parse_args()
    run(a.n_scenes, a.size, [float(x) for x in a.deltas.split(",")], a.perturb_seeds, a.out_dir)


if __name__ == "__main__":
    main()
