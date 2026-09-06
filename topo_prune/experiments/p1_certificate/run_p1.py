#!/usr/bin/env python3
"""P1 experiment E1 -- the falsification audit on a really-pruned YOLO26n.

Pipeline (plan §3.3-§3.4, detector retarget per DEVIATIONS.md D1):

  load yolo26n  ->  for each (method, budget):
    prune the classification head           (topo_prune.prune.prune_head)
    capture ref + pruned class-logit maps    (topo_prune.capture)
      on a calibration split  -> epsilon_empirical, epsilon_certified per (class, level)
      on a held-out split     -> per-image superlevel diagrams
    certify each (image, class, level) triple with both epsilon
    audit: does beta_k(S_c^(i),M) == beta_k(S_c^(i)) on every certified triple?

Writes run.json (fixed schema) + prints the coverage / violation / looseness
table. COCO-pretrained weights stand in for the WSD checkpoint (REPO_MAP.md B2);
swap --weights when it lands.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "topo_prune" / "src"))

from topo_prune.capture import capture_images, linf_gap
from topo_prune.certificate import certify
from topo_prune.filtrations import superlevel_diagram
from topo_prune.head_maps import logit
from topo_prune.metrics import betti_number_error
from topo_prune.prune import prune_head


def _epsilon_certified(ref_caps, prune_records, detect_weight_by_level):
    """Sum_{k dropped} |W[c, level, k]| * max_calib,space |feat_k|  (DEVIATIONS.md D1.4)."""
    # per-channel max |penultimate feature| over the calibration set
    max_feat: dict[int, np.ndarray] = {}
    for cap in ref_caps:
        for level, feat in cap.penultimate.items():
            m = np.abs(feat).max(axis=(1, 2))
            max_feat[level] = m if level not in max_feat else np.maximum(max_feat[level], m)
    eps: dict[tuple[int, int], float] = {}
    for rec in prune_records:
        w = detect_weight_by_level[rec.level]  # [nc, c3]
        dropped = list(rec.dropped_channels)
        if not dropped:
            contrib = np.zeros(w.shape[0])
        else:
            contrib = (np.abs(w[:, dropped]) * max_feat[rec.level][dropped][None, :]).sum(axis=1)
        for c in range(w.shape[0]):
            eps[(c, rec.level)] = float(contrib[c])
    return eps


def _active_classes(cap, tau_floor: float) -> set[int]:
    active = set()
    for arr in cap.logit_maps.values():
        peak = arr.reshape(arr.shape[0], -1).max(axis=1)
        active |= set(np.where(peak >= tau_floor)[0].tolist())
    return active


def run(weights: str, n_images: int, imgsz: int, methods, budgets, conf: float, out_dir: Path):
    from ultralytics import YOLO

    tau = logit(conf)
    tau_floor = logit(0.05)  # a class must clear this somewhere to be audited
    coco = ROOT / "third_party" / "datasets" / "coco128" / "images" / "train2017"
    images = sorted(coco.glob("*.jpg"))[:n_images]
    half = len(images) // 2
    calib, test = images[:half], images[half:]

    base = YOLO(weights)
    base_module = base.model
    detect = base_module.model[-1]
    weight_by_level = {
        i: b[-1].weight.detach().cpu().numpy()[:, :, 0, 0] for i, b in enumerate(detect.one2one_cv3)
    }
    nc = detect.nc

    def _use(module):
        base.model = module
        base.predictor = None  # force predict() to rebind to the swapped module

    t0 = time.time()
    _use(base_module)
    ref_calib = capture_images(base, calib, imgsz=imgsz, with_penultimate=True)
    ref_test = capture_images(base, test, imgsz=imgsz)
    print(f"reference capture: {len(calib)}+{len(test)} imgs in {time.time() - t0:.0f}s")

    arms = []
    for method in methods:
        for budget in budgets:
            ta = time.time()
            pruned, records = prune_head(base_module, fraction=budget, method=method, seed=42)
            actual = float(np.mean([r.actual_fraction for r in records]))
            _use(pruned)
            pr_calib = capture_images(base, calib, imgsz=imgsz, with_penultimate=True)
            pr_test = capture_images(base, test, imgsz=imgsz)
            _use(base_module)

            eps_emp: dict[tuple[int, int], float] = {}
            for rc, pc in zip(ref_calib, pr_calib):
                for key, g in linf_gap(rc.logit_maps, pc.logit_maps).items():
                    eps_emp[key] = max(eps_emp.get(key, 0.0), g)
            eps_cert = _epsilon_certified(ref_calib, records, weight_by_level)

            stats = {
                k: 0 for k in (
                    "triples", "cert_emp", "cert_cert", "viol_emp", "viol_cert",
                    "betti_changed_total",
                )
            }
            looseness = []
            violations = []
            max_test_linf = 0.0
            emp_bound_held = True
            for rt, pt in zip(ref_test, pr_test):
                test_gap = linf_gap(rt.logit_maps, pt.logit_maps)
                for c in _active_classes(rt, tau_floor):
                    for level in rt.logit_maps:
                        true_linf = test_gap[(c, level)]
                        max_test_linf = max(max_test_linf, true_linf)
                        ref_d = superlevel_diagram(rt.logit_maps[level][c])
                        pru_d = superlevel_diagram(pt.logit_maps[level][c])
                        be = betti_number_error(pru_d, ref_d, tau)["total"]
                        stats["triples"] += 1
                        if be:
                            stats["betti_changed_total"] += 1
                        for tag, eps in (("emp", eps_emp), ("cert", eps_cert)):
                            e = eps[(c, level)]
                            if tag == "emp" and true_linf > e:
                                emp_bound_held = False
                            r = certify(ref_d, tau, e)
                            if r.certified:
                                stats[f"cert_{tag}"] += 1
                                if be:
                                    stats[f"viol_{tag}"] += 1
                                    violations.append({
                                        "image": rt.image, "class": int(c), "level": int(level),
                                        "epsilon_tag": tag, "epsilon": e,
                                        "true_linf_gap": true_linf,
                                        "epsilon_bounded_the_gap": bool(true_linf <= e),
                                        "betti_error": int(be),
                                    })
                    ee = eps_emp[(c, 0)]
                    if ee > 0:
                        looseness.append(eps_cert[(c, 0)] / ee)

            arm = {
                "method": method,
                "target_fraction": budget,
                "actual_fraction": actual,
                "epsilon_empirical_median": float(np.median(list(eps_emp.values()))),
                "epsilon_certified_median": float(np.median(list(eps_cert.values()))),
                "looseness_median": float(np.median(looseness)) if looseness else None,
                "max_test_linf_gap": max_test_linf,
                "empirical_bound_held_on_test": emp_bound_held,
                "audited_triples": stats["triples"],
                "coverage_empirical": stats["cert_emp"] / max(1, stats["triples"]),
                "coverage_certified": stats["cert_cert"] / max(1, stats["triples"]),
                "violations_empirical": stats["viol_emp"],
                "violations_certified": stats["viol_cert"],
                "betti_changed_triples": stats["betti_changed_total"],
                "violation_detail": violations,
                "seconds": round(time.time() - ta, 1),
            }
            arms.append(arm)
            print(
                f"[{method} @ {budget:.0%}] cov_emp={arm['coverage_empirical']:.2%} "
                f"cov_cert={arm['coverage_certified']:.2%} "
                f"viol_emp={arm['violations_empirical']} viol_cert={arm['violations_certified']} "
                f"emp_bound_held={emp_bound_held} looseness~{arm['looseness_median']}"
            )

    manifest = {
        "experiment": "p1_certificate_audit_detection",
        "utc": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "note": "COCO-pretrained YOLO26n stand-in for the WSD checkpoint (REPO_MAP.md B2)",
        "imgsz": imgsz,
        "operating_point_conf": conf,
        "tau_logit": tau,
        "n_calib": len(calib),
        "n_test": len(test),
        "nc": int(nc),
        "python": platform.python_version(),
        "arms": arms,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"run_{stamp}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir / f'run_{stamp}.json'}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolo26n.pt")
    p.add_argument("--n-images", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--methods", default="magnitude,random")
    p.add_argument("--budgets", default="0.2,0.5")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=ROOT / "topo_prune" / "experiments" / "p1_certificate" / "runs")
    a = p.parse_args()
    run(
        a.weights, a.n_images, a.imgsz,
        a.methods.split(","), [float(x) for x in a.budgets.split(",")],
        a.conf, a.out_dir,
    )


if __name__ == "__main__":
    main()
