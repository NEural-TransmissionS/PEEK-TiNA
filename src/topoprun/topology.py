"""Persistent-homology summaries for scalar PEEK maps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PersistenceDiagram:
    dimension: int
    intervals: tuple[tuple[float, float], ...]
    essential_births: tuple[float, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PersistenceSummary:
    dimension: int
    finite_features: int
    total_persistence: float
    mean_persistence: float
    max_persistence: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TopologyDistanceRecord:
    image: str
    module: int
    dimension: int
    bottleneck: float
    wasserstein_2: float

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("PEEK map must be a non-empty, finite 2-D array")
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high == low else (values - low) / (high - low)


def persistence_diagrams(
    peek_map: np.ndarray,
    dimensions: tuple[int, ...] = (0, 1),
    filtration: str = "superlevel",
) -> list[PersistenceDiagram]:
    """Return normalized cubical persistence diagrams, retaining finite intervals."""
    try:
        import gudhi
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to use topology analysis") from exc

    field = _normalized(peek_map)
    if filtration == "superlevel":
        # CubicalComplex implements a lower-star filtration. Negating the normalized
        # PEEK map makes high-information regions enter first.
        field = 1.0 - field
    elif filtration != "sublevel":
        raise ValueError("filtration must be 'sublevel' or 'superlevel'")
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=field)
    complex_.compute_persistence()
    diagrams = []
    for dimension in dimensions:
        intervals = complex_.persistence_intervals_in_dimension(dimension)
        finite = intervals[np.isfinite(intervals[:, 1])] if intervals.size else np.empty((0, 2))
        essential = intervals[~np.isfinite(intervals[:, 1]), 0] if intervals.size else np.empty(0)
        diagrams.append(
            PersistenceDiagram(
                dimension=dimension,
                intervals=tuple(map(tuple, finite.tolist())),
                essential_births=tuple(map(float, essential)),
            )
        )
    return diagrams


def summarize_diagrams(diagrams: list[PersistenceDiagram]) -> list[PersistenceSummary]:
    summaries = []
    for diagram in diagrams:
        finite = np.asarray(diagram.intervals, dtype=float).reshape(-1, 2)
        lifetimes = finite[:, 1] - finite[:, 0] if finite.size else np.empty(0)
        summaries.append(
            PersistenceSummary(
                dimension=diagram.dimension,
                finite_features=int(lifetimes.size),
                total_persistence=float(lifetimes.sum()),
                mean_persistence=float(lifetimes.mean()) if lifetimes.size else 0.0,
                max_persistence=float(lifetimes.max()) if lifetimes.size else 0.0,
            )
        )
    return summaries


def summarize_map(
    peek_map: np.ndarray,
    dimensions: tuple[int, ...] = (0, 1),
    filtration: str = "superlevel",
) -> list[PersistenceSummary]:
    return summarize_diagrams(persistence_diagrams(peek_map, dimensions, filtration))


def compare_diagrams(
    reference: list[PersistenceDiagram],
    candidate: list[PersistenceDiagram],
    wasserstein_order: float = 2.0,
) -> dict[int, dict[str, float]]:
    """Compare finite diagrams with bottleneck and q-Wasserstein distances."""
    try:
        import gudhi
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to compare persistence diagrams") from exc

    ref = {diagram.dimension: diagram for diagram in reference}
    cand = {diagram.dimension: diagram for diagram in candidate}
    if ref.keys() != cand.keys():
        raise ValueError("Reference and candidate must contain the same dimensions")
    if wasserstein_order <= 0:
        raise ValueError("wasserstein_order must be positive")
    output = {}
    for dimension, reference_diagram in ref.items():
        a = np.asarray(reference_diagram.intervals, dtype=float).reshape(-1, 2)
        b = np.asarray(cand[dimension].intervals, dtype=float).reshape(-1, 2)
        bottleneck = 0.0 if np.array_equal(a, b) else float(gudhi.bottleneck_distance(a, b))
        n, m, q = len(a), len(b), wasserstein_order
        if n + m == 0:
            wasserstein = 0.0
        else:
            cost = np.full((n + m, n + m), np.inf)
            if n and m:
                cost[:n, :m] = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2) ** q
            if n:
                cost[np.arange(n), m + np.arange(n)] = ((a[:, 1] - a[:, 0]) / 2) ** q
            if m:
                cost[n + np.arange(m), np.arange(m)] = ((b[:, 1] - b[:, 0]) / 2) ** q
            cost[n:, m:] = 0.0
            rows, columns = linear_sum_assignment(cost)
            wasserstein = float(cost[rows, columns].sum() ** (1 / q))
        output[dimension] = {"bottleneck": bottleneck, f"wasserstein_{q:g}": wasserstein}
    return output


def compare_summaries(
    reference: list[PersistenceSummary], candidate: list[PersistenceSummary]
) -> float:
    """Return a scale-free topology drift score (zero means identical summaries)."""
    ref, cand = {x.dimension: x for x in reference}, {x.dimension: x for x in candidate}
    if ref.keys() != cand.keys():
        raise ValueError("Reference and candidate must contain the same dimensions")
    errors = []
    for dimension, reference_summary in ref.items():
        for name in ("finite_features", "total_persistence", "max_persistence"):
            a = float(getattr(reference_summary, name))
            b = float(getattr(cand[dimension], name))
            errors.append(abs(a - b) / max(abs(a), 1e-12))
    return float(np.mean(errors)) if errors else 0.0


def _distribution_summary(
    values: np.ndarray, bootstrap_resamples: int, random: np.random.Generator
) -> dict[str, float]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("aggregate values must be a non-empty finite vector")
    bootstrapped = random.choice(values, size=(bootstrap_resamples, len(values)), replace=True).mean(1)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "bootstrap_95_low": float(np.quantile(bootstrapped, 0.025)),
        "bootstrap_95_high": float(np.quantile(bootstrapped, 0.975)),
    }


def aggregate_topology_distances(
    records: list[TopologyDistanceRecord],
    images: tuple[str, ...],
    modules: tuple[int, ...],
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Aggregate complete image/module topology comparisons using images as replicates."""
    if not images or len(images) != len(set(images)):
        raise ValueError("images must be a non-empty unique sequence")
    if not modules or len(modules) != len(set(modules)):
        raise ValueError("modules must be a non-empty unique sequence")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    expected = {(image, module, dimension) for image in images for module in modules for dimension in (0, 1)}
    observed = {(record.image, record.module, record.dimension) for record in records}
    if len(observed) != len(records):
        raise ValueError("topology records contain duplicate image/module/dimension keys")
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        raise ValueError(f"topology record coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    if any(
        value < 0 or not np.isfinite(value)
        for record in records
        for value in (record.bottleneck, record.wasserstein_2)
    ):
        raise ValueError("topology distances must be finite and non-negative")

    lookup = {
        (record.image, record.module, record.dimension): record for record in records
    }
    random = np.random.default_rng(seed)
    by_module = defaultdict(dict)
    for module in modules:
        for dimension in (0, 1):
            selected = [lookup[(image, module, dimension)] for image in images]
            for metric in ("bottleneck", "wasserstein_2"):
                values = np.asarray([getattr(record, metric) for record in selected], dtype=float)
                by_module[str(module)][f"{metric}_h{dimension}"] = _distribution_summary(
                    values, bootstrap_resamples, random
                )

    overall = {}
    for metric in ("bottleneck", "wasserstein_2"):
        # Average the predeclared modules and dimensions within each image first.
        image_values = np.asarray(
            [
                np.mean(
                    [
                        getattr(lookup[(image, module, dimension)], metric)
                        for module in modules
                        for dimension in (0, 1)
                    ]
                )
                for image in images
            ],
            dtype=float,
        )
        overall[metric] = _distribution_summary(image_values, bootstrap_resamples, random)
    return {
        "independent_unit": "image",
        "images": len(images),
        "modules": list(modules),
        "dimensions": [0, 1],
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "module_summaries": dict(by_module),
        "equal_weight_overall": overall,
    }
