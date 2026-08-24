"""Across-seed uncertainty summaries for independent training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SeedSummary:
    count: int
    mean: float
    standard_deviation: float | None
    confidence_level: float
    confidence_low: float | None
    confidence_high: float | None
    reportable: bool

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_seeds(
    values: list[float], confidence: float = 0.95, minimum_reportable: int = 3
) -> SeedSummary:
    """Return a Student-t interval across independent seed-level measurements."""
    if not values or not all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite sequence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    array = np.asarray(values, dtype=float)
    count = len(array)
    mean = float(array.mean())
    if count < 2:
        standard_deviation = low = high = None
    else:
        from scipy.stats import t

        standard_deviation = float(array.std(ddof=1))
        half_width = float(
            t.ppf((1 + confidence) / 2, df=count - 1) * standard_deviation / np.sqrt(count)
        )
        low, high = mean - half_width, mean + half_width
    return SeedSummary(
        count=count,
        mean=mean,
        standard_deviation=standard_deviation,
        confidence_level=confidence,
        confidence_low=low,
        confidence_high=high,
        reportable=count >= minimum_reportable,
    )
