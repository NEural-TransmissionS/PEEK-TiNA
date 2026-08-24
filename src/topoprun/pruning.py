"""Validated records and scoring for matched-compute pruning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

METHODS = {"magnitude", "random", "peek_variance", "topology"}


@dataclass(frozen=True)
class AcceptanceCriteria:
    compute_tolerance: float
    maximum_map_drop: float
    maximum_bottleneck: float
    maximum_wasserstein_2: float
    minimum_peek_variance_retention: float


@dataclass(frozen=True)
class ScoreWeights:
    accuracy: float
    bottleneck: float
    wasserstein_2: float
    peek_variance: float
    compute_match: float


@dataclass(frozen=True)
class PruningProtocol:
    modules: tuple[int, ...]
    target_flops_reductions: tuple[float, ...]
    methods: tuple[str, ...]
    criteria: AcceptanceCriteria
    weights: ScoreWeights

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PruningProtocol:
        criteria = AcceptanceCriteria(**payload["acceptance"])
        weights = ScoreWeights(**payload["score_weights"])
        protocol = cls(
            modules=tuple(payload["modules"]),
            target_flops_reductions=tuple(payload["target_flops_reductions"]),
            methods=tuple(payload["methods"]),
            criteria=criteria,
            weights=weights,
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if not self.modules or len(set(self.modules)) != len(self.modules):
            raise ValueError("modules must be a non-empty unique sequence")
        if not self.target_flops_reductions or any(
            not 0 < value < 1 for value in self.target_flops_reductions
        ):
            raise ValueError("target FLOPs reductions must be between zero and one")
        if set(self.methods) != METHODS:
            raise ValueError(f"methods must be exactly {sorted(METHODS)}")
        values = asdict(self.criteria)
        if any(not 0 < value < 1 for value in values.values()):
            raise ValueError("acceptance criteria must be between zero and one")
        weight_values = tuple(asdict(self.weights).values())
        if any(value < 0 for value in weight_values) or abs(sum(weight_values) - 1.0) > 1e-9:
            raise ValueError("score weights must be non-negative and sum to one")


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate_id: str
    method: str
    target_flops_reduction: float
    actual_flops_reduction: float
    actual_parameter_reduction: float
    physical_structure_removed: bool
    validation_map50_95: float
    peek_variance_retention: float
    topology_bottleneck: dict[int, dict[int, float]]
    topology_wasserstein_2: dict[int, dict[int, float]]
    checkpoint: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateMeasurement:
        converted = dict(payload)
        for key in ("topology_bottleneck", "topology_wasserstein_2"):
            converted[key] = {
                int(module): {int(dimension): float(value) for dimension, value in values.items()}
                for module, values in converted[key].items()
            }
        measurement = cls(**converted)
        measurement.validate()
        return measurement

    def validate(self) -> None:
        if not self.candidate_id or not self.checkpoint:
            raise ValueError("candidate_id and checkpoint are required")
        if self.method not in METHODS:
            raise ValueError(f"unknown pruning method: {self.method}")
        for name in (
            "target_flops_reduction",
            "actual_flops_reduction",
            "actual_parameter_reduction",
            "validation_map50_95",
            "peek_variance_retention",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    score: float
    validation_map_drop: float
    mean_bottleneck: float
    mean_wasserstein_2: float
    compute_error: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = ";".join(self.rejection_reasons)
        return payload


def _module_mean(
    values: dict[int, dict[int, float]], modules: tuple[int, ...], metric: str
) -> float:
    if set(values) != set(modules):
        raise ValueError(f"{metric} must contain exactly modules {modules}")
    module_means = []
    for module in modules:
        dimensions = values[module]
        if set(dimensions) != {0, 1}:
            raise ValueError(f"{metric} module {module} must contain dimensions 0 and 1")
        if any(value < 0 for value in dimensions.values()):
            raise ValueError(f"{metric} distances must be non-negative")
        module_means.append(mean(dimensions.values()))
    return mean(module_means)


def evaluate_candidate(
    candidate: CandidateMeasurement,
    protocol: PruningProtocol,
    reference_validation_map50_95: float,
) -> CandidateEvaluation:
    """Apply frozen gates and return a lower-is-better composite diagnostic score."""
    if not 0 < reference_validation_map50_95 <= 1:
        raise ValueError("reference validation mAP must be between zero and one")
    if candidate.method not in protocol.methods:
        raise ValueError(f"method {candidate.method} is not declared by the protocol")
    if not any(
        abs(candidate.target_flops_reduction - target) < 1e-9
        for target in protocol.target_flops_reductions
    ):
        raise ValueError("candidate target is not a declared FLOPs budget")

    bottleneck = _module_mean(candidate.topology_bottleneck, protocol.modules, "bottleneck")
    wasserstein = _module_mean(
        candidate.topology_wasserstein_2, protocol.modules, "wasserstein_2"
    )
    map_drop = max(0.0, reference_validation_map50_95 - candidate.validation_map50_95)
    compute_error = abs(candidate.actual_flops_reduction - candidate.target_flops_reduction)
    criteria = protocol.criteria
    reasons = []
    if not candidate.physical_structure_removed or candidate.actual_parameter_reduction <= 0:
        reasons.append("no_physical_structure_removal")
    if compute_error > criteria.compute_tolerance:
        reasons.append("compute_budget_mismatch")
    if map_drop > criteria.maximum_map_drop:
        reasons.append("validation_map_drop")
    if bottleneck > criteria.maximum_bottleneck:
        reasons.append("bottleneck_drift")
    if wasserstein > criteria.maximum_wasserstein_2:
        reasons.append("wasserstein_drift")
    if candidate.peek_variance_retention < criteria.minimum_peek_variance_retention:
        reasons.append("peek_variance_loss")

    weights = protocol.weights
    score = (
        weights.accuracy * map_drop / criteria.maximum_map_drop
        + weights.bottleneck * bottleneck / criteria.maximum_bottleneck
        + weights.wasserstein_2 * wasserstein / criteria.maximum_wasserstein_2
        + weights.peek_variance
        * max(0.0, 1.0 - candidate.peek_variance_retention)
        / (1.0 - criteria.minimum_peek_variance_retention)
        + weights.compute_match * compute_error / criteria.compute_tolerance
    )
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        score=score,
        validation_map_drop=map_drop,
        mean_bottleneck=bottleneck,
        mean_wasserstein_2=wasserstein,
        compute_error=compute_error,
    )
