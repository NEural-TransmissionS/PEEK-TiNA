from pathlib import Path

import pytest
import yaml

from topoprun.pruning import CandidateMeasurement, PruningProtocol, evaluate_candidate

ROOT = Path(__file__).resolve().parents[1]


def protocol() -> PruningProtocol:
    payload = yaml.safe_load((ROOT / "configs/pruning_yolo26.yaml").read_text())
    return PruningProtocol.from_dict(payload["pruning"])


def measurement(**overrides) -> CandidateMeasurement:
    modules = (4, 6, 10, 16, 19, 22)
    payload = {
        "candidate_id": "magnitude_020",
        "method": "magnitude",
        "target_flops_reduction": 0.20,
        "actual_flops_reduction": 0.205,
        "actual_parameter_reduction": 0.18,
        "physical_structure_removed": True,
        "validation_map50_95": 0.42,
        "peek_variance_retention": 0.90,
        "topology_bottleneck": {str(module): {"0": 0.02, "1": 0.04} for module in modules},
        "topology_wasserstein_2": {
            str(module): {"0": 0.03, "1": 0.05} for module in modules
        },
        "checkpoint": "artifacts/magnitude_020.pt",
    }
    payload.update(overrides)
    return CandidateMeasurement.from_dict(payload)


def test_protocol_is_valid_and_weights_sum_to_one():
    result = protocol()
    assert result.modules == (4, 6, 10, 16, 19, 22)
    assert sum(vars(result.weights).values()) == pytest.approx(1.0)


def test_candidate_passes_frozen_gates():
    result = evaluate_candidate(measurement(), protocol(), reference_validation_map50_95=0.434)
    assert result.accepted
    assert result.rejection_reasons == ()
    assert result.score > 0


def test_candidate_reports_every_failed_gate():
    result = evaluate_candidate(
        measurement(
            actual_flops_reduction=0.24,
            actual_parameter_reduction=0.0,
            physical_structure_removed=False,
            validation_map50_95=0.39,
            peek_variance_retention=0.70,
            topology_bottleneck={str(m): {"0": 0.2, "1": 0.2} for m in (4, 6, 10, 16, 19, 22)},
            topology_wasserstein_2={
                str(m): {"0": 0.3, "1": 0.3} for m in (4, 6, 10, 16, 19, 22)
            },
        ),
        protocol(),
        reference_validation_map50_95=0.434,
    )
    assert not result.accepted
    assert set(result.rejection_reasons) == {
        "no_physical_structure_removal",
        "compute_budget_mismatch",
        "validation_map_drop",
        "bottleneck_drift",
        "wasserstein_drift",
        "peek_variance_loss",
    }


def test_candidate_requires_all_frozen_modules():
    item = measurement()
    payload = {**item.__dict__, "topology_bottleneck": {4: {0: 0.1, 1: 0.1}}}
    incomplete = CandidateMeasurement(**payload)
    with pytest.raises(ValueError, match="exactly modules"):
        evaluate_candidate(incomplete, protocol(), reference_validation_map50_95=0.434)
