import pytest

from topoprun.topology import TopologyDistanceRecord, aggregate_topology_distances


def records():
    return [
        TopologyDistanceRecord(
            image=image,
            module=module,
            dimension=dimension,
            bottleneck=0.01 * (image_index + module + dimension),
            wasserstein_2=0.02 * (image_index + module + dimension),
        )
        for image_index, image in enumerate(("a.jpg", "b.jpg"))
        for module in (4, 6)
        for dimension in (0, 1)
    ]


def test_complete_topology_aggregation_is_deterministic():
    first = aggregate_topology_distances(
        records(), ("a.jpg", "b.jpg"), (4, 6), bootstrap_resamples=100, seed=42
    )
    second = aggregate_topology_distances(
        records(), ("a.jpg", "b.jpg"), (4, 6), bootstrap_resamples=100, seed=42
    )
    assert first == second
    assert first["independent_unit"] == "image"
    assert first["equal_weight_overall"]["bottleneck"]["count"] == 2
    assert set(first["module_summaries"]) == {"4", "6"}


def test_topology_aggregation_rejects_incomplete_coverage():
    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate_topology_distances(
            records()[:-1], ("a.jpg", "b.jpg"), (4, 6), bootstrap_resamples=10
        )


def test_topology_aggregation_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_topology_distances(
            records() + [records()[0]],
            ("a.jpg", "b.jpg"),
            (4, 6),
            bootstrap_resamples=10,
        )
