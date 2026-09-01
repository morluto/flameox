from __future__ import annotations

from flameox.providers.contracts import ProviderAnalysis, canonical_provider_projection


def test_provider_projection_normalizes_non_finite_floats_for_canonical_json() -> None:
    projected = canonical_provider_projection(
        ProviderAnalysis(
            provider_id="fixture",
            provider_version="1",
            blocks=[{"values": [float("nan"), float("inf"), float("-inf"), 1.5]}],
            rows_observed=1,
            complete=True,
            limitations=[],
        )
    )

    assert projected is not None
    assert projected.blocks == [{"values": ["NaN", "Infinity", "-Infinity", 1.5]}]
