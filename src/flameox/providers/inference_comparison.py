from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any, Literal

from flameox.canonical import canonical_bytes, digest_model
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

type Compatibility = Literal["compatible", "partial", "heterogeneous"]


def field_identities(
    groups: Mapping[str, tuple[Mapping[str, Any], Sequence[str]]],
) -> tuple[dict[str, str], list[str]]:
    """Digest identity fields independently so absent metadata remains distinguishable."""
    identity: dict[str, str] = {}
    unavailable: list[str] = []
    for group, (values, fields) in groups.items():
        for field in fields:
            name = f"{group}.{field}"
            if field not in values:
                unavailable.append(name)
                continue
            identity[name] = digest_model(
                {field: values[field]}, projection=f"flameox.inference.{name}/v1"
            )
    return identity, unavailable


def comparison_identity(analysis: ProviderAnalysis) -> tuple[dict[str, Any], set[str]]:
    metrics = analysis.blocks[0].get("values", {}) if analysis.blocks else {}
    if not isinstance(metrics, dict):
        return {}, {"system", "workload"}
    identity = metrics.get("comparison_identity", {})
    unavailable = metrics.get("comparison_identity_unavailable", [])
    return (
        dict(identity) if isinstance(identity, dict) else {},
        {str(field) for field in unavailable if isinstance(field, str)}
        if isinstance(unavailable, list)
        else {"system", "workload"},
    )


def assess_comparison(
    analyses: Sequence[ProviderAnalysis], arguments: Mapping[str, Any]
) -> tuple[Compatibility, dict[str, list[Any | None]], list[str]]:
    observed = [comparison_identity(analysis) for analysis in analyses]
    fields = set().union(*(set(identity) | unavailable for identity, unavailable in observed))
    differences: dict[str, list[Any | None]] = {}
    unavailable_fields: set[str] = set()
    for field in sorted(fields):
        values = [identity.get(field) for identity, _unavailable in observed]
        present = [value for value in values if value is not None]
        if len(present) != len(values):
            unavailable_fields.add(field)
        if len({canonical_bytes(value) for value in present}) > 1:
            differences[field] = values

    allow_heterogeneous = arguments.get("allow_heterogeneous") is True
    if differences and not allow_heterogeneous:
        raise ProviderFailure(
            "INVALID_INPUT",
            "Inference inputs have incompatible observed identities; set "
            "allow_heterogeneous=true for an explicitly exploratory comparison",
            details={"differing_fields": sorted(differences)},
        )
    compatibility: Compatibility = (
        "heterogeneous" if differences else "partial" if unavailable_fields else "compatible"
    )
    return compatibility, differences, sorted(unavailable_fields)


def mean_ratios(
    series: Sequence[Sequence[float]], baseline_index: int
) -> list[tuple[float, float | None]]:
    """Compare arithmetic means while rejecting unrepresentable derived values."""
    try:
        means = [fmean(values) for values in series]
    except OverflowError as error:
        raise ProviderFailure(
            "LIMIT_EXCEEDED", "Inference comparison mean exceeds the finite numeric range"
        ) from error
    baseline = means[baseline_index]
    comparisons = [(mean, mean / baseline if baseline else None) for mean in means]
    if any(
        not math.isfinite(mean) or (ratio is not None and not math.isfinite(ratio))
        for mean, ratio in comparisons
    ):
        raise ProviderFailure(
            "LIMIT_EXCEEDED", "Inference comparison exceeds the finite numeric range"
        )
    return comparisons
