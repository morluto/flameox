from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from flameox.canonical import canonical_bytes
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

type Compatibility = Literal["compatible", "partial", "heterogeneous"]


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
