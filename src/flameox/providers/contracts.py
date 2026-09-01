from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MAX_CANONICAL_INTEGER = 2**53 - 1
_MIN_CANONICAL_INTEGER = -(2**53) + 1


@dataclass(frozen=True, slots=True)
class ProviderAnalysis:
    provider_id: str
    provider_version: str
    blocks: list[dict[str, Any]]
    rows_observed: int
    complete: bool
    limitations: list[str]


class ProviderFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def canonical_provider_projection(analysis: ProviderAnalysis | None) -> ProviderAnalysis | None:
    """Project native provider values into Flameox's lossless canonical JSON domain."""
    if analysis is None:
        return None
    return ProviderAnalysis(
        provider_id=analysis.provider_id,
        provider_version=analysis.provider_version,
        blocks=_canonical_value(analysis.blocks),
        rows_observed=analysis.rows_observed,
        complete=analysis.complete,
        limitations=analysis.limitations,
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int):
        if value < _MIN_CANONICAL_INTEGER or value > _MAX_CANONICAL_INTEGER:
            return str(value)
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    return str(value)
