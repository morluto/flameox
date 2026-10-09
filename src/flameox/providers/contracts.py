from __future__ import annotations

import math
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
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        remediation: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        self.remediation = remediation


def canonical_provider_projection(analysis: ProviderAnalysis | None) -> ProviderAnalysis | None:
    """Project native provider values into Flameox's lossless canonical JSON domain."""
    if analysis is None:
        return None
    return ProviderAnalysis(
        provider_id=_canonical_text(analysis.provider_id),
        provider_version=_canonical_text(analysis.provider_version),
        blocks=_canonical_value(analysis.blocks),
        rows_observed=analysis.rows_observed,
        complete=analysis.complete,
        limitations=[_canonical_text(item) for item in analysis.limitations],
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, str):
        return _canonical_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value < _MIN_CANONICAL_INTEGER or value > _MAX_CANONICAL_INTEGER:
            return str(value)
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {_canonical_text(str(key)): _canonical_value(item) for key, item in value.items()}
    return _canonical_text(str(value))


def _canonical_text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ProviderFailure(
            "DECODE_FAILURE", "Provider output contains invalid Unicode text."
        ) from error
    return value
