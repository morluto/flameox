from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
