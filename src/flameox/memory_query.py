from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from flameox.models import ContractModel


class MemoryAllocationView(StrEnum):
    HIGH_WATERMARK = "high_watermark"
    RETAINED_END = "retained_end"
    ALLOCATION_VOLUME = "allocation_volume"
    TEMPORARY = "temporary"

    @property
    def metric(self) -> str:
        return (
            "memory.allocated"
            if self is MemoryAllocationView.ALLOCATION_VOLUME
            else f"memory.{self.value}"
        )


class MemoryRanking(StrEnum):
    SELF = "self"
    INCLUSIVE = "inclusive"


type MemoryFilePrefix = Annotated[str, Field(min_length=1, max_length=4_096)]


class MemoryFrameQuery(ContractModel):
    view: MemoryAllocationView = MemoryAllocationView.HIGH_WATERMARK
    ranking: MemoryRanking = MemoryRanking.SELF
    project_only: bool = False
    include_file_prefixes: tuple[MemoryFilePrefix, ...] = Field(default=(), max_length=32)
    exclude_file_prefixes: tuple[MemoryFilePrefix, ...] = Field(default=(), max_length=32)
    include_module_prefixes: tuple[MemoryFilePrefix, ...] = Field(default=(), max_length=32)
    exclude_module_prefixes: tuple[MemoryFilePrefix, ...] = Field(default=(), max_length=32)
    exclude_zero_self: bool = True

    def broadened(self) -> MemoryFrameQuery:
        return MemoryFrameQuery(
            view=self.view,
            ranking=self.ranking,
            exclude_zero_self=False,
        )
