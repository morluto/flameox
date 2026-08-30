from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict, computed_field, model_validator

from flameox.models import ContractModel


class CollectionResultContract(ContractModel):
    """A result whose returned count is derived from its authoritative item tuple."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    page_items_field: ClassVar[str]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def returned(self) -> int:
        items = getattr(self, self.page_items_field)
        if not isinstance(items, tuple):
            raise TypeError(f"{self.page_items_field} must be a tuple")
        return len(items)


class CursorPageContract(CollectionResultContract):
    """A cursor page whose continuation state is derived from its cursor."""

    next_cursor: str | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def truncated(self) -> bool:
        return self.next_cursor is not None

    @model_validator(mode="after")
    def total_covers_page(self) -> CursorPageContract:
        total = getattr(self, "total", None)
        if total is None:
            return self
        if not isinstance(total, int) or isinstance(total, bool):
            raise TypeError("page total must be an integer")
        if total < self.returned:
            raise ValueError("page total cannot be smaller than its returned items")
        if self.truncated and total <= self.returned:
            raise ValueError("page continuation cursor requires unreturned items")
        return self


class BoundedCollectionContract(CollectionResultContract):
    """A head-bounded collection whose truncation is derived from its total."""

    total_items_field: ClassVar[str] = "total"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def truncated(self) -> bool:
        return self._total > self.returned

    @property
    def _total(self) -> int:
        total = getattr(self, self.total_items_field)
        if not isinstance(total, int) or isinstance(total, bool):
            raise TypeError("page total must be an integer")
        return total

    @model_validator(mode="after")
    def total_covers_page(self) -> BoundedCollectionContract:
        if self._total < self.returned:
            raise ValueError("page total cannot be smaller than its returned items")
        return self
