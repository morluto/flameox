from __future__ import annotations

from typing import Any, ClassVar

from pydantic import ConfigDict, computed_field, model_validator

from flameox.models import ContractModel


def _advertise_collection_projections(schema: dict[str, Any]) -> None:
    """Expose read-only projections in validation-mode protocol schemas."""

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    properties.setdefault(
        "returned",
        {"readOnly": True, "title": "Returned", "type": "integer"},
    )
    properties.setdefault(
        "truncated",
        {"readOnly": True, "title": "Truncated", "type": "boolean"},
    )
    required = schema.setdefault("required", [])
    if isinstance(required, list):
        for field in ("returned", "truncated"):
            if field not in required:
                required.append(field)


class CollectionResultContract(ContractModel):
    """A result whose returned count is derived from its authoritative item tuple."""

    # MCP 2.0 asks Pydantic for a validation-mode schema even though this model is
    # an output. Advertising the computed projections explicitly avoids mixing
    # validation and serialization $defs for nested result models.
    model_config = ConfigDict(json_schema_extra=_advertise_collection_projections)

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
