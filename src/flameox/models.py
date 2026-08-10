from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator


class ContractModel(BaseModel):
    """Strict immutable model shared by flameox's persisted and protocol contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @field_validator("*", mode="after")
    @classmethod
    def require_aware_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        return value

    def validated_copy(self, *, update: Mapping[str, Any] | None = None) -> Self:
        """Copy an immutable contract while re-parsing all updated fields."""

        values = self.model_dump(mode="python", exclude_computed_fields=True)
        if update is not None:
            values.update(update)
        return self.__class__.model_validate(values)
