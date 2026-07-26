from __future__ import annotations

from datetime import datetime
from typing import Any

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
