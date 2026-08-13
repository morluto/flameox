from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, ValidationError

from flameox.models import ContractModel

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1


class IntegerValue(ContractModel):
    """An exact integer that can be represented by the evidence schema."""

    kind: Literal["integer"] = "integer"
    value: Annotated[StrictInt, Field(ge=_INT64_MIN, le=_INT64_MAX)]


class FloatingValue(ContractModel):
    """A finite floating-point value."""

    kind: Literal["floating"] = "floating"
    value: Annotated[StrictFloat, Field(allow_inf_nan=False)]


class UnsignedIntegerValue(ContractModel):
    """An exact unsigned integer from a provider with uint64 semantics."""

    kind: Literal["unsigned_integer"] = "unsigned_integer"
    value: Annotated[StrictInt, Field(ge=0, le=_UINT64_MAX)]


NumericValue = Annotated[
    IntegerValue | UnsignedIntegerValue | FloatingValue,
    Field(discriminator="kind"),
]


def parse_numeric_value(value: object) -> NumericValue | None:
    """Parse an optional factor value without treating booleans as integers."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        try:
            return IntegerValue(value=value)
        except ValidationError:
            return None
    if isinstance(value, float):
        return FloatingValue(value=value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed_integer = int(value)
        except ValueError:
            try:
                parsed_float = float(value)
            except ValueError:
                return None
            return FloatingValue(value=parsed_float) if math.isfinite(parsed_float) else None
        try:
            return IntegerValue(value=parsed_integer)
        except ValidationError:
            return None
    return None
