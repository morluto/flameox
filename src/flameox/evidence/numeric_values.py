from __future__ import annotations

from pydantic import ValidationError

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.scalars import (
    FloatingValue,
    IntegerValue,
    NumericValue,
    UnsignedIntegerValue,
)


def numeric_value_from_columns(
    integer_value: object,
    floating_value: object,
    *,
    field_name: str,
) -> NumericValue | None:
    """Parse the persisted two-column representation into one domain value."""

    try:
        if integer_value is not None and floating_value is not None:
            raise ValueError("both numeric representations are populated")
        if integer_value is not None:
            return IntegerValue.model_validate({"value": integer_value})
        if floating_value is not None:
            return FloatingValue.model_validate({"value": floating_value})
    except (ValidationError, ValueError) as error:
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            f"Persisted {field_name} is invalid.",
            details={"invalid_field": field_name},
        ) from error
    return None


def numeric_value_to_columns(value: NumericValue | None) -> tuple[int | None, float | None]:
    """Project one domain value onto the stable evidence columns."""

    if isinstance(value, IntegerValue):
        return value.value, None
    if isinstance(value, FloatingValue):
        return None, value.value
    if isinstance(value, UnsignedIntegerValue):
        raise ValueError("unsigned integers require the tagged measurement projection")
    return None, None


def tagged_numeric_value_from_columns(
    integer_value: object,
    floating_value: object,
    unsigned_value: object,
    value_kind: object,
    *,
    field_name: str,
) -> NumericValue | None:
    populated = sum(value is not None for value in (integer_value, floating_value, unsigned_value))
    try:
        if populated > 1:
            raise ValueError("multiple numeric representations are populated")
        if unsigned_value is not None:
            if value_kind != "unsigned_integer":
                raise ValueError("unsigned values require an unsigned_integer tag")
            return UnsignedIntegerValue.model_validate({"value": unsigned_value})
        if value_kind not in {None, "integer", "floating"}:
            raise ValueError("numeric value kind does not match a supported representation")
        parsed = numeric_value_from_columns(
            integer_value,
            floating_value,
            field_name=field_name,
        )
        if isinstance(parsed, IntegerValue) and value_kind not in {None, "integer"}:
            raise ValueError("integer value kind does not match its representation")
        if isinstance(parsed, FloatingValue) and value_kind not in {None, "floating"}:
            raise ValueError("floating value kind does not match its representation")
        if parsed is None and value_kind is not None:
            raise ValueError("numeric value kind is present without a value")
        return parsed
    except (ValidationError, ValueError) as error:
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            f"Persisted {field_name} is invalid.",
            details={"invalid_field": field_name},
        ) from error


def tagged_numeric_value_to_columns(
    value: NumericValue | None,
) -> tuple[int | None, float | None, int | None, str | None]:
    if isinstance(value, IntegerValue):
        return value.value, None, None, value.kind
    if isinstance(value, FloatingValue):
        return None, value.value, None, value.kind
    if isinstance(value, UnsignedIntegerValue):
        return None, None, value.value, value.kind
    return None, None, None, None
