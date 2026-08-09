from __future__ import annotations

from pydantic import ValidationError

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.scalars import FloatingValue, IntegerValue, NumericValue


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
    return None, None
