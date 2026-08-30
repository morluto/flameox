from __future__ import annotations

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.domain.scalars import FloatingValue, IntegerValue, UnsignedIntegerValue
from flameox.evidence import (
    numeric_value_from_columns,
    numeric_value_to_columns,
    tagged_numeric_value_from_columns,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "columns"),
    [
        (None, (None, None)),
        (IntegerValue(value=42), (42, None)),
        (FloatingValue(value=42.5), (None, 42.5)),
    ],
)
def test_numeric_value_projection_round_trips(
    value: IntegerValue | FloatingValue | None,
    columns: tuple[int | None, float | None],
) -> None:
    assert numeric_value_to_columns(value) == columns
    assert numeric_value_from_columns(*columns, field_name="metric") == value


@pytest.mark.parametrize(
    ("integer_value", "floating_value"),
    [
        (42, 42.0),
        (2**63, None),
        (None, float("inf")),
    ],
)
def test_numeric_value_columns_report_invalid_persisted_values(
    integer_value: object,
    floating_value: object,
) -> None:
    with pytest.raises(DomainError) as raised:
        numeric_value_from_columns(integer_value, floating_value, field_name="metric")

    assert raised.value.code is ErrorCode.WORKSPACE_INVALID
    assert raised.value.details == {"invalid_field": "metric"}


@pytest.mark.parametrize("value", (2**63, 2**63 + 1, 2**64 - 1))
def test_tagged_unsigned_numeric_values_round_trip_exactly(value: int) -> None:
    expected = UnsignedIntegerValue(value=value)
    columns = (None, None, value, "unsigned_integer")

    assert tagged_numeric_value_from_columns(*columns, field_name="metric") == expected


@pytest.mark.parametrize(
    "columns",
    (
        (None, None, 2**64, "unsigned_integer"),
        (None, None, 1, "integer"),
        (1, None, 1, "unsigned_integer"),
    ),
)
def test_tagged_numeric_values_reject_invalid_or_ambiguous_variants(
    columns: tuple[object, object, object, object],
) -> None:
    with pytest.raises(DomainError) as raised:
        tagged_numeric_value_from_columns(*columns, field_name="metric")

    assert raised.value.code is ErrorCode.WORKSPACE_INVALID
