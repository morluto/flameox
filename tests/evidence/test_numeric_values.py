from __future__ import annotations

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.domain.scalars import FloatingValue, IntegerValue
from flameox.evidence import numeric_value_from_columns, numeric_value_to_columns


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
