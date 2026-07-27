"""Regression tests for the silent block-key collision in ``_samples``.

The unblocked path of ``ComparisonService._samples`` writes
``values[unit_key]`` without checking whether the key already exists. The
blocked path guards the same situation by raising. These tests pin the
defensive invariant that the unblocked path must also raise on collision
rather than silently dropping a measurement.

NOTE: the production guard at the top of ``_samples`` currently makes the
multi-member unblocked path unreachable, so these tests construct the
minimal single-member scenario where a collision is still possible (a single
run emitting two measurements with the same worker/worker_run/value_index).
"""
from __future__ import annotations

import pytest

from flameox.application.comparisons import ComparisonService
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import RunSet, RunSetMember


class _FakeSnapshot:
    def __init__(self, measurements_by_run: dict[str, list[tuple]]) -> None:
        self._measurements = measurements_by_run

    def execute(self, sql: str, params: tuple) -> object:
        if "FROM measurements" in sql:
            return _Rows(self._measurements.get(params[0], []))
        if "FROM trials" in sql:
            return _Rows([])
        return _Rows([])


class _Rows:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


def _row(value_int, value_index, worker_id="0", worker_run_index=0,
         block_id=None) -> tuple:
    return (value_int, None, block_id, worker_id, worker_run_index, value_index)


def _samples(measurements, members):
    snapshot = _FakeSnapshot(measurements)
    run_set = RunSet(
        run_set_id="sha256:" + "0" * 64,
        corpus_commit_id="sha256:" + "a" * 64,
        selection={},
        members=members,
        membership_digest="sha256:" + "b" * 64,
    )
    return ComparisonService.__new__(ComparisonService)._samples(
        snapshot, run_set, "m", "ns"
    )


def test_unblocked_path_raises_on_duplicate_key() -> None:
    """A single run with two measurements sharing a key must raise, not drop."""
    measurements = {
        "run-a": [
            _row(10, 0),
            _row(20, 0),  # same (worker, worker_run, value_index) -> collision
        ],
    }
    members = (RunSetMember(run_id="run-a", included=True, order=0),)
    with pytest.raises(DomainError) as exc:
        _samples(measurements, members)
    assert exc.value.code is ErrorCode.COMPARISON_INVALID


def test_unblocked_path_accepts_distinct_keys() -> None:
    """Distinct value_index values must not raise and must both be retained."""
    measurements = {
        "run-a": [_row(10, 0), _row(20, 1)],
    }
    members = (RunSetMember(run_id="run-a", included=True, order=0),)
    result = _samples(measurements, members)
    assert sorted(result.values.values()) == [10.0, 20.0]
    assert result.eligible == 2
