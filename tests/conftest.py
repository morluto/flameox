from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.ownership import load_ownership, ownership_by_path
from tests.support.providers import (
    PROVIDER_MARKERS,
    provider_available,
)

_TEST_ROOT = Path(__file__).parent
_PROJECT_ROOT = _TEST_ROOT.parent
_OWNERSHIP_PATH = _TEST_ROOT / "ownership.toml"


def _skip_unavailable_provider(item: pytest.Item, markers: set[str]) -> None:
    for marker in PROVIDER_MARKERS.intersection(markers):
        if not provider_available(marker):
            item.add_marker(pytest.mark.skip(reason=f"optional provider unavailable: {marker}"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    try:
        records = ownership_by_path(load_ownership(_OWNERSHIP_PATH))
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    for item in items:
        path = Path(str(item.fspath)).resolve().relative_to(_PROJECT_ROOT).as_posix()
        record = records.get(path)
        if record is None:
            raise pytest.UsageError(f"Test path is missing from tests/ownership.toml: {path}")
        marker_names = set(record.markers)
        for marker in marker_names:
            item.add_marker(getattr(pytest.mark, marker))
        marker_names.update(marker.name for marker in item.iter_markers())
        _skip_unavailable_provider(item, marker_names)
