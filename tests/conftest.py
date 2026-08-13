from __future__ import annotations

import pytest

from tests.support.providers import (
    PROVIDER_MARKERS,
    provider_available,
)


def _skip_unavailable_provider(item: pytest.Item, markers: set[str]) -> None:
    for marker in PROVIDER_MARKERS.intersection(markers):
        if not provider_available(marker):
            item.add_marker(pytest.mark.skip(reason=f"optional provider unavailable: {marker}"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        _skip_unavailable_provider(item, marker_names)
