from __future__ import annotations

import importlib.util

import pytest


@pytest.fixture(autouse=True)
def isolated_client_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "GEMINI_CLI_HOME",
        "XDG_CONFIG_HOME",
        "OPENCODE_CONFIG",
        "OPENCODE_CONFIG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    unavailable = {
        marker
        for marker, package in {"requires_memray": "memray", "requires_torch": "torch"}.items()
        if importlib.util.find_spec(package) is None
    }
    for item in items:
        for marker in unavailable:
            if item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason=f"optional provider unavailable: {marker}"))
