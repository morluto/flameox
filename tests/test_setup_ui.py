from __future__ import annotations

from typing import cast

import pytest
from questionary import Choice

from flameox import setup_ui
from flameox.adapters.client_setup import SetupClient
from flameox.application.setup import SetupInspection


class _CheckboxQuestion:
    def __init__(self, choices: list[Choice]) -> None:
        self.choices = choices

    def ask(self) -> list[SetupClient]:
        return [cast(SetupClient, choice.value) for choice in self.choices if choice.checked]


def test_connect_preselects_detected_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def checkbox(message: str, *, choices: list[Choice]) -> _CheckboxQuestion:
        assert message == "Select MCP clients to connect:"
        return _CheckboxQuestion(choices)

    monkeypatch.setattr("flameox.setup_ui.questionary.checkbox", checkbox)
    inspection = SetupInspection(
        active_version=None,
        active_executable=None,
        configured_clients=(),
        detected_clients=(SetupClient.CLAUDE, SetupClient.GEMINI),
        installed_versions=(),
    )

    selected = setup_ui.choose_clients(inspection, remove=False)

    assert selected == (SetupClient.CLAUDE, SetupClient.GEMINI)
