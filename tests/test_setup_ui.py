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


class _SelectQuestion:
    def __init__(self, choices: list[Choice]) -> None:
        self.choices = choices

    def ask(self) -> str:
        return "configure"


def test_stale_bootstrap_does_not_offer_or_select_a_runtime_downgrade(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected: list[Choice] = []

    def select(message: str, *, choices: list[Choice]) -> _SelectQuestion:
        assert message == "What would you like to do?"
        selected.extend(choices)
        return _SelectQuestion(choices)

    monkeypatch.setattr("flameox.setup_ui.questionary.select", select)
    inspection = SetupInspection(
        active_version="0.1.5",
        active_executable=None,
        configured_clients=(SetupClient.CLAUDE,),
        detected_clients=(),
        installed_versions=("0.1.5",),
    )

    setup_ui.print_banner(inspection, bootstrap_version="0.1.1")
    banner = capsys.readouterr().out
    assert "this setup bootstrap (0.1.1) is older than the active runtime (0.1.5)" in banner
    assert "npx flameox@latest setup" in banner

    assert setup_ui.choose_action(inspection, "0.1.1") == "configure"
    assert [choice.title for choice in selected] == [
        "Connect or update MCP clients",
        "Disconnect MCP clients",
        "Verify connected clients and the active runtime",
        "Exit",
    ]
    assert setup_ui.effective_runtime_version(inspection, "0.1.1") == "0.1.5"


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
