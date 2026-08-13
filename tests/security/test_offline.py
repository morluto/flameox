from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from flameox.analysis import RecipeService
from flameox.application import (
    ImportArtifactRequest,
    ImportService,
    workspace_status,
)
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


def test_control_process_operations_do_not_open_network_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[object] = []

    def refuse_connect(_socket: socket.socket, address: object) -> None:
        attempts.append(address)
        raise AssertionError(f"unexpected network connection: {address!r}")

    def refuse_create_connection(*args: Any, **kwargs: Any) -> None:
        attempts.append((args, kwargs))
        raise AssertionError("unexpected network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    workspace = Workspace.initialize(tmp_path)
    artifact = tmp_path / "local.bin"
    artifact.write_bytes(b"local")

    ImportService(workspace).import_artifact(ImportArtifactRequest(path=artifact))
    workspace_status(workspace)
    RecipeService(workspace).failures(limit=10)

    assert attempts == []
