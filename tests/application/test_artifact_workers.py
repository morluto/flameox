from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.artifact_workers import ArtifactWorker
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


def _read_response(tmp_path: Path, payload: str) -> dict[str, object]:
    worker = ArtifactWorker(Workspace.initialize(tmp_path))
    response = tmp_path / "response.json"
    response.write_text(payload)
    return worker._load_response(0, b"", response, name="test")


def test_worker_response_parses_success_payload_without_assuming_its_shape(tmp_path: Path) -> None:
    response = _read_response(tmp_path, '{"ok":true,"rows":[{"id":1}],"truncated":false}')

    assert response == {"ok": True, "rows": [{"id": 1}], "truncated": False}


def test_worker_response_preserves_declared_failure_code(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as error:
        _read_response(
            tmp_path,
            '{"ok":false,"code":"CAPABILITY_UNAVAILABLE","message":"not installed"}',
        )

    assert error.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert error.value.message == "not installed"


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"ok":false,"code":"UNKNOWN","message":"drift"}',
        '{"ok":false,"code":"PROCESS_FAILED"}',
        "not json",
    ],
)
def test_worker_response_refuses_malformed_or_drifted_envelopes(
    tmp_path: Path,
    payload: str,
) -> None:
    with pytest.raises(DomainError) as error:
        _read_response(tmp_path, payload)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
