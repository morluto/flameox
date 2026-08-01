from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from flameox.application import (
    ConfigureWorkloadRequest,
    WorkloadConfig,
    WorkloadService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


def _request(
    name: str,
    *,
    operation: Literal["create", "replace"] = "create",
    argv: tuple[str, ...] = ("python", "-c", "print('ok')"),
    expected_configuration_id: str | None = None,
) -> ConfigureWorkloadRequest:
    return ConfigureWorkloadRequest(
        name=name,
        operation=operation,
        config=WorkloadConfig(argv=argv),
        expected_configuration_id=expected_configuration_id,
    )


def test_configure_workload_is_idempotent_and_records_configuration_source(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    request = _request("probe")

    first = service.configure(request)
    config_text = (tmp_path / "flameox.toml").read_text()
    second = service.configure(request)

    assert first.action == "created"
    assert first.changed_paths == ("flameox.toml",)
    assert first.configuration_source == "agent"
    assert second.action == "unchanged"
    assert second.configuration_id == first.configuration_id
    assert second.workload_definition_id == first.workload_definition_id
    assert second.changed_paths == ()
    assert (tmp_path / "flameox.toml").read_text() == config_text
    assert service.list_declared(kind="workload", limit=10).workflows[0].name == "probe"


def test_replace_requires_current_configuration_digest_and_preserves_unrelated_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """# keep this project note
schema_version = 1

[workloads.alpha]
argv = ["python", "-c", "print('alpha')"]

[workloads.other]
# keep this unrelated workload
argv = ["python", "-c", "print('other')"]

[experiments.compare]
workload = "alpha"
variants = ["baseline", "candidate"]
"""
    )
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    current = service.configuration_status()
    assert current.status == "valid"
    assert current.configuration_id is not None
    assert current.workload_names == ("alpha", "other")
    assert current.next_tool == "list_declared_workflows"
    original = (tmp_path / "flameox.toml").read_text()

    with pytest.raises(DomainError) as stale:
        service.configure(
            _request(
                "alpha",
                operation="replace",
                argv=("python", "-c", "print('new alpha')"),
                expected_configuration_id="sha256:" + "0" * 64,
            )
        )

    assert stale.value.code is ErrorCode.REVISION_CONFLICT
    assert (tmp_path / "flameox.toml").read_text() == original

    updated = service.configure(
        _request(
            "alpha",
            operation="replace",
            argv=("python", "-c", "print('new alpha')"),
            expected_configuration_id=current.configuration_id,
        )
    )
    updated_text = (tmp_path / "flameox.toml").read_text()

    assert updated.action == "updated"
    assert "# keep this project note" in updated_text
    assert "# keep this unrelated workload" in updated_text
    assert "[workloads.other]" in updated_text
    assert "[experiments.compare]" in updated_text
    assert "print('new alpha')" in updated_text
    assert "print('alpha')" not in updated_text
    assert updated.configuration_source == "agent"


def test_invalid_configuration_is_reported_and_never_overwritten(tmp_path: Path) -> None:
    invalid = """schema_version = 1

[workloads.probe]
argv = ["python", "-c", "print('ok')"]

[experiments.broken]
workload = "missing"
variants = ["one", "two"]
"""
    (tmp_path / "flameox.toml").write_text(invalid)
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)

    status = service.configuration_status()
    assert status.status == "invalid"
    assert status.config_path == "flameox.toml"
    assert status.configuration_id is None
    assert status.next_tool is None
    assert len(status.diagnostics) == 1
    assert len(status.diagnostics[0]) <= 512

    with pytest.raises(DomainError) as refused:
        service.configure(_request("new"))

    assert refused.value.code is ErrorCode.WORKSPACE_INVALID
    assert (tmp_path / "flameox.toml").read_text() == invalid
