from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from flameox import __version__
from flameox.setup import SetupFailure, mcp_launcher, prepare_providers


def test_mcp_launcher_pins_python_release_and_provider_extras() -> None:
    command, args = mcp_launcher(["memray", "nsight-compute", "py-spy"])

    assert command == "uvx"
    assert args == [
        "--python",
        "3.12",
        "--from",
        f"flameox[cpu,memory]=={__version__}",
        "flameox",
    ]


def test_preparation_runs_and_returns_the_same_exact_uvx_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    prepared = prepare_providers(["memray", "py-spy", "memray"], tmp_path)

    requirement = f"flameox[cpu,memory]=={__version__}"
    assert prepared.requested_providers == ["memray", "py-spy"]
    assert prepared.prepared_managed_providers == ["memray", "py-spy"]
    assert prepared.preparation_status == "prepared"
    assert prepared.restart_required is True
    assert calls == [
        [
            "/usr/bin/uvx",
            "--python",
            "3.12",
            "--from",
            requirement,
            "flameox",
            "--version",
        ]
    ]
    assert prepared.launcher_command == "uvx"
    assert prepared.launcher_args == [
        "--python",
        "3.12",
        "--from",
        requirement,
        "flameox",
        "mcp",
        "serve",
        "--project-root",
        str(tmp_path),
    ]


def test_host_only_preparation_returns_guidance_without_requiring_uvx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: None)

    prepared = prepare_providers(["nsight-compute"], tmp_path)

    assert prepared.prepared_managed_providers == []
    assert [item.provider_id for item in prepared.external_requirements] == ["nsight-compute"]
    assert prepared.preparation_command == []
    assert prepared.preparation_status == "not_applicable"
    assert prepared.restart_required is False


def test_uvx_is_required_before_managed_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: None)

    with pytest.raises(SetupFailure, match="requires uvx"):
        prepare_providers(["memray"], tmp_path)


def test_uvx_failure_is_normalized_as_setup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr(
        "flameox.setup.subprocess.run",
        lambda command, **_kwargs: CompletedProcess(command, 2),
    )

    with pytest.raises(SetupFailure, match="status 2"):
        prepare_providers(["memray"], tmp_path)


def test_uvx_start_failure_is_normalized_as_setup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        raise PermissionError("denied")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr("flameox.setup.subprocess.run", fail)

    with pytest.raises(SetupFailure, match="could not prepare"):
        prepare_providers(["memray"], tmp_path)


def test_unknown_provider_is_rejected_before_preparation(tmp_path: Path) -> None:
    with pytest.raises(SetupFailure, match="Unknown provider"):
        prepare_providers(["mystery"], tmp_path)
