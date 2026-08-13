from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from flameox import __version__
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.domain import CapabilityExtra, ProcessResult, process_termination_from_returncode
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)

pytestmark = pytest.mark.unit


class _ProviderBroker(SubprocessBroker):
    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    def run_sync(self, request: ExecutionRequest, **_: Any) -> ExecutionOutcome:
        self.requests.append(request)
        argv = request.argv
        stdout = b""
        if argv[1:] == ("--version",):
            stdout = b"uv 0.9.0\n"
        elif argv[1] == "venv":
            root = Path(argv[-1])
            python = root / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
        elif argv[1:3] == ("pip", "install"):
            python = Path(argv[argv.index("--python") + 1])
            executable = python.parent / "py-spy"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
        elif argv[1:4] == ("-I", "-c", argv[3]):
            stdout = json.dumps(
                {
                    "executable": argv[0],
                    "prefix": str(Path(argv[0]).parent.parent),
                    "versions": {"flameox": __version__, "py-spy": "0.4.2"},
                }
            ).encode()
        else:  # pragma: no cover - a new setup subprocess must be reviewed explicitly
            raise AssertionError(argv)
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(0),
                cleanup_complete=True,
            ),
            stdout=stdout,
            stderr=b"",
            resolved_executable=Path(argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


def test_provider_setup_publishes_verified_environment_without_mutating_control_python(
    tmp_path: Path,
) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    broker = _ProviderBroker()
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=broker,
        uv_executable=str(uv),
    )

    runtime = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )

    assert runtime.python == runtime.root / "bin" / "python"
    assert runtime.executable == runtime.root / "bin" / "py-spy"
    assert runtime.receipt.distributions == {"flameox": __version__, "py-spy": "0.4.2"}
    assert runtime.receipt.python_sha256.startswith("sha256:")
    assert runtime.receipt.executable_sha256 is not None
    install = next(
        request for request in broker.requests if request.argv[1:3] == ("pip", "install")
    )
    assert install.argv[install.argv.index("--python") + 1].startswith(str(tmp_path / "providers"))
    assert "--no-config" in install.argv
    assert "--no-sources" in install.argv
    assert all(
        request.argv[0] != str(runtime.python) or request.argv[1] == "-I"
        for request in broker.requests
    )
    assert install.argv[install.argv.index("--python") + 1] != sys.executable
    assert f"flameox=={__version__}" in install.argv
    assert not any(value.startswith("flameox[") for value in install.argv)
    create = next(request for request in broker.requests if request.argv[1] == "venv")
    assert "--relocatable" in create.argv


def test_provider_discovery_is_passive_and_rejects_changed_executable(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    broker = _ProviderBroker()
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=broker,
        uv_executable=str(uv),
    )
    runtime = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )
    call_count = len(broker.requests)

    discovered = manager.find(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
    )

    assert discovered is not None
    assert discovered.receipt.environment_id == runtime.receipt.environment_id
    exact = manager.get(runtime.receipt.environment_id)
    assert exact is not None
    assert exact.receipt == runtime.receipt
    assert manager.get("../provider-runtime") is None
    assert manager.get("sha256:" + "f" * 64) is None
    assert len(broker.requests) == call_count

    assert runtime.executable is not None
    runtime.executable.write_text("#!/bin/sh\nexit 1\n")
    assert manager.find(extra=CapabilityExtra.CPU, requirement="py-spy>=0.4.2,<0.5") is None


@pytest.mark.parametrize("field", ("python_relative_path", "executable_relative_path"))
def test_provider_discovery_rejects_receipt_paths_outside_runtime(
    tmp_path: Path,
    field: str,
) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=_ProviderBroker(),
        uv_executable=str(uv),
    )
    runtime = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )
    outside = tmp_path / "outside"
    outside.write_text("#!/bin/sh\nexit 0\n")
    receipt_path = runtime.root / "provider-runtime.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = str(outside)
    receipt[field.replace("relative_path", "sha256")] = manager._sha256(outside)
    receipt_path.write_text(json.dumps(receipt))

    assert manager.get(runtime.receipt.environment_id) is None
