from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from flameox import __version__
from flameox.application import CapabilityService
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.domain import (
    CapabilityExtra,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ProcessResult,
    process_termination_from_returncode,
)
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.storage import Workspace

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
        elif argv[1] == "build":
            output = Path(argv[argv.index("--out-dir") + 1])
            source = Path(argv[-1])
            wheel = output / "flameox-0.1.14-py3-none-any.whl"
            wheel.write_bytes((source / "src" / "flameox" / "fixture.py").read_bytes())
        elif argv[1:3] == ("pip", "compile"):
            lock = Path(argv[argv.index("--output-file") + 1])
            lock.write_text(
                "flameox==0.1.14 --hash=sha256:" + "a" * 64 + "\n"
                "py-spy==0.4.2 --hash=sha256:" + "b" * 64 + "\n"
            )
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
        elif Path(argv[0]).name == "git" and argv[1:] == ("rev-parse", "HEAD"):
            stdout = b"0123456789abcdef0123456789abcdef01234567\n"
        elif Path(argv[0]).name == "git" and argv[1] == "status":
            stdout = b" M src/flameox/fixture.py\n"
        elif argv[1:3] == ("-I", "-c"):
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


class _FailingProviderBroker(_ProviderBroker):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase

    def run_sync(self, request: ExecutionRequest, **kwargs: Any) -> ExecutionOutcome:
        is_failure = (self.phase == "build" and request.argv[1] == "build") or (
            self.phase == "install" and request.argv[1:3] == ("pip", "install")
        ) or (
            self.phase == "resolve" and request.argv[1:3] == ("pip", "compile")
        )
        if not is_failure:
            return super().run_sync(request, **kwargs)
        self.requests.append(request)
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(1),
                cleanup_complete=True,
            ),
            stdout=b"",
            stderr=(
                b"No solution found for flameox version"
                if self.phase in {"install", "resolve"}
                else b"wheel backend failed"
            ),
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


class _MutatingBuildBroker(_ProviderBroker):
    def run_sync(self, request: ExecutionRequest, **kwargs: Any) -> ExecutionOutcome:
        outcome = super().run_sync(request, **kwargs)
        if request.argv[1] == "build":
            source = Path(request.argv[-1])
            (source / "src" / "flameox" / "fixture.py").write_text("VALUE = 2\n")
        return outcome


def _source_checkout(root: Path) -> Path:
    (root / "src" / "flameox").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
    (root / "README.md").write_text("fixture\n")
    (root / "LICENSE").write_text("MIT\n")
    implementation = root / "src" / "flameox" / "fixture.py"
    implementation.write_text("VALUE = 1\n")
    return implementation


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
        package_source="index",
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
    assert "--require-hashes" in install.argv
    lock = Path(install.argv[install.argv.index("-r") + 1])
    assert runtime.receipt.installation_lock_sha256 == manager._sha256(lock)
    assert manager.get(runtime.receipt.environment_id) is not None
    for index in range(65):
        (manager.root / f"{index:064x}").mkdir()
    assert (
        manager.find_distribution(
            extra=CapabilityExtra.CPU,
            requirement="py-spy==0.4.2",
        )
        == runtime
    )
    assert (
        manager.find_distribution(
            extra=CapabilityExtra.CPU,
            requirement="py-spy==0.4.3",
        )
        is None
    )
    repeated = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )
    assert repeated.receipt.environment_id == runtime.receipt.environment_id
    lock.write_text("changed\n")
    assert manager.get(runtime.receipt.environment_id) is None
    assert not any(value.startswith("flameox[") for value in install.argv)
    create = next(request for request in broker.requests if request.argv[1] == "venv")
    assert "--relocatable" in create.argv


def test_provider_setup_builds_and_binds_the_active_source_wheel(tmp_path: Path) -> None:
    source = tmp_path / "source"
    implementation = _source_checkout(source)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    broker = _ProviderBroker()
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=broker,
        uv_executable=str(uv),
        package_source="source",
        flameox_source_root=source,
    )

    first = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )

    assert first.receipt.flameox_package_source == "local_wheel"
    assert first.receipt.flameox_package_sha256 is not None
    assert first.receipt.flameox_source_tree_sha256 is not None
    assert first.receipt.flameox_source_revision == "0123456789abcdef0123456789abcdef01234567"
    assert first.receipt.flameox_source_dirty is True
    build = next(request for request in broker.requests if request.argv[1] == "build")
    assert build.argv[2:4] == ("--wheel", "--out-dir")
    install = next(
        request for request in broker.requests if request.argv[1:3] == ("pip", "install")
    )
    installed_wheel = Path(install.argv[-2])
    assert installed_wheel.is_file()
    assert installed_wheel.parent.parent == tmp_path / "providers" / "wheels"
    assert installed_wheel.parent.name == first.receipt.flameox_package_sha256.removeprefix(
        "sha256:"
    )
    assert manager.get(first.receipt.environment_id) is not None

    installed_wheel.write_bytes(b"changed")
    assert manager.get(first.receipt.environment_id) is None

    receipt_path = first.root / "provider-runtime.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["flameox_package_source"] = "index"
    receipt_path.write_text(json.dumps(receipt))
    assert manager.get(first.receipt.environment_id) is None

    implementation.write_text("VALUE = 2\n")
    second = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )
    assert second.receipt.environment_id != first.receipt.environment_id
    assert second.receipt.flameox_package_sha256 != first.receipt.flameox_package_sha256


def test_provider_setup_reports_source_build_phase_and_cleans_staging(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_checkout(source)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=_FailingProviderBroker("build"),
        uv_executable=str(uv),
        package_source="source",
        flameox_source_root=source,
    )

    with pytest.raises(DomainError) as failure:
        manager.prepare(extra=CapabilityExtra.CPU, requirement="py-spy>=0.4.2,<0.5")

    assert failure.value.code is ErrorCode.PROCESS_FAILED
    assert failure.value.details["phase"] == "build_flameox_wheel"
    assert failure.value.retryable is True
    assert not tuple((tmp_path / "providers").glob(".flameox-wheel.*"))


def test_provider_setup_rejects_source_changes_during_build(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_checkout(source)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=_MutatingBuildBroker(),
        uv_executable=str(uv),
        package_source="source",
        flameox_source_root=source,
    )

    with pytest.raises(DomainError) as failure:
        manager.prepare(extra=CapabilityExtra.CPU, requirement="py-spy>=0.4.2,<0.5")

    assert "source changed" in failure.value.details["reason"]
    assert not tuple((tmp_path / "providers").glob(".flameox-wheel.*"))


def test_provider_setup_classifies_unsatisfiable_index_version_as_non_transient(
    tmp_path: Path,
) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=_FailingProviderBroker("resolve"),
        uv_executable=str(uv),
        package_source="index",
    )

    with pytest.raises(DomainError) as failure:
        manager.prepare(extra=CapabilityExtra.CPU, requirement="py-spy>=0.4.2,<0.5")

    assert failure.value.code is ErrorCode.PROCESS_FAILED
    assert failure.value.details["phase"] == "resolve_provider_lock"
    assert failure.value.retryable is False


def test_provider_setup_auto_detects_an_editable_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source checkout"
    _source_checkout(source)

    class _Distribution:
        @staticmethod
        def read_text(filename: str) -> str | None:
            assert filename == "direct_url.json"
            return json.dumps(
                {"url": source.as_uri(), "dir_info": {"editable": True}}
            )

    monkeypatch.setattr(metadata, "distribution", lambda _name: _Distribution())

    manager = ProviderRuntimeManager(tmp_path / "providers")

    assert manager.flameox_source_root == source.resolve()


def test_managed_py_spy_is_discoverable_after_verified_setup(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    runtimes = ProviderRuntimeManager(
        workspace.paths.records / "provider-runtimes",
        broker=_ProviderBroker(),
        uv_executable=str(uv),
        package_source="index",
    )
    runtimes.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )

    capabilities = CapabilityService(workspace)
    capabilities.provider_runtimes = runtimes
    report = capabilities.get("py-spy")

    assert report.status is CapabilityStatus.AVAILABLE
    assert report.executable is not None
    assert Path(report.executable).name == "py-spy"


def test_provider_discovery_is_passive_and_rejects_changed_executable(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    broker = _ProviderBroker()
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=broker,
        uv_executable=str(uv),
        package_source="index",
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


def test_verified_provider_use_detects_runtime_mutation(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=_ProviderBroker(),
        uv_executable=str(uv),
        package_source="index",
    )
    runtime = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )

    with pytest.raises(DomainError) as raised, manager.verified_use(runtime):
        assert runtime.executable is not None
        runtime.executable.write_text("#!/bin/sh\nexit 1\n")

    assert raised.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_provider_discovery_accepts_a_contained_python_symlink(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    manager = ProviderRuntimeManager(
        tmp_path / "providers",
        broker=_ProviderBroker(),
        uv_executable=str(uv),
        package_source="index",
    )
    runtime = manager.prepare(
        extra=CapabilityExtra.CPU,
        requirement="py-spy>=0.4.2,<0.5",
        executable_name="py-spy",
    )
    external_python = tmp_path / "managed-python"
    external_python.write_text("#!/bin/sh\nexit 0\n")
    external_python.chmod(0o755)
    runtime.python.unlink()
    runtime.python.symlink_to(external_python)
    receipt_path = runtime.root / "provider-runtime.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["python_sha256"] = manager._sha256(external_python)
    receipt["environment_tree_sha256"] = manager._tree_sha256(runtime.root)
    receipt_path.write_text(json.dumps(receipt))

    assert manager.get(runtime.receipt.environment_id) is not None


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
        package_source="index",
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
