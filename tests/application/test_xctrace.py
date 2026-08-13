from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flameox.application.xctrace import (
    XctraceImportRequest,
    XctraceService,
    _archive_trace_bundle,
)
from flameox.domain import DomainError, ProcessResult, process_termination_from_returncode
from flameox.execution import ExecutionOutcome, ExecutionRequest, ProcessContainment
from flameox.storage import RunStore, Workspace

pytestmark = pytest.mark.integration


class _Broker:
    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        self.requests.append(request)
        arguments = request.argv[1:]
        stdout = b""
        if arguments == ("xctrace", "version"):
            stdout = b"xctrace version 16.0 (17F42)\n"
        elif arguments == ("xctrace", "list", "templates"):
            stdout = b"== Standard Templates ==\nMetal System Trace\nTime Profiler\n"
        elif arguments[:2] == ("xctrace", "export"):
            output = Path(arguments[arguments.index("--output") + 1])
            output.write_text("<?xml version='1.0'?><trace-toc><run number='1' /></trace-toc>")
        return ExecutionOutcome(
            process=ProcessResult(termination=process_termination_from_returncode(0)),
            stdout=stdout,
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


@pytest.mark.anyio
async def test_xctrace_import_preserves_native_bundle_and_bounded_toc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    trace = tmp_path / "metal.trace"
    (trace / "Data").mkdir(parents=True)
    (trace / "Data" / "run.data").write_bytes(b"native trace bytes")
    broker = _Broker()
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which", lambda _name, path=None: sys.executable
    )
    monkeypatch.setattr("flameox.application.xctrace.sys.platform", "darwin")

    result = await XctraceService(workspace, broker=broker).import_trace(
        XctraceImportRequest(trace_path=trace)
    )

    assert result.xctrace_version == "xctrace version 16.0 (17F42)"
    assert result.native_member_count == 1
    assert result.native_byte_length == len(b"native trace bytes")
    run = RunStore(workspace).read(result.run_id)
    assert [item.role for item in run.artifacts] == ["native_trace_bundle", "xctrace_toc"]
    assert all(item.sensitivity == "sensitive" for item in run.artifacts)
    assert any(request.argv[1:3] == ("xctrace", "export") for request in broker.requests)


def test_trace_archiving_rejects_symlink_members(tmp_path: Path) -> None:
    trace = tmp_path / "unsafe.trace"
    trace.mkdir()
    target = tmp_path / "outside"
    target.write_text("sensitive")
    os.symlink(target, trace / "linked")

    with pytest.raises(DomainError, match="cannot contain links"):
        _archive_trace_bundle(trace, tmp_path / "trace.tar", max_bytes=1024)
