"""Process-level NVBench capture tests using a fake benchmark executable."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from flameox.application import CaptureResult, CaptureService, ExecutionPolicy
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CapabilityPermissionStatus,
    CapabilityReport,
    CapabilityStatus,
    CaptureStatus,
    ExecutionStatus,
    ProbeKind,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment


def _fake_nvbench_bench(path: Path) -> None:
    """Write a fake NVBench benchmark executable.

    The script reads ``--jsonbin <path>`` from argv, writes a JSON document
    to ``<path>``, and writes ``sample_count`` float32 values to
    ``<path>-bin/0.bin``.  An undeclared sibling ``<path>-bin/1.bin`` is
    also written to verify it is NOT registered.

    Modes:
    - ``mode=ok``: exit 0, write valid JSON + sidecar + undeclared sibling.
    - ``mode=malformed``: exit 0, write invalid JSON (no sidecars).
    - ``mode=missing_sidecar``: exit 0, write valid JSON but no sidecar dir.
    - ``mode=nonzero``: exit 7, write valid JSON + sidecar + undeclared sibling.
    - ``mode=nonzero_malformed``: exit 7, write invalid JSON.
    """
    path.write_text(
        """#!/usr/bin/env python3
import json as _json
import pathlib
import struct
import sys

argv = sys.argv
jsonbin_idx = argv.index("--jsonbin")
output = pathlib.Path(argv[jsonbin_idx + 1])
mode = "ok"
for arg in argv:
    if arg.startswith("mode="):
        mode = arg[len("mode="):]

samples = [0.003120, 0.003145, 0.003108]
sample_count = len(samples)

if mode == "malformed" or mode == "nonzero_malformed":
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{not valid json")
else:
    doc = {
        "meta": {
            "argv": ["bench", "--jsonbin", str(output)],
            "version": {
                "json": {"major": 1, "minor": 0, "patch": 0, "string": "1.0.0"},
                "nvbench": {"major": 0, "minor": 1, "patch": 0, "string": "0.1.0"},
            },
        },
        "devices": [{"id": 0, "name": "NVIDIA GeForce RTX 4090"}],
        "benchmarks": [
            {
                "name": "cub.bench.scan",
                "index": 0,
                "axes": [],
                "states": [
                    {
                        "name": "[T=I32 Elements=2^16]",
                        "device": 0,
                        "type_config_index": 0,
                        "is_skipped": False,
                        "summaries": [
                            {
                                "tag": "nv/json/bin:sample_times",
                                "name": "Sample Times File",
                                "description": "Binary file.",
                                "hint": "file/sample_times",
                                "hide": "Not needed in table.",
                                "data": [
                                    {
                                        "name": "filename",
                                        "type": "string",
                                        "value": str(output.name) + "-bin/0.bin",
                                    },
                                    {
                                        "name": "size",
                                        "type": "int64",
                                        "value": str(sample_count),
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json.dumps(doc))
    if mode != "missing_sidecar":
        sidecar_dir = output.parent / (output.name + "-bin")
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        (sidecar_dir / "0.bin").write_bytes(
            struct.pack(f"<{sample_count}f", *samples)
        )
        # Undeclared sibling that must NOT be registered
        (sidecar_dir / "1.bin").write_bytes(
            struct.pack("<2f", 0.99, 0.88)
        )

if mode.startswith("nonzero"):
    raise SystemExit(7)
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_service(
    workspace: Workspace,
    bench: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CaptureService:
    report = CapabilityReport(
        adapter="nvbench",
        status=CapabilityStatus.AVAILABLE,
        executable=str(bench),
        version="fake-nvbench",
        supported_modes=("benchmark",),
        supported_formats=("nvbench-json", "nvbench-jsonbin"),
        permission_status=CapabilityPermissionStatus.GRANTED,
        probe_kind=ProbeKind.ACTIVE,
    )
    service = CaptureService(workspace)
    monkeypatch.setattr(service.capabilities, "get", lambda _adapter: report)

    async def probe(_adapter: str, *, refresh: bool = False) -> CapabilityReport:
        assert refresh
        return report

    monkeypatch.setattr(service.capabilities, "probe", probe)
    return service


async def _capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
) -> tuple[CaptureService, CaptureResult]:
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    bench = tmp_path / "fake-bench"
    _fake_nvbench_bench(bench)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.bench]
argv = ["./fake-bench", "mode={mode}"]
timeout_seconds = 10
"""
    )
    service = _make_service(workspace, bench, monkeypatch)
    plan = await service.plan(
        workload_name="bench",
        adapter="nvbench",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    return service, await service.execute(plan.plan_id)


def _benchmark_registrations(result: CaptureResult) -> list[ArtifactRegistration]:
    return [item for item in result.run.artifacts if item.kind is ArtifactKind.BENCHMARK_SAMPLES]


@pytest.mark.anyio
async def test_nvbench_successful_capture_registers_declared_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = await _capture(tmp_path, monkeypatch, mode="ok")

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    benchmark_regs = _benchmark_registrations(result)
    roles = {item.role for item in benchmark_regs}
    assert "primary" in roles
    assert "nvbench_sidecar" in roles
    sidecar_regs = [item for item in benchmark_regs if item.role == "nvbench_sidecar"]
    assert len(sidecar_regs) == 1
    # display_name must match the declared relative path
    assert sidecar_regs[0].display_name == "nvbench.json-bin/0.bin"
    # The undeclared sibling (1.bin) must not be registered
    display_names = {item.display_name for item in benchmark_regs}
    assert "nvbench.json-bin/1.bin" not in display_names


@pytest.mark.anyio
async def test_nvbench_successful_malformed_json_is_preserved_as_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = await _capture(tmp_path, monkeypatch, mode="malformed")

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert [item.role for item in _benchmark_registrations(result)] == ["primary"]
    assert not list(service.workspace.paths.quarantine.glob("*/manifest.json"))
    assert any(detail.code == "expected_output_invalid" for detail in result.run.limitation_details)


@pytest.mark.anyio
async def test_nvbench_successful_missing_sidecar_is_preserved_as_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = await _capture(tmp_path, monkeypatch, mode="missing_sidecar")

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert [item.role for item in _benchmark_registrations(result)] == ["primary"]
    assert not list(service.workspace.paths.quarantine.glob("*/manifest.json"))


@pytest.mark.anyio
async def test_nvbench_nonzero_preserves_partial_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = await _capture(tmp_path, monkeypatch, mode="nonzero")

    assert result.run.execution_status is ExecutionStatus.FAILED
    # Partial artifacts are preserved (not quarantined)
    assert result.run.capture_status is CaptureStatus.REGISTERED
    benchmark_regs = _benchmark_registrations(result)
    roles = {item.role for item in benchmark_regs}
    assert "primary" in roles
    assert "nvbench_sidecar" in roles
    sidecar_regs = [item for item in benchmark_regs if item.role == "nvbench_sidecar"]
    assert len(sidecar_regs) == 1
    assert sidecar_regs[0].display_name == "nvbench.json-bin/0.bin"
    # Undeclared sibling must not be registered
    display_names = {item.display_name for item in benchmark_regs}
    assert "nvbench.json-bin/1.bin" not in display_names
    # Must have nonzero limitation; valid output has no expected_output_invalid
    limitation_codes = {detail.code for detail in result.run.limitation_details}
    assert "nonzero_exit" in limitation_codes
    # Must NOT be quarantined
    manifests = list(service.workspace.paths.quarantine.glob("*/manifest.json"))
    assert not manifests


@pytest.mark.anyio
async def test_nvbench_nonzero_malformed_preserves_json_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = await _capture(tmp_path, monkeypatch, mode="nonzero_malformed")

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    benchmark_regs = _benchmark_registrations(result)
    # The partial JSON is preserved as primary, but no sidecars
    roles = {item.role for item in benchmark_regs}
    assert "primary" in roles
    assert "nvbench_sidecar" not in roles
    # Must NOT be quarantined (partial preservation on nonzero)
    manifests = list(service.workspace.paths.quarantine.glob("*/manifest.json"))
    assert not manifests
    limitation_codes = {detail.code for detail in result.run.limitation_details}
    assert "nonzero_exit" in limitation_codes
    assert "expected_output_invalid" in limitation_codes
