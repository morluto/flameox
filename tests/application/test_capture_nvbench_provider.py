"""Process-level NVBench capture tests using a fake benchmark executable."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from flameox.adapters.nvbench import NvbenchExtractor
from flameox.application import CaptureResult, CaptureService, ExecutionPolicy
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    ExecutionStatus,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment


def _fake_nvbench_bench(path: Path) -> None:
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
        (sidecar_dir / "1.bin").write_bytes(struct.pack("<2f", 0.99, 0.88))

if mode.startswith("nonzero"):
    raise SystemExit(7)
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


async def _capture(
    tmp_path: Path,
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
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="bench",
        adapter="nvbench",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    return service, await service.execute(plan.plan_token)


def _benchmark_registrations(result: CaptureResult) -> list[ArtifactRegistration]:
    return [item for item in result.run.artifacts if item.kind is ArtifactKind.BENCHMARK_SAMPLES]


@pytest.mark.anyio
async def test_nvbench_successful_capture_registers_declared_sidecars(
    tmp_path: Path,
) -> None:
    service, result = await _capture(tmp_path, mode="ok")

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    benchmark_regs = _benchmark_registrations(result)
    roles = {item.role for item in benchmark_regs}
    assert "primary" in roles
    assert "nvbench_sidecar" in roles
    sidecar_regs = [item for item in benchmark_regs if item.role == "nvbench_sidecar"]
    assert len(sidecar_regs) == 1
    assert sidecar_regs[0].display_name == "nvbench.json-bin/0.bin"
    display_names = {item.display_name for item in benchmark_regs}
    assert "nvbench.json-bin/1.bin" not in display_names
    extracted = NvbenchExtractor(service.workspace).extract(result.run.run_id)
    assert extracted.measurement_count == 3
    assert extracted.benchmark_count == 1


@pytest.mark.parametrize(
    ("mode", "expected_roles", "expected_limitation_codes"),
    (
        ("malformed", {"primary"}, {"expected_output_invalid"}),
        ("missing_sidecar", {"primary"}, {"expected_output_invalid"}),
        ("nonzero", {"primary", "nvbench_sidecar"}, {"nonzero_exit"}),
        (
            "nonzero_malformed",
            {"primary"},
            {"nonzero_exit", "expected_output_invalid"},
        ),
    ),
    ids=(
        "malformed-success",
        "missing-sidecar-success",
        "valid-nonzero",
        "malformed-nonzero",
    ),
)
@pytest.mark.anyio
async def test_nvbench_failed_attempts_preserve_only_valid_partial_artifacts(
    tmp_path: Path,
    mode: str,
    expected_roles: set[str],
    expected_limitation_codes: set[str],
) -> None:
    service, result = await _capture(tmp_path, mode=mode)

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert {item.role for item in _benchmark_registrations(result)} == expected_roles
    assert not list(service.workspace.paths.quarantine.glob("*/manifest.json"))
    limitation_codes = {detail.code for detail in result.run.limitation_details}
    assert limitation_codes & {"nonzero_exit", "expected_output_invalid"} == (
        expected_limitation_codes
    )
