from __future__ import annotations

import cProfile
import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.runtime_contracts import CaptureTarget, PathSource, RuntimeFailure
from flameox.stateless import AnalysisRuntime


def test_pstats_profile_is_bounded_deterministic_cpu_evidence(tmp_path: Path) -> None:
    profile = tmp_path / "profile.pstats"

    def work() -> int:
        return sum(range(20))

    profiler = cProfile.Profile()
    profiler.runcall(work)
    profiler.dump_stats(profile)
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="pstats", producer="cProfile")],
            {"metric": "cumulative_time_seconds"},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "python-pstats"
    assert result["blocks"][0]["values"]["metric"] == "cumulative_time_seconds"
    work_row = next(row for row in result["blocks"][1]["rows"] if row["function"] == "work")
    assert work_row["total_calls"] == 1
    assert work_row["primitive_calls"] == 1
    assert work_row["cumulative_time_seconds"] >= work_row["self_time_seconds"]
    assert any("no compatibility guarantee" in item for item in result["limitations"])


def test_pyspy_speedscope_profile_ranks_typed_frames(tmp_path: Path) -> None:
    profile = tmp_path / "profile.speedscope.json"
    profile.write_text(
        json.dumps(
            {
                "$schema": "https://www.speedscope.app/file-format-schema.json",
                "shared": {
                    "frames": [
                        {"name": "main", "file": "app.py", "line": 1},
                        {"name": "work", "file": "app.py", "line": 4},
                    ]
                },
                "profiles": [
                    {
                        "type": "sampled",
                        "name": "process 1",
                        "unit": "seconds",
                        "startValue": 0,
                        "endValue": 3,
                        "samples": [[0, 1], [0, 1], [0]],
                        "weights": [1, 1, 1],
                    }
                ],
            }
        )
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="py-spy", producer="py-spy")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "py-spy-speedscope"
    assert result["blocks"][0]["values"] == {"frame_count": 2, "sample_count": 3}
    assert result["blocks"][1]["rows"][0]["function"] == "work"
    assert result["blocks"][1]["rows"][0]["self_weight"] == 2


def test_pyspy_speedscope_profile_keeps_resolved_samples_when_one_stack_is_empty(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.speedscope.json"
    profile.write_text(
        json.dumps(
            {
                "shared": {"frames": [{"name": "work", "file": "app.py", "line": 4}]},
                "profiles": [
                    {
                        "type": "sampled",
                        "samples": [[], [0]],
                        "weights": [1, 2],
                    }
                ],
            }
        )
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="py-spy", producer="py-spy")],
            {},
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"] == {
        "frame_count": 1,
        "sample_count": 2,
        "unresolved_sample_count": 1,
    }
    assert result["blocks"][1]["rows"] == [
        {
            "frame_index": 0,
            "function": "work",
            "file": "app.py",
            "line": 4,
            "column": None,
            "self_weight": 2.0,
            "inclusive_weight": 2.0,
        }
    ]
    assert any("1 of 2" in limitation for limitation in result["limitations"])


def test_collapsed_perf_stacks_are_bounded_cpu_evidence(tmp_path: Path) -> None:
    profile = tmp_path / "perf.collapsed"
    profile.write_text("main;scan 7\nmain;parse 2\nmain;scan 3\n")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="perf", producer="perf")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "perf-collapsed"
    assert result["blocks"][0]["values"]["sample_count"] == 12
    assert result["blocks"][1]["rows"] == [
        {"function": "scan", "self_samples": 10, "unit": "samples"},
        {"function": "parse", "self_samples": 2, "unit": "samples"},
    ]


def test_pyspy_capture_rejects_ambient_path_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload_python = sys.executable
    unmanaged_bin = tmp_path / "control" / "bin"
    unmanaged_bin.mkdir(parents=True)
    unmanaged_python = unmanaged_bin / "python"
    unmanaged_python.symlink_to(workload_python)
    monkeypatch.setattr("flameox.stateless.sys.executable", str(unmanaged_python))
    executable = tmp_path / "bin" / "py-spy"
    executable.parent.mkdir()
    profile = {
        "shared": {"frames": [{"name": "captured"}]},
        "profiles": [{"type": "sampled", "samples": [[0]], "weights": [1]}],
    }
    executable.write_text(
        f"#!{workload_python}\n"
        "import json, pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        f"output.write_text(json.dumps({profile!r}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[workload_python, "-c", "print('target')"],
                        provider_id="py-spy",
                        capture_arguments={"rate": 250, "gil": True},
                    ),
                    "cpu.hotspots",
                )
        finally:
            runtime.close()

        assert failure.value.code == "UNAVAILABLE_CAPABILITY"

    anyio.run(exercise)


@pytest.mark.process
def test_perf_capture_converts_native_data_in_session_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin" / "perf"
    executable.parent.mkdir()
    calls = tmp_path / "calls.txt"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        f"calls = pathlib.Path({str(calls)!r})\n"
        "with calls.open('a') as stream: stream.write(' '.join(arguments) + '\\n')\n"
        "if arguments[0] == 'record':\n"
        "    output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "    output.write_bytes(b'PERFILE2')\n"
        "else:\n"
        "    print('target 1 [000] 1.0: cycles:')\n"
        "    print('        1000 leaf+0x1 (app)')\n"
        "    print('        2000 root+0x2 (app)')\n"
        "    print()\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    provider_id="perf",
                    capture_arguments={"frequency": 199, "call_graph": "fp"},
                ),
                "cpu.hotspots",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"]["id"] == "perf-collapsed"
    assert result["blocks"][1]["rows"] == [
        {"function": "leaf", "self_samples": 1, "unit": "samples"}
    ]
    record, script = calls.read_text().splitlines()
    assert "record --freq 199 --call-graph fp" in record
    assert script.startswith("script --input ")


def test_node_capture_uses_explicit_profile_name_and_analyzes_v8_output(tmp_path: Path) -> None:
    executable = tmp_path / "fake-node"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

directory = Path(next(
    value.split("=", 1)[1]
    for value in sys.argv
    if value.startswith("--cpu-prof-dir=")
))
name = next(
    value.split("=", 1)[1]
    for value in sys.argv
    if value.startswith("--cpu-prof-name=")
)
(directory / name).write_text(json.dumps({
    "nodes": [{
        "id": 1,
        "callFrame": {
            "functionName": "captured", "url": "app.js",
            "lineNumber": 0, "columnNumber": 0
        },
        "hitCount": 1,
        "children": []
    }],
    "samples": [1]
}))
"""
    )
    executable.chmod(0o755)

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(argv=[str(executable), "app.js"], provider_id="node-cpu-profile"),
                "cpu.hotspots",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    capture = result["capture"]
    assert isinstance(capture, dict)
    executions = capture["executions"]
    assert isinstance(executions, list)
    capture_argv = executions[0]["capture_argv"]
    assert capture_argv[1] == "--cpu-prof"
    assert "--cpu-prof-name=profile.cpuprofile" in capture_argv
    provider = result["provider"]
    assert isinstance(provider, dict)
    assert provider["id"] == "v8-cpu-profile"
