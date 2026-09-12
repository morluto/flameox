from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from coverage import CoverageData

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    MAX_ROWS,
    CaptureTarget,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


@pytest.mark.process
def test_coverage_data_uses_isolated_reader_and_continuation(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("one = 1\ntwo = 2\nthree = 3\n")
    artifact = tmp_path / ".coverage"
    data = CoverageData(basename=str(artifact))
    data.add_lines({str(source): {1, 2, 3}})
    data.write()
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        first = runtime.analyze(
            "coverage.summary",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
        )
        second = runtime.analyze(
            "coverage.summary",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    assert first["provider"]["id"] == "coverage.py"
    assert first["blocks"][0]["values"] == {
        "file_count": 1,
        "line_count": 3,
        "arc_count": 0,
    }
    assert [row["line_from"] for row in first["blocks"][1]["rows"]] == [1, 2]
    assert [row["line_from"] for row in second["blocks"][1]["rows"]] == [3]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_provider_continuation_stops_at_a_truthfully_reported_bounded_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("\n" * 1_205)
    artifact = tmp_path / ".coverage"
    data = CoverageData(basename=str(artifact))
    data.add_lines({str(source): set(range(1, 1_206))})
    data.write()
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    pages: list[dict[str, Any]] = []
    continuation: str | None = None
    try:
        while True:
            page = runtime.analyze(
                "coverage.summary",
                [PathSource(path=str(artifact))],
                {},
                limits=RequestLimits(max_rows=100),
                continuation=continuation,
            )
            pages.append(page)
            continuation = page["continuation"]
            if continuation is None:
                break
    finally:
        runtime.close()

    rows = [row for page in pages for row in page["blocks"][1]["rows"]]
    assert len(rows) == MAX_ROWS + 1
    assert [row["line_from"] for row in rows] == list(range(1, MAX_ROWS + 2))
    assert pages[-1]["coverage"] == {
        "rows_returned": 1,
        "rows_observed": 1_205,
        "complete": False,
    }
    assert pages[-1]["truncation"] == {"reason": "provider_limit", "next_offset": 1_001}
    assert any("truncated to 1001" in item for item in pages[0]["limitations"])


@pytest.mark.process
def test_coverage_capture_uses_explicit_empty_config_and_native_data(tmp_path: Path) -> None:
    script = tmp_path / "covered.py"
    script.write_text("value = 1\nprint(value)\n")

    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=20)
        )
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(script)],
                    cwd=str(tmp_path),
                    provider_id="coverage",
                    capture_arguments={"branch": True, "source": [str(tmp_path)]},
                ),
                "coverage.summary",
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "succeeded"
            assert "--rcfile" in execution["capture_argv"]
            assert result["provider"]["id"] == "coverage.py"
            assert result["blocks"][0]["values"]["line_count"] >= 2
            assert result["inputs"][0]["format"] == "coverage"
            assert not (tmp_path / ".coverage").exists()
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_coverage_capture_rejects_workload_interpreter_without_provider(tmp_path: Path) -> None:
    workload_python = tmp_path / "python"
    workload_python.write_text("#!/bin/sh\nexit 7\n")
    workload_python.chmod(0o755)

    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=20)
        )
        try:
            with pytest.raises(RuntimeFailure) as raised:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[str(workload_python), "workload.py"],
                        cwd=str(tmp_path),
                        provider_id="coverage",
                    ),
                    "coverage.summary",
                )
        finally:
            runtime.close()

        assert raised.value.code == "UNAVAILABLE_CAPABILITY"
        assert "workload interpreter" in raised.value.message

    anyio.run(exercise)
