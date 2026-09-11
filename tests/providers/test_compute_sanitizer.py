from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CaptureTarget, RequestLimits


@pytest.mark.parametrize("error_count", [1, 133])
def test_compute_sanitizer_capture_emits_typed_xml_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_count: int
) -> None:
    executable = tmp_path / "bin" / "compute-sanitizer"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--save') + 1])\n"
        f"count = {error_count}\n"
        "limit = int(arguments[arguments.index('--print-limit') + 1]) "
        "if '--print-limit' in arguments else 100\n"
        "if limit: count = min(count, limit)\n"
        "record = '''<record><kind>Precise</kind><level>Error</level>\n"
        "<what><text>Invalid write: Access is out of bounds</text><size>4</size></what>\n"
        "<where><func>write_values</func><path>kernel.cu</path><line>8</line></where>\n"
        "</record>'''\n"
        "output.write_text('<ComputeSanitizerOutput>' + record * count "
        "+ '</ComputeSanitizerOutput>')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(max_rows=1000)
        )
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="compute-sanitizer",
                    capture_arguments={"tool": "memcheck"},
                ),
                "sanitizer.failures",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"] == {
        "id": "compute-sanitizer",
        "version": "flameox.workers.compute_sanitizer/v1",
    }
    capture = result["capture"]
    assert isinstance(capture, dict)
    execution = capture["executions"][0]
    assert execution["capture_argv"][:7] == [
        "compute-sanitizer",
        "--tool",
        "memcheck",
        "--print-limit",
        "0",
        "--xml",
        "--save",
    ]
    blocks = result["blocks"]
    assert isinstance(blocks, list)
    assert blocks[0]["values"] == {"memory_access": error_count}
    assert result["coverage"]["rows_observed"] == error_count
