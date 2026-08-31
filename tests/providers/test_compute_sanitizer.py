from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.runtime_contracts import CaptureTarget
from flameox.stateless import AnalysisRuntime


def test_compute_sanitizer_capture_emits_typed_xml_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin" / "compute-sanitizer"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--save') + 1])\n"
        "output.write_text('''<?xml version=\"1.0\"?>\n"
        "<ComputeSanitizerOutput><record><kind>Precise</kind><level>Error</level>\n"
        "<what><text>Invalid write: Access is out of bounds</text><size>4</size></what>\n"
        "<where><func>write_values</func><path>kernel.cu</path><line>8</line></where>\n"
        "</record></ComputeSanitizerOutput>''')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
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
    assert execution["capture_argv"][:5] == [
        "compute-sanitizer",
        "--tool",
        "memcheck",
        "--xml",
        "--save",
    ]
    blocks = result["blocks"]
    assert isinstance(blocks, list)
    assert blocks[0]["values"] == {"memory_access": 1}
