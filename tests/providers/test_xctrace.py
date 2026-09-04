from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.providers.contracts import ProviderFailure
from flameox.providers.xctrace import XctraceProvider
from flameox.runtime_contracts import CaptureTarget
from flameox.stateless import AnalysisRuntime


@pytest.mark.unit
def test_xctrace_malformed_toc_is_a_typed_decode_failure(tmp_path: Path) -> None:
    toc = tmp_path / "toc.xml"
    toc.write_text("<trace><run></trace>")

    with pytest.raises(ProviderFailure) as failure:
        XctraceProvider.analyze(toc, max_rows=10, provider_version="test")

    assert failure.value.code == "DECODE_FAILURE"


@pytest.mark.process
def test_xctrace_capture_preserves_bundle_and_analyzes_bounded_toc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin" / "xcrun"
    executable.parent.mkdir()
    calls = tmp_path / "calls.txt"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        f"calls = pathlib.Path({str(calls)!r})\n"
        "with calls.open('a') as stream: stream.write(' '.join(arguments) + '\\n')\n"
        "output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "if arguments[1] == 'record':\n"
        "    output.mkdir()\n"
        "    (output / 'native.data').write_bytes(b'native')\n"
        "else:\n"
        "    output.write_text('<trace-toc><run name=\"captured\"/></trace-toc>')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="xctrace",
                    capture_arguments={"template": "Metal System Trace"},
                ),
                "trace.summary",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"]["id"] == "xctrace"
    assert result["blocks"][0]["values"] == {"toc_element_count": 2}
    assert result["blocks"][1]["rows"][0]["attributes"] == {"name": "captured"}
    record, export = calls.read_text().splitlines()
    assert "xctrace record --template Metal System Trace" in record
    assert "--launch --" in record
    assert export.startswith("xctrace export --input ")
    assert " --toc --output " in export
