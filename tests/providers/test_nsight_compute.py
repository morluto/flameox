from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.providers.contracts import ProviderAnalysis
from flameox.providers.nsight_compute import find_report_interface
from flameox.runtime_contracts import CaptureTarget, PathSource, RuntimeFailure
from flameox.stateless import AnalysisRuntime


def test_explicit_report_fails_as_unavailable_without_vendor_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "kernel.ncu-rep"
    report.write_bytes(b"native-report")
    monkeypatch.setattr("flameox.providers.nsight_compute.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "flameox.providers.nsight_compute.find_report_interface", lambda _executable: None
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("gpu.kernel_metrics", [PathSource(path=str(report))], {})
    finally:
        runtime.close()

    assert failure.value.code == "UNAVAILABLE_CAPABILITY"
    assert not (tmp_path / ".flameox").exists()


def test_vendor_interface_is_resolved_from_explicit_ncu_install(tmp_path: Path) -> None:
    executable = tmp_path / "bin" / "ncu"
    executable.parent.mkdir()
    executable.touch()
    interface = tmp_path / "extras" / "python" / "ncu_report.py"
    interface.parent.mkdir(parents=True)
    interface.touch()

    assert find_report_interface(executable) == interface


@pytest.mark.process
def test_nsight_compute_capture_uses_typed_bounded_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "bin" / "ncu"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        f"pathlib.Path({str(calls)!r}).write_text(' '.join(arguments))\n"
        "output = pathlib.Path(arguments[arguments.index('--export') + 1])\n"
        "output.write_bytes(b'native-report')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        monkeypatch.setattr(
            runtime.nsight_compute,
            "analyze",
            lambda path, **_kwargs: ProviderAnalysis(
                provider_id="nsight-compute",
                provider_version="test",
                blocks=[
                    {"type": "metrics", "values": {"report_count": 1}},
                    {"type": "table", "rows": [{"report_size": path.stat().st_size}]},
                ],
                rows_observed=1,
                complete=True,
                limitations=[],
            ),
        )
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    provider_id="nsight-compute",
                    capture_arguments={
                        "replay_mode": "kernel",
                        "launch_skip": 2,
                        "launch_count": 3,
                        "section": ["SpeedOfLight"],
                    },
                ),
                "gpu.kernel_metrics",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"] == {"id": "nsight-compute", "version": "test"}
    assert result["blocks"][1]["rows"] == [{"report_size": len(b"native-report")}]
    invocation = calls.read_text()
    assert "--replay-mode kernel" in invocation
    assert "--launch-skip 2 --launch-count 3" in invocation
    assert "--section SpeedOfLight" in invocation
