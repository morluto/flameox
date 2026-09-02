from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.providers.nsight_compute import find_report_interface
from flameox.runtime_contracts import CaptureTarget, PathSource, RuntimeFailure
from flameox.stateless import AnalysisRuntime


def _write_fake_report_interface(executable: Path) -> None:
    interface = executable.parent.parent / "extras" / "python" / "ncu_report.py"
    interface.parent.mkdir(parents=True)
    interface.touch()


def test_explicit_report_fails_as_unavailable_without_vendor_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "kernel.ncu-rep"
    report.write_bytes(b"native-report")
    monkeypatch.setattr("flameox.providers.nsight_compute.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "flameox.providers.nsight_compute.find_report_interface", lambda _executable: None
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
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


def test_capture_rejects_missing_vendor_interface_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "executed"
    executable = tmp_path / "bin" / "ncu"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent))
    monkeypatch.setattr(
        "flameox.providers.nsight_compute.find_report_interface", lambda _executable: None
    )

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "pass"],
                        cwd=str(tmp_path),
                        provider_id="nsight-compute",
                    ),
                    "gpu.kernel_metrics",
                )
        finally:
            runtime.close()

        assert failure.value.code == "UNAVAILABLE_CAPABILITY"
        assert failure.value.details["provider_id"] == "nsight-compute"

    anyio.run(exercise)
    assert not marker.exists()


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
    _write_fake_report_interface(executable)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
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
                    cwd=str(tmp_path),
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


@pytest.mark.process
def test_preserved_native_capture_survives_analysis_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin" / "ncu"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--export') + 1])\n"
        "output.write_bytes(b'authoritative-native-report')\n"
    )
    executable.chmod(0o755)
    _write_fake_report_interface(executable)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> tuple[str, dict[str, Any]]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")

        def fail_analysis(_path: Path, **_kwargs: object) -> ProviderAnalysis:
            raise ProviderFailure("DECODE_FAILURE", "fixture decoder rejected report")

        monkeypatch.setattr(runtime.nsight_compute, "analyze", fail_analysis)
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="nsight-compute",
                ),
                "gpu.kernel_metrics",
                preserve=True,
            )
            return result["preserved"]["evidence_id"], result
        finally:
            runtime.close()

    evidence_id, result = anyio.run(exercise)
    assert result["capture"]["executions"][0]["status"] == "succeeded"
    assert result["analysis_failure"] == {
        "code": "DECODE_FAILURE",
        "message": "fixture decoder rejected report",
        "details": {},
    }

    restarted = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        manifest = restarted.read_evidence(evidence_id)
        projection = restarted.repository.read_agent_projection(evidence_id)
    finally:
        restarted.close()
    assert projection["body"]["analysis_request"]["failure"] == {"code": "DECODE_FAILURE"}
    native = next(
        item for item in manifest["body"]["artifacts"] if item["format"] == "nsight-compute"
    )
    payload = (
        tmp_path
        / ".flameox"
        / "artifacts"
        / "sha256"
        / native["sha256"][:2]
        / native["sha256"]
        / "payload"
    )
    assert payload.read_bytes() == b"authoritative-native-report"
