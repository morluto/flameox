from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CaptureTarget, RuntimeFailure


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


@pytest.mark.skipif(sys.platform == "win32", reason="Symlink creation requires platform privileges")
def test_vendor_reader_symlink_loop_has_a_typed_unavailable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flameox.runtime_contracts import PathSource

    vendor = tmp_path / "vendor"
    interface = vendor / "2026.1" / "extras" / "python" / "ncu_report.py"
    interface.parent.mkdir(parents=True)
    interface.symlink_to(interface.name)
    monkeypatch.setattr("flameox.providers.nsight_compute._INSTALL_ROOTS", (vendor,))
    monkeypatch.setenv("PATH", "")
    artifact = tmp_path / "native.ncu-rep"
    artifact.write_bytes(b"native report")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "gpu.kernel_metrics", [PathSource(path=str(artifact), format="nsight-compute")], {}
            )
        assert failure.value.code == "UNAVAILABLE_CAPABILITY"
        assert str(interface) not in str(failure.value)
    finally:
        runtime.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Symlink creation requires platform privileges")
def test_vendor_discovery_ignores_unavailable_newer_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flameox.providers.nsight_compute import find_report_interface

    valid = tmp_path / "2025.1" / "extras" / "python" / "ncu_report.py"
    invalid = tmp_path / "2026.1" / "extras" / "python" / "ncu_report.py"
    valid.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    valid.write_text("# official reader")
    invalid.symlink_to(invalid.name)
    monkeypatch.setattr("flameox.providers.nsight_compute._INSTALL_ROOTS", (tmp_path,))
    assert find_report_interface() == valid
