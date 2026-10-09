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
