from __future__ import annotations

import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.execution import ProcessExecutionError
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RequestLimits, RuntimeFailure


@pytest.mark.process
@pytest.mark.parametrize("exporter", ["xcrun", "nsys"])
@pytest.mark.parametrize("failure_mode", ["nonzero", "missing", "timeout"])
def test_native_export_failure_discards_partial_files_and_retry_reuses_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exporter: str, failure_mode: str
) -> None:
    """Real fault-injection executables exercise the broker, not native vendor exporters."""
    mode = tmp_path / "mode"
    mode.write_text(failure_mode)
    calls = tmp_path / "calls"
    fixture = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"name": ["work"], "duration": [42]}), fixture)
    executable = tmp_path / exporter
    executable.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys, time\n"
        "from pathlib import Path\n"
        f"mode = Path({str(mode)!r}).read_text()\n"
        f"with Path({str(calls)!r}).open('a') as stream: stream.write('run\\n')\n"
        "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "(output.parent / 'partial-sidecar').write_text('partial')\n"
        "if mode != 'missing':\n"
        f"    if {exporter!r} == 'nsys':\n"
        "        output = output.with_suffix('.parquetdir')\n"
        "        output.mkdir()\n"
        f"        shutil.copyfile({str(fixture)!r}, output / 'CUDA_GPU_KERN_SUM.parquet')\n"
        "    else:\n"
        "        output.write_text('<trace><run name=\"sample\"/></trace>')\n"
        "if mode == 'timeout': time.sleep(30)\n"
        "sys.exit(7 if mode == 'nonzero' else 0)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    native = tmp_path / "native"
    native.write_bytes(b"native artifact")
    source = PathSource(path=str(native), format="xctrace" if exporter == "xcrun" else "nsys-rep")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "summarize_trace", [source], {}, limits=RequestLimits(timeout_seconds=1)
            )
        assert failure.value.code == (
            "DECODE_FAILURE" if failure_mode == "nonzero" else "EXECUTION_FAILURE"
        )
        if failure_mode == "timeout":
            cause = failure.value.__cause__
            assert isinstance(cause, ProcessExecutionError)
            assert cause.process.cancellation_cause is not None
            assert cause.process.cancellation_cause.value == "timeout"
            assert cause.process.cleanup_complete
        assert not list((runtime.scratch / "conversions").iterdir())

        mode.write_text("success")
        first = runtime.analyze("summarize_trace", [source], {}, limits=RequestLimits(max_rows=1))
        second = runtime.analyze("summarize_trace", [source], {}, limits=RequestLimits(max_rows=2))
        assert first["coverage"]["rows_returned"] >= 1
        assert second["coverage"]["complete"]
        assert calls.read_text().splitlines() == ["run", "run"]
    finally:
        runtime.close()
