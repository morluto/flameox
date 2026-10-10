from __future__ import annotations

from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RuntimeFailure


@pytest.mark.parametrize(
    ("capability", "format_name"),
    [
        ("cpu.hotspots", "py-spy"),
        ("benchmark.summary", "nvbench"),
        ("inference.summary", "vllm-benchmark"),
        ("inference.summary", "sglang-benchmark"),
        ("inference.summary", "mooncake-trace"),
        ("kernel.validation", "kernel-validation"),
        ("triton.autotune", "triton"),
        ("triton.autotune", "triton-cache"),
    ],
)
@pytest.mark.parametrize(
    "native",
    [b"\xff", b'{"x":' + b"9" * 5_000 + b"}", b"[" * 10_000 + b"0" + b"]" * 10_000],
    ids=["invalid-unicode", "integer-conversion-limit", "recursive-json"],
)
def test_native_json_decode_failures_remain_typed(
    tmp_path: Path, capability: str, format_name: str, native: bytes
) -> None:
    artifact = tmp_path / "native.json"
    artifact.write_bytes(native)
    source_path = tmp_path if format_name == "nvbench" else artifact
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(capability, [PathSource(path=str(source_path), format=format_name)], {})
        assert failure.value.code == "DECODE_FAILURE"
        assert artifact.read_bytes() == native
    finally:
        runtime.close()
