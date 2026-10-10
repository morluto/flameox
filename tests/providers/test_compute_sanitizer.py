from __future__ import annotations

from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource


@pytest.mark.process
def test_compute_sanitizer_xml_projects_classification_and_addresses(tmp_path: Path) -> None:
    artifact = tmp_path / "sanitizer.xml"
    artifact.write_text(
        """<ComputeSanitizerOutput>
<record>
  <kind>Precise</kind><level>Error</level>
  <what><text>Invalid write: Access is out of bounds</text><space>global</space>
    <size>4</size><accessAddress>0x10000002a</accessAddress></what>
  <where><func>write_values</func><path>kernels/vector_add.cu</path><line>8</line></where>
  <who><threadIdx><x>3</x><y>1</y><z>0</z></threadIdx>
    <blockIdx><x>2</x><y>0</y><z>0</z></blockIdx></who>
</record>
</ComputeSanitizerOutput>"""
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "sanitizer.failures",
            [PathSource(path=str(artifact), format="compute-sanitizer")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "compute-sanitizer"
    assert result["blocks"][0]["values"] == {"memory_access": 1}
    assert result["blocks"][1]["rows"] == [
        {
            "kind": "Precise",
            "level": "Error",
            "classification": "memory_access",
            "message": "Invalid write: Access is out of bounds",
            "memory_space": "global",
            "access_address": 4_294_967_338,
            "access_size": 4,
            "error_address": None,
            "direction": None,
            "error": None,
            "function": "write_values",
            "path": "kernels/vector_add.cu",
            "line": 8,
            "pc": None,
            "thread": {"x": 3, "y": 1, "z": 0},
            "block": {"x": 2, "y": 0, "z": 0},
            "frames": [],
        }
    ]


@pytest.mark.process
@pytest.mark.parametrize("frame_count", [1001, 1002])
def test_compute_sanitizer_nested_stack_omission_marks_coverage_incomplete(
    tmp_path: Path, frame_count: int
) -> None:
    artifact = tmp_path / "sanitizer.xml"
    frames = "".join(
        f"<frame><func>frame_{index}</func><line>1</line></frame>" for index in range(frame_count)
    )
    artifact.write_text(
        "<ComputeSanitizerOutput><record><kind>precise</kind><hostStack>"
        + frames
        + "</hostStack></record></ComputeSanitizerOutput>"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "sanitizer.failures", [PathSource(path=str(artifact), format="compute-sanitizer")], {}
        )
        assert len(result["blocks"][1]["rows"]) == 1
        assert len(result["blocks"][1]["rows"][0]["frames"]) == 1001
        assert result["coverage"]["rows_observed"] == 1
        assert result["coverage"]["complete"] is (frame_count == 1001)
    finally:
        runtime.close()
