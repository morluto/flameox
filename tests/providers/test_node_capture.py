from __future__ import annotations

import json
import shutil
from pathlib import Path

import anyio
import pytest

from flameox.providers.capture import resolve_capture_artifacts
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CaptureTarget, RuntimeFailure, WorkloadBudget


@pytest.mark.parametrize("limit", ["bytes", "files"])
def test_node_profile_collection_bounds_cover_all_files(tmp_path: Path, limit: str) -> None:
    for index in range(2):
        (tmp_path / f"CPU.{index}.cpuprofile").write_bytes(b"12345")
    with pytest.raises(RuntimeFailure) as failure:
        resolve_capture_artifacts(
            tmp_path,
            format_name="cpuprofile",
            role="cpu-profile",
            provider_id="node-cpu-profile",
            max_bytes=9 if limit == "bytes" else 10,
            max_files=1 if limit == "files" else 2,
        )
    assert failure.value.code == "LIMIT_EXCEEDED"


@pytest.mark.process
def test_node_timeout_preserves_completed_child_profile(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for native profile capture")
    marker = tmp_path / "child-completed"
    workload = tmp_path / "workload.js"
    workload.write_text(
        "if (process.argv[2] === 'child') {\n"
        "  function childOnlyHotspot() {\n"
        "    const end = Date.now() + 50;\n"
        "    while (Date.now() < end) Math.sqrt(Date.now());\n"
        "  }\n"
        "  childOnlyHotspot();\n"
        "} else {\n"
        "  require('node:child_process').fork(__filename, ['child']).on('exit', code => {\n"
        "    if (code !== 0) throw new Error('child failed');\n"
        f"    require('node:fs').writeFileSync({json.dumps(str(marker))}, 'done');\n"
        "    setInterval(() => {}, 1000);\n"
        "  });\n"
        "}\n"
    )

    async def exercise() -> str:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[node, str(workload)],
                    cwd=str(tmp_path),
                    provider_id="node-cpu-profile",
                    budget=WorkloadBudget(timeout_seconds=2),
                ),
                "rank_cpu_hotspots",
                preserve=True,
            )
            assert marker.read_text() == "done"
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "failed"
            assert execution["failure_code"] == "EXECUTION_TIMEOUT"
            assert result["analysis_failure"] is None
            assert any(row["function"] == "childOnlyHotspot" for row in result["blocks"][1]["rows"])
            return str(result["preserved"]["evidence_id"])
        finally:
            runtime.close()

    evidence_id = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        inspected = reopened.read_evidence(evidence_id)
        artifacts = inspected["body"]["artifacts"]
        assert artifacts
        assert all("/cpu-profile/CPU." in artifact["role"] for artifact in artifacts)
        assert all(artifact["format"] == "cpuprofile" for artifact in artifacts)
    finally:
        reopened.close()
