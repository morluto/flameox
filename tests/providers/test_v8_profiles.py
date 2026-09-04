from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.providers.capture import CAPTURE_BUILDERS
from flameox.runtime_contracts import CAPTURE_PROVIDER_CONTRACTS, CaptureTarget
from flameox.stateless import AnalysisRuntime


def test_every_capture_contract_has_exactly_one_registered_builder() -> None:
    assert set(CAPTURE_BUILDERS) == set(CAPTURE_PROVIDER_CONTRACTS)


@pytest.mark.process
def test_node_heap_capture_is_analyzable_memory_evidence(tmp_path: Path) -> None:
    executable = tmp_path / "fake-node"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

directory = Path(next(
    value.split("=", 1)[1]
    for value in sys.argv
    if value.startswith("--heap-prof-dir=")
))
name = next(
    value.split("=", 1)[1]
    for value in sys.argv
    if value.startswith("--heap-prof-name=")
)
(directory / name).write_text(json.dumps({
    "head": {
        "callFrame": {
            "functionName": "captured", "url": "app.js",
            "lineNumber": 0, "columnNumber": 0
        },
        "selfSize": 128,
        "id": 1,
        "children": []
    },
    "samples": [{"size": 128, "nodeId": 1}]
}))
"""
    )
    executable.chmod(0o755)

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[str(executable), "app.js"],
                    cwd=str(tmp_path),
                    provider_id="node-heap-profile",
                ),
                "memory.hotspots",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    capture = result["capture"]
    assert isinstance(capture, dict)
    capture_argv = capture["executions"][0]["capture_argv"]
    assert capture_argv[1] == "--heap-prof"
    assert any(value == "--heap-prof-name=profile.heapprofile" for value in capture_argv)
    assert result["provider"]["id"] == "v8-heap-profile"
