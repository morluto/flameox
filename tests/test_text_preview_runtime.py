from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from mcp import Client

from flameox.mcp.server import FlameoxServer
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import EvidenceSource, PathSource, RequestLimits, RuntimeFailure


@pytest.mark.parametrize("invalid_utf8", [False, True])
def test_text_fragments_preserve_native_bytes_and_resume_after_restart(
    tmp_path: Path, invalid_utf8: bool
) -> None:
    path = tmp_path / "output.log"
    content = ("日🙂\r\n" + "long" * 2000 + "\nlast").encode()
    if invalid_utf8:
        content += b"\xff\xfe"
    path.write_bytes(content)
    sources = [PathSource(path=str(path), format="text")]
    options = {"text_fragment_chars": 128}
    limits = RequestLimits(max_rows=2)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        first = runtime.analyze("artifact.preview", sources, options, limits=limits)
        fragments = [row["text"] for row in first["blocks"][1]["rows"]]
        token = first["continuation"]
        assert token is not None
        preserved = runtime.preserve_evidence(first["analysis_id"])
    finally:
        runtime.close()
    restarted = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        while token is not None:
            page = restarted.analyze(
                "artifact.preview",
                [EvidenceSource(kind="evidence", evidence_id=preserved["evidence_id"])],
                options,
                limits=limits,
                continuation=token,
            )
            fragments.extend(row["text"] for row in page["blocks"][1]["rows"])
            token = page["continuation"]
        assert "".join(fragments) == content.decode(errors="replace")
        assert page["coverage"]["complete"] is True
    finally:
        restarted.close()
    assert path.read_bytes() == content


@pytest.mark.parametrize("change", ["options", "content"])
def test_text_fragment_continuation_binds_options_and_native_identity(
    tmp_path: Path, change: str
) -> None:
    path = tmp_path / "output.log"
    path.write_text("a" * 100)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    sources = [PathSource(path=str(path), format="text")]
    limits = RequestLimits(max_rows=1)
    options = {"text_fragment_chars": 8}
    try:
        first = runtime.analyze("artifact.preview", sources, options, limits=limits)
        if change == "options":
            options = {"text_fragment_chars": 4}
        else:
            path.write_text("b" * 100)
        with pytest.raises(RuntimeFailure):
            runtime.analyze(
                "artifact.preview",
                sources,
                options,
                limits=limits,
                continuation=first["continuation"],
            )
    finally:
        runtime.close()


def test_mcp_exposes_and_executes_optional_text_fragments(tmp_path: Path) -> None:
    path = tmp_path / "output.log"
    path.write_text("x" * 300_000)

    async def exercise() -> None:
        async with Client(
            FlameoxServer(evidence_directory=tmp_path / "store"), raise_exceptions=True
        ) as client:
            result = await client.call_tool(
                "preview_artifact",
                {
                    "sources": [{"kind": "path", "path": str(path), "format": "text"}],
                    "text_fragment_chars": 128,
                },
            )
            assert result.is_error is False
            assert result.structured_content["coverage"]["complete"] is False
            assert result.structured_content["continuation"] is None
            assert result.structured_content["next_page"]["tool"] == "preview_artifact"

    anyio.run(exercise)
