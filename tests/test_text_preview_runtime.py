from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from mcp import Client

from flameox.mcp import create_server
from flameox.runtime_contracts import EvidenceSource, PathSource, RequestLimits, RuntimeFailure
from flameox.stateless import AnalysisRuntime


def test_text_fragments_preserve_native_bytes_and_resume_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "output.log"
    content = ("日🙂\r\n" + "long" * 2000 + "\nlast").encode()
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
        assert "".join(fragments).encode() == content
        assert page["coverage"]["complete"] is True
    finally:
        restarted.close()
    assert path.read_bytes() == content


def test_oversized_line_can_be_recovered_without_changing_default_offsets(tmp_path: Path) -> None:
    path = tmp_path / "long.log"
    path.write_bytes(b"x" * 300_000 + b"\nshort\n")
    runtime = AnalysisRuntime()
    sources = [PathSource(path=str(path), format="text")]
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("artifact.preview", sources, {})
        assert failure.value.code == "LIMIT_EXCEEDED"
        assert "text_fragment_chars" in failure.value.details["recovery"]
        ordinary = runtime.analyze("artifact.preview", sources, {"offset": 1})
        assert ordinary["blocks"][1]["rows"][0]["text"] == "short"
        recovered = runtime.analyze("artifact.preview", sources, {"text_fragment_chars": 1024})
        assert recovered["continuation"] is not None
        assert recovered["coverage"]["complete"] is False
        assert all(len(row["text"]) <= 1024 for row in recovered["blocks"][1]["rows"])
    finally:
        runtime.close()


@pytest.mark.parametrize("change", ["options", "content"])
def test_text_fragment_continuation_binds_options_and_native_identity(
    tmp_path: Path, change: str
) -> None:
    path = tmp_path / "output.log"
    path.write_text("a" * 100)
    runtime = AnalysisRuntime()
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


def test_fragment_option_rejects_nontext_sources(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("[]")
    runtime = AnalysisRuntime()
    try:
        with pytest.raises(RuntimeFailure, match="text file sources"):
            runtime.analyze(
                "artifact.preview", [PathSource(path=str(path))], {"text_fragment_chars": 128}
            )
    finally:
        runtime.close()


def test_minimal_fragment_reports_result_envelope_limit(tmp_path: Path) -> None:
    path = tmp_path / "output.log"
    path.write_text("abc")
    runtime = AnalysisRuntime()
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(path), format="text")],
                {"text_fragment_chars": 1},
                limits=RequestLimits(max_result_bytes=1024),
            )
        assert "larger max_result_bytes" in failure.value.details["recovery"]
        assert "text_fragment_chars=128" not in failure.value.details["recovery"]
    finally:
        runtime.close()


def test_mcp_exposes_and_executes_text_fragment_recovery(tmp_path: Path) -> None:
    path = tmp_path / "output.log"
    path.write_text("x" * 300_000)

    async def exercise() -> None:
        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "preview_artifact",
                {
                    "sources": [{"kind": "path", "path": str(path), "format": "text"}],
                    "options": {"text_fragment_chars": 128},
                },
            )
            assert result.is_error is False
            assert result.structured_content["coverage"]["complete"] is False
            assert result.structured_content["continuation"] is not None

    anyio.run(exercise)
