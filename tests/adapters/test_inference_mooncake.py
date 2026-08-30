from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.inference import (
    InferenceArtifactExtractor,
    MooncakeRequestRow,
    MooncakeTraceParser,
)
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode, Sensitivity
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


def _write_mooncake_trace(path: Path, lines: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )


def test_mooncake_parse_preserves_request_shape_and_timing(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {
                "timestamp": 0,
                "input_length": 6755,
                "output_length": 500,
                "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            },
            {
                "timestamp": 3052,
                "input_length": 13538,
                "output_length": 71,
                "hash_ids": [0, 237, 238],
            },
            {
                "timestamp": 6105,
                "input_length": 1048,
                "output_length": 26,
                "hash_ids": [0, 1039, 1040],
            },
        ],
    )

    summary, rows = MooncakeTraceParser().parse(trace)

    assert summary.request_count == 3
    assert summary.max_input_length == 13538
    assert summary.max_output_length == 500
    assert summary.timestamp_span_ms == 6105
    assert summary.prefix_hash_count == 20
    assert summary.limitations == ()
    assert [row.line_index for row in rows] == [0, 1, 2]
    assert [row.timestamp_ms for row in rows] == [0, 3052, 6105]
    assert all(row.evidence_level == "observed" for row in rows)
    assert all(isinstance(row, MooncakeRequestRow) for row in rows)
    assert len({row.request_id for row in rows}) == 3


def test_mooncake_iter_rows_streams_one_at_a_time(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 10, "output_length": 5, "hash_ids": [0]},
            {"timestamp": 100, "input_length": 20, "output_length": 10, "hash_ids": [1, 2]},
        ],
    )

    rows = list(MooncakeTraceParser().iter_rows(trace))

    assert len(rows) == 2
    assert rows[0].prefix_hash_count == 1
    assert rows[1].prefix_hash_count == 2


def test_mooncake_accepts_optional_sensitive_payload_without_normalizing_it(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {
                "timestamp": 0,
                "input_length": 10,
                "output_length": 5,
                "messages": [{"role": "user", "content": "secret prompt"}],
                "tools": [{"name": "private_tool"}],
                "payload": {"private": "body"},
            }
        ],
    )

    _summary, rows = MooncakeTraceParser().parse(trace)

    assert rows[0].prefix_hash_count == 0
    normalized = json.dumps(rows[0].model_dump(mode="json"))
    assert "secret prompt" not in normalized
    assert "private_tool" not in normalized
    assert "body" not in normalized


def test_mooncake_parse_skips_blank_lines(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text(
        json.dumps({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]})
        + "\n\n  \n"
        + json.dumps({"timestamp": 5, "input_length": 2, "output_length": 2, "hash_ids": [1]})
        + "\n",
        encoding="utf-8",
    )

    summary, rows = MooncakeTraceParser().parse(trace)

    assert summary.request_count == 2
    assert [row.line_index for row in rows] == [0, 3]


def test_mooncake_parse_truncates_at_max_rows(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 1, "input_length": 1, "output_length": 1, "hash_ids": [1]},
            {"timestamp": 2, "input_length": 1, "output_length": 1, "hash_ids": [2]},
        ],
    )

    summary, _rows = MooncakeTraceParser(max_rows=2).parse(trace)

    assert summary.request_count == 2
    assert summary.limitations == ("Trace truncated at 2 requests.",)


def test_mooncake_parse_at_exact_row_limit_is_not_truncated(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 1, "input_length": 1, "output_length": 1, "hash_ids": [1]},
        ],
    )

    summary, rows = MooncakeTraceParser(max_rows=2).parse(trace)

    assert len(rows) == 2
    assert summary.limitations == ()


def test_mooncake_parse_reports_nonzero_first_timestamp(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 500, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 1000, "input_length": 1, "output_length": 1, "hash_ids": [1]},
        ],
    )

    summary, _rows = MooncakeTraceParser().parse(trace)

    assert summary.limitations == ("The first request timestamp is not zero milliseconds.",)


def test_mooncake_parse_reports_timestamp_regression(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 100, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 50, "input_length": 1, "output_length": 1, "hash_ids": [1]},
        ],
    )

    summary, _rows = MooncakeTraceParser().parse(trace)

    assert any("regressed" in note for note in summary.limitations)


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ({"timestamp": 0, "input_length": 1}, "output_length"),
        ({"timestamp": -1, "input_length": 1, "output_length": 1, "hash_ids": [0]}, "timestamp"),
        (
            {"timestamp": 0, "input_length": 1.5, "output_length": 1, "hash_ids": [0]},
            "input_length",
        ),
        (
            {"timestamp": 0, "input_length": 1, "output_length": True, "hash_ids": [0]},
            "output_length",
        ),
        ({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": "bad"}, "hash_ids"),
        ({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [-1]}, "hash_ids"),
        ({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [True]}, "hash_ids"),
    ],
)
def test_mooncake_rejects_malformed_lines(
    tmp_path: Path, line: dict[str, object], hint: str
) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_non_json_line(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_empty_file(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text("\n\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_oversized_line(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    valid = {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]}
    valid["hash_ids"] = list(range(20_000))
    line = json.dumps(valid)
    trace.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser(max_rows=10).parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_unterminated_oversized_line(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_bytes(b"{" + (b"x" * (MooncakeTraceParser.max_line_bytes + 1)))

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_extractor_publishes_bounded_request_evidence(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    trace = tmp_path / "trace.jsonl"
    _write_mooncake_trace(
        trace,
        [{"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [1]}],
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.INFERENCE_REQUEST_TRACE,
            sensitivity=Sensitivity.SENSITIVE,
        )
    )

    result = InferenceArtifactExtractor(workspace).extract_trace(imported.run.run_id)
    page = EvidenceQueryService(workspace).inference_requests(run_id=imported.run.run_id, limit=1)

    assert result.request_count == 1
    assert result.corpus_commit_id.startswith("sha256:")
    assert page.requests[0].input_tokens == 10
    assert page.requests[0].cache_hit is None


def test_mooncake_extractor_enforces_independent_request_row_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace.initialize(tmp_path)
    trace = tmp_path / "trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 10, "output_length": 2},
            {"timestamp": 1, "input_length": 20, "output_length": 4},
        ],
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.INFERENCE_REQUEST_TRACE,
            sensitivity=Sensitivity.SENSITIVE,
        )
    )
    monkeypatch.setattr(InferenceArtifactExtractor, "max_request_rows", 1)

    result = InferenceArtifactExtractor(workspace).extract_trace(imported.run.run_id)

    assert workspace.config.storage.max_rows_per_generation > 1
    assert result.request_count == 1
    assert result.limitations == ("Trace truncated at 1 requests.",)
