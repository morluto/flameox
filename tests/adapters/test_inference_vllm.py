from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.inference import (
    InferenceArtifactExtractor,
    VllmResultParser,
)
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


def _vllm_metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "completed": 313,
        "total_input": 1_000_000,
        "total_output": 50_000,
        "request_throughput": 5.18,
        "request_goodput": 5.18,
        "output_throughput": 827.0,
        "total_token_throughput": 1735.0,
        "mean_ttft_ms": 2187.98,
        "median_ttft_ms": 2100.0,
        "std_ttft_ms": 120.0,
        "percentiles_ttft_ms": [[50.0, 2100.0], [90.0, 2500.0], [99.0, 3000.0]],
        "mean_tpot_ms": 26.59,
        "median_tpot_ms": 25.0,
        "std_tpot_ms": 3.0,
        "percentiles_tpot_ms": [[50.0, 25.0], [90.0, 30.0], [99.0, 35.0]],
        "mean_itl_ms": 26.0,
        "median_itl_ms": 25.0,
        "std_itl_ms": 3.0,
        "percentiles_itl_ms": [[50.0, 25.0], [90.0, 30.0], [99.0, 35.0]],
        "mean_e2el_ms": 5000.0,
        "median_e2el_ms": 4900.0,
        "std_e2el_ms": 200.0,
        "percentiles_e2el_ms": [[50.0, 4900.0], [90.0, 5500.0], [99.0, 6000.0]],
    }
    base.update(overrides)
    return base


def _vllm_result(**overrides: object) -> dict[str, object]:
    metrics = _vllm_metrics(**(overrides.pop("metrics", {})))  # type: ignore[arg-type]
    base: dict[str, object] = {
        "metrics": metrics,
        "successful_requests": 313,
        "failed_requests": 0,
        "total_requests": 313,
        "actual_duration": 60.48,
        "original_time_span": 60.0,
        "time_scale": 1.0,
    }
    base.update(overrides)
    return base


def test_vllm_parse_normalizes_aggregate_metrics(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps(_vllm_result()), encoding="utf-8")

    document, rows = VllmResultParser().parse(result)

    assert document.successful_requests == 313
    assert document.failed_requests == 0
    names = {row.name for row in rows}
    assert "vllm.request_throughput" in names
    assert "vllm.request_goodput" in names
    assert "vllm.output_throughput" in names
    assert "vllm.total_token_throughput" in names
    assert "vllm.total_input_tokens" in names
    assert "vllm.total_output_tokens" in names
    assert "vllm.completed_requests" in names
    assert "vllm.failed_requests" in names
    assert "vllm.total_requests" in names
    assert "vllm.duration_seconds" in names
    assert "vllm.time_to_first_token.mean_ms" in names
    assert "vllm.time_to_first_token.median_ms" in names
    assert "vllm.time_to_first_token.std_ms" in names
    assert "vllm.time_to_first_token.p50_ms" in names
    assert "vllm.time_to_first_token.p99_ms" in names
    assert "vllm.time_per_output_token.mean_ms" in names
    assert "vllm.inter_token_latency.mean_ms" in names
    assert "vllm.end_to_end_latency.mean_ms" in names
    assert all(row.evidence_level == "derived" for row in rows)
    assert all(
        row.unit in {"requests/sec", "tokens/sec", "count", "requests", "ms", "s"} for row in rows
    )
    throughput = next(row for row in rows if row.name == "vllm.request_throughput")
    assert throughput.value_float == pytest.approx(5.18)
    assert throughput.aggregation == "aggregate"
    assert throughput.dimensions["producer"] == "vllm.benchmark_serving"
    assert throughput.dimensions["time_scale"] == "1.0"
    by_name = {row.name: row.value_float for row in rows}
    assert by_name["vllm.request_goodput"] == pytest.approx(5.18)
    assert by_name["vllm.completed_requests"] == 313
    assert by_name["vllm.failed_requests"] == 0
    assert by_name["vllm.total_requests"] == 313
    assert by_name["vllm.duration_seconds"] == pytest.approx(60.48)


def test_vllm_parse_omits_request_goodput_when_provider_does_not_emit_it() -> None:
    payload = _vllm_result()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    del metrics["request_goodput"]

    document, rows = VllmResultParser().parse_payload(payload)

    assert document.metrics.request_goodput is None
    assert all(row.name != "vllm.request_goodput" for row in rows)


def test_vllm_parse_drops_raw_payload_fields(tmp_path: Path) -> None:
    payload = _vllm_result()
    payload["raw_prompts"] = ["secret prompt text"]
    payload["error_log"] = "connection refused to http://10.0.0.1:8000"
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps(payload), encoding="utf-8")

    document, rows = VllmResultParser().parse(result)

    serialized = json.dumps([row.model_dump(mode="json") for row in rows])
    assert "secret prompt text" not in serialized
    assert "10.0.0.1" not in serialized
    assert document.successful_requests == 313


def test_vllm_parse_rejects_result_above_document_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_vllm_result()))
    monkeypatch.setattr(VllmResultParser, "max_document_bytes", 8)

    with pytest.raises(DomainError) as caught:
        VllmResultParser().parse(result)

    assert caught.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_missing_metrics(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps({"successful_requests": 1}), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_mismatched_totals(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(_vllm_result(successful_requests=200, failed_requests=0, total_requests=313)),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_completed_mismatch(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(
            _vllm_result(
                metrics={"completed": 100},
                successful_requests=313,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_non_json(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text("{not json", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_negative_latency(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(_vllm_result(metrics={"mean_ttft_ms": -1.0})),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


@pytest.mark.parametrize(
    "percentile_pair",
    [(-1.0, 1.0), (101.0, 1.0), (50.0, -1.0), (float("nan"), 1.0), (50.0, float("inf"))],
)
def test_vllm_parse_rejects_invalid_percentile_pairs(
    tmp_path: Path, percentile_pair: tuple[float, float]
) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(
            _vllm_result(metrics={"percentiles_ttft_ms": [percentile_pair]}),
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


@pytest.mark.parametrize("payload", [[], "not an object", 1])
def test_vllm_parse_wraps_non_object_json_as_domain_error(tmp_path: Path, payload: object) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_infinite_latency(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    payload = _vllm_result()
    payload["metrics"]["mean_ttft_ms"] = float("inf")  # type: ignore[index]
    result.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_payload_accepts_dict_directly() -> None:
    document, rows = VllmResultParser().parse_payload(_vllm_result())

    assert document.successful_requests == 313
    assert len(rows) > 0


def test_vllm_parse_accepts_native_save_result_shape() -> None:
    metrics = _vllm_metrics()
    metrics["num_prompts"] = 320
    metrics["duration"] = 60.48
    metrics["total_input_tokens"] = metrics.pop("total_input")
    metrics["total_output_tokens"] = metrics.pop("total_output")

    document, rows = VllmResultParser().parse_payload(metrics)

    assert document.successful_requests == 313
    assert document.failed_requests == 7
    assert document.total_requests == 320
    assert any(row.name == "vllm.request_throughput" for row in rows)


def test_vllm_parse_rejects_boolean_native_request_counts() -> None:
    with pytest.raises(DomainError) as error:
        VllmResultParser().parse_payload({"completed": True, "num_prompts": True})

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_native_result_without_duration() -> None:
    metrics = _vllm_metrics()
    metrics["num_prompts"] = 313

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse_payload(metrics)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_extractor_publishes_aggregate_measurements(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_vllm_result()))
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=result_path, kind=ArtifactKind.INFERENCE_RESULT)
    )

    result = InferenceArtifactExtractor(workspace).extract_vllm_result(imported.run.run_id)

    assert result.measurement_count > 0
    assert result.corpus_commit_id.startswith("sha256:")


def test_extraction_idempotency_is_scoped_to_target_evidence_run(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_vllm_result()))
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=result_path, kind=ArtifactKind.INFERENCE_RESULT)
    )
    extractor = InferenceArtifactExtractor(workspace)
    first_target = "canonical-run-1"
    second_target = "canonical-run-2"

    first = extractor.extract_vllm_result(imported.run.run_id, evidence_run_id=first_target)
    second = extractor.extract_vllm_result(imported.run.run_id, evidence_run_id=second_target)

    assert first.corpus_commit_id != second.corpus_commit_id
    first_rows = EvidenceQueryService(workspace).measurements(run_id=first_target, limit=100)
    second_rows = EvidenceQueryService(workspace).measurements(run_id=second_target, limit=100)
    assert first_rows.measurements
    assert second_rows.measurements
    assert {row.run_id for row in first_rows.measurements} == {first_target}
    assert {row.run_id for row in second_rows.measurements} == {second_target}
