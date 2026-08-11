from __future__ import annotations

import pytest
from pydantic import ValidationError

from flameox.analysis import (
    InferenceProtocolIdentity,
    OracleResult,
    ScheduleIdentity,
    ServerConfigIdentity,
    TraceIdentity,
    compare_inference_protocols,
)
from flameox.domain import ComparisonValidity, OracleStatus

_DIGEST = "sha256:" + "a" * 64


def _protocol(**overrides: object) -> InferenceProtocolIdentity:
    base: dict[str, object] = {
        "provider": "mooncake-trace-replayer",
        "provider_version": "1.0",
        "provider_executable_digest": "sha256:" + "b" * 64,
        "trace": {
            "format": "mooncake",
            "producer": "kvcache-ai/Mooncake",
            "producer_version": "1",
            "artifact_digest": _DIGEST,
            "window_start_ms": 0,
            "window_end_ms": 60_000,
            "request_count": 313,
        },
        "schedule": {
            "preserve_timing": True,
            "time_scale": 1.0,
            "max_concurrency": 10,
            "request_rate": 5.0,
            "burstiness": 1.0,
            "duration_seconds": 60.0,
        },
        "model": {
            "model_id": "NousResearch/Llama-3.2-1B",
            "model_revision": "main",
            "tokenizer_id": "NousResearch/Llama-3.2-1B",
            "tokenizer_revision": "main",
            "trust_remote_code": False,
            "dtype": "auto",
        },
        "server": {
            "backend": "vllm",
            "endpoint": "/v1/completions",
            "kv_transfer_config": {"kv_connector": "MooncakeConnector"},
            "cache_backend": "mooncake",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "managed_server_command_digest": "sha256:" + "a" * 64,
            "server_executable_digest": "sha256:" + "e" * 64,
            "server_version": "0.26.0",
        },
        "hardware": {
            "accelerator_kind": "cuda",
            "accelerator_count": 1,
            "accelerator_model": "A100",
            "driver_version": "535.0",
            "runtime_version": "12.1",
            "topology_digest": "sha256:" + "c" * 64,
        },
        "profiler": {
            "profiler": "none",
            "attached": False,
        },
        "oracle": {
            "kind": "contract_check",
            "estimand": "request_throughput",
            "tolerance_absolute": 1.0,
            "tolerance_relative": 0.05,
            "command_digest": "sha256:" + "d" * 64,
        },
        "oracle_result": {"status": "pass", "reason": "within_tolerance"},
    }
    base.update(overrides)
    return InferenceProtocolIdentity.model_validate(base)


def _apply(
    protocol: InferenceProtocolIdentity, path: str, value: object
) -> InferenceProtocolIdentity:
    data = protocol.model_dump(mode="json")
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return InferenceProtocolIdentity.model_validate(data)


# ---------------------------------------------------------------------------
# Valid comparisons
# ---------------------------------------------------------------------------


def test_identical_protocols_are_valid() -> None:
    baseline = _protocol()
    candidate = _protocol()

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.VALID
    assert result.mismatches == ()
    assert result.is_comparable is True


def test_protocols_with_different_oracle_status_are_invalid() -> None:
    baseline = _protocol()
    candidate = _protocol(
        oracle_result=OracleResult(status=OracleStatus.FAIL, reason="outside_tolerance"),
    )

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.INVALID
    assert [mismatch.field for mismatch in result.mismatches] == ["oracle_result.status"]


def test_missing_semantic_oracle_results_are_exploratory() -> None:
    baseline = _protocol(oracle_result=None)
    candidate = _protocol(oracle_result=None)

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert any(reason.field == "oracle_result.status" for reason in result.exploratory_reasons)


# ---------------------------------------------------------------------------
# Invalid comparisons — exact mismatch fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "baseline_value", "candidate_value"),
    [
        ("provider", "mooncake-trace-replayer", "aiperf"),
        ("provider_version", "1.0", "2.0"),
        ("trace.format", "mooncake", "aiperf"),
        ("trace.producer", "kvcache-ai/Mooncake", "other"),
        ("trace.artifact_digest", _DIGEST, "sha256:" + "b" * 64),
        ("trace.window_start_ms", 0, 1000),
        ("trace.window_end_ms", 60_000, 30_000),
        ("trace.request_count", 313, 200),
        ("schedule.preserve_timing", True, False),
        ("schedule.time_scale", 1.0, 2.0),
        ("schedule.max_concurrency", None, 4),
        ("schedule.duration_seconds", 60.0, 30.0),
        ("model.model_id", "NousResearch/Llama-3.2-1B", "Qwen/Qwen2.5-7B-Instruct"),
        ("model.model_revision", "main", "v1.0"),
        ("model.tokenizer_id", "NousResearch/Llama-3.2-1B", "other/tokenizer"),
        ("model.dtype", "auto", "float16"),
        ("server.backend", "vllm", "openai-chat"),
        ("server.endpoint", "/v1/completions", "/v1/chat/completions"),
        ("server.cache_backend", "mooncake", "vllm_paged"),
        ("server.tensor_parallel_size", 1, 2),
        ("server.gpu_memory_utilization", 0.9, 0.8),
        ("server.max_model_len", 4096, 8192),
        ("hardware.accelerator_kind", "cuda", "cpu"),
        ("hardware.accelerator_count", 1, 2),
        ("hardware.accelerator_model", "A100", "H100"),
        ("hardware.driver_version", "535.0", "540.0"),
        ("profiler.profiler", "none", "torch_profiler"),
        ("oracle.kind", "execution_check", "contract_check"),
        ("oracle.estimand", "request_throughput", "ttft"),
        ("oracle.tolerance_absolute", 1.0, 2.0),
        ("oracle.tolerance_relative", 0.05, 0.10),
    ],
)
def test_declared_facet_mismatch_is_invalid_with_exact_field(
    path: str, baseline_value: object, candidate_value: object
) -> None:
    baseline = _apply(_protocol(), path, baseline_value)
    candidate = _apply(_protocol(), path, candidate_value)

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.INVALID
    assert result.is_comparable is False
    fields = [m.field for m in result.mismatches]
    assert path in fields
    mismatch = next(m for m in result.mismatches if m.field == path)
    assert mismatch.baseline != mismatch.candidate


def test_profiler_attachment_mismatch_is_invalid_between_valid_states() -> None:
    baseline = _protocol()
    candidate = _apply(_protocol(), "profiler.profiler", "torch_profiler")
    candidate = _apply(candidate, "profiler.attached", True)

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.INVALID
    assert any(mismatch.field == "profiler.attached" for mismatch in result.mismatches)


def test_kv_transfer_config_mismatch_is_invalid() -> None:
    baseline = _protocol()
    candidate = _apply(
        baseline, "server.kv_transfer_config", {"kv_connector": "MooncakeStoreConnector"}
    )

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.INVALID
    assert any(m.field == "server.kv_transfer_config" for m in result.mismatches)


def test_multiple_mismatches_all_reported() -> None:
    baseline = _protocol()
    candidate = _apply(baseline, "model.model_id", "Qwen/Qwen2.5-7B-Instruct")
    candidate = _apply(candidate, "hardware.accelerator_count", 2)
    candidate = _apply(candidate, "oracle.kind", "execution_check")

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.INVALID
    fields = {m.field for m in result.mismatches}
    assert {"model.model_id", "hardware.accelerator_count", "oracle.kind"} <= fields


# ---------------------------------------------------------------------------
# Exploratory comparisons — undeclared facets
# ---------------------------------------------------------------------------


def test_missing_required_facet_is_exploratory() -> None:
    baseline = _apply(_protocol(), "provider_version", None)
    candidate = _apply(_protocol(), "provider_version", None)

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert any(r.field == "provider_version" for r in result.exploratory_reasons)
    assert result.mismatches == ()


@pytest.mark.parametrize(
    ("path", "value", "expected_field"),
    [
        ("provider_executable_digest", None, "provider_executable_digest"),
        ("trace.artifact_digest", None, "trace.artifact_digest"),
        ("model.model_revision", None, "model.model_revision"),
        ("model.tokenizer_id", None, "model.tokenizer_id"),
        ("model.tokenizer_revision", None, "model.tokenizer_revision"),
        (
            "server.managed_server_command_digest",
            None,
            "server.managed_server_command_digest",
        ),
        ("hardware.accelerator_kind", "unknown", "hardware.accelerator_kind"),
        ("hardware.topology_digest", None, "hardware.topology_digest"),
        ("oracle.command_digest", None, "oracle.command_digest"),
    ],
)
def test_confirmatory_identity_requires_contextual_facets(
    path: str,
    value: object,
    expected_field: str,
) -> None:
    protocol = _apply(_protocol(), path, value)

    result = compare_inference_protocols(protocol, protocol)

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert any(reason.field == expected_field for reason in result.exploratory_reasons)


def test_optional_inapplicable_facets_missing_on_both_sides_remain_valid() -> None:
    baseline = _apply(_protocol(), "trace.window_start_ms", None)
    baseline = _apply(baseline, "trace.window_end_ms", None)
    baseline = _apply(baseline, "schedule.request_rate", None)
    baseline = _apply(baseline, "schedule.burstiness", None)
    candidate = baseline

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.VALID
    assert result.exploratory_reasons == ()


def test_one_side_declared_other_missing_is_mismatch_not_exploratory() -> None:
    baseline = _protocol()
    candidate = _apply(_protocol(), "hardware.driver_version", None)

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.INVALID
    assert any(m.field == "hardware.driver_version" for m in result.mismatches)
    assert not any(r.field == "hardware.driver_version" for r in result.exploratory_reasons)


def test_accelerator_driver_is_required_for_confirmatory_comparison() -> None:
    baseline = _apply(_protocol(), "hardware.driver_version", None)
    candidate = _apply(_protocol(), "hardware.driver_version", None)

    result = compare_inference_protocols(baseline, candidate)

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert any(r.field == "hardware.driver_version" for r in result.exploratory_reasons)


def test_unprofiled_runs_do_not_require_a_profiler_version() -> None:
    result = compare_inference_protocols(_protocol(), _protocol())

    assert result.validity is ComparisonValidity.VALID
    assert not any(r.field == "profiler.profiler_version" for r in result.exploratory_reasons)


def test_attached_profiler_requires_its_version() -> None:
    baseline = _apply(_protocol(), "profiler.profiler", "torch_profiler")
    baseline = _apply(baseline, "profiler.attached", True)

    result = compare_inference_protocols(baseline, baseline)

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert any(r.field == "profiler.profiler_version" for r in result.exploratory_reasons)


def test_attached_profiler_cannot_use_the_none_variant() -> None:
    data = _protocol().model_dump(mode="json")
    data["profiler"] = {"profiler": "none", "attached": True}

    with pytest.raises(ValidationError):
        InferenceProtocolIdentity.model_validate(data)


def test_nonpassing_semantic_oracles_remain_exploratory() -> None:
    protocol = _protocol(
        oracle_result=OracleResult(status=OracleStatus.FAIL, reason="contract_failed"),
    )

    result = compare_inference_protocols(protocol, protocol)

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert any(r.field == "oracle_result.status" for r in result.exploratory_reasons)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_protocol_identity_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError):
        TraceIdentity(format="mooncake", producer="x", artifact_digest="not-a-digest")


def test_schedule_rejects_non_positive_time_scale() -> None:
    with pytest.raises(ValueError):
        ScheduleIdentity(preserve_timing=True, time_scale=0.0)


def test_server_rejects_out_of_range_gpu_utilization() -> None:
    with pytest.raises(ValueError):
        ServerConfigIdentity(backend="vllm", gpu_memory_utilization=1.5)


def test_oracle_result_rejects_empty_reason() -> None:
    with pytest.raises(ValueError):
        OracleResult(status=OracleStatus.PASS, reason="")


def test_kv_transfer_config_with_delimiter_chars_does_not_collide() -> None:
    """Distinct KV-transfer configs with commas/equals in keys/values must not collide.

    Regression for #288: the old ``_normalize()`` used unescaped
    ``",".join(f"{k}={v}")`` which could make distinct dicts produce the
    same normalized string. The fix uses canonical JSON serialization.
    """
    from flameox.analysis.inference_protocol import _normalize

    config_a = {"a=b,c": "d"}
    config_b = {"a": "b=c,d"}

    norm_a = _normalize(config_a)
    norm_b = _normalize(config_b)
    assert norm_a != norm_b, (
        f"Distinct configs must not collide: {config_a!r} vs {config_b!r} "
        f"both normalized to {norm_a!r}"
    )
