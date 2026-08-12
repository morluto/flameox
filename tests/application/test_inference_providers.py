from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.application.inference_providers import (
    AIPerfProfileRequest,
    InferenceEndpointType,
    InferenceTool,
    SglangBenchServingRequest,
    VllmBenchServeRequest,
    discover_inference_tool,
    discover_sglang,
    parse_inference_tool_discovery,
    probe_existing_vllm_server,
)


def test_aiperf_replay_argv_preserves_fixed_original_schedule(tmp_path: Path) -> None:
    request = AIPerfProfileRequest(
        executable=Path("/tools/aiperf"),
        base_url="http://127.0.0.1:8000",
        model="model",
        tokenizer="tokenizer",
        trace_path=tmp_path / "trace.jsonl",
        output_dir=tmp_path / "output",
        request_count=None,
    )

    assert request.argv() == (
        "/tools/aiperf",
        "profile",
        "--url",
        "http://127.0.0.1:8000",
        "--model",
        "model",
        "--endpoint-type",
        "chat",
        "--output-artifact-dir",
        str(tmp_path / "output"),
        "--export-level",
        "records",
        "--tokenizer",
        "tokenizer",
        "--streaming",
        "--input-file",
        str(tmp_path / "trace.jsonl"),
        "--custom-dataset-type",
        "mooncake_trace",
        "--fixed-schedule",
        "--random-seed",
        "0",
    )


def test_provider_requests_refuse_non_loopback_servers(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        VllmBenchServeRequest(
            executable=Path("/tools/vllm"),
            base_url="http://example.test:8000",
            model="model",
            result_path=tmp_path / "result.json",
            num_prompts=1,
        )


def test_tool_discovery_requires_an_executable_when_available() -> None:
    with pytest.raises(ValidationError):
        parse_inference_tool_discovery(
            {
                "tool": "aiperf",
                "available": True,
                "executable": None,
            }
        )


def test_aiperf_synthetic_argv_bounds_request_count(tmp_path: Path) -> None:
    request = AIPerfProfileRequest(
        executable=Path("/tools/aiperf"),
        base_url="http://127.0.0.1:8000",
        model="model",
        trace_path=None,
        fixed_schedule=False,
        output_dir=tmp_path / "output",
        request_count=12,
    )

    assert request.argv()[request.argv().index("--request-count") + 1] == "12"


def test_vllm_bench_argv_requires_structured_result_file(tmp_path: Path) -> None:
    request = VllmBenchServeRequest(
        executable=Path("/tools/vllm"),
        base_url="http://localhost:8000",
        model="model",
        result_path=tmp_path / "results.json",
        num_prompts=5,
    )

    assert "--save-result" in request.argv()
    assert "--save-detailed" not in request.argv()
    assert "results.json" in request.argv()
    assert request.argv()[request.argv().index("--backend") + 1] == "openai-chat"
    assert request.argv()[request.argv().index("--endpoint") + 1] == "/v1/chat/completions"
    assert "--endpoint-type" not in request.argv()
    assert "--streaming" not in request.argv()
    assert request.argv()[request.argv().index("--num-warmups") + 1] == "0"


def test_vllm_bench_refuses_unsupported_non_streaming_mode(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        VllmBenchServeRequest.model_validate(
            {
                "executable": Path("/tools/vllm"),
                "base_url": "http://localhost:8000",
                "model": "model",
                "result_path": tmp_path / "results.json",
                "num_prompts": 5,
                "streaming": False,
            }
        )


def test_sglang_bench_refuses_unsupported_non_streaming_mode(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        SglangBenchServingRequest.model_validate(
            {
                "benchmark_python": Path("/opt/sglang/bin/python"),
                "base_url": "http://127.0.0.1:8000",
                "model": "model",
                "result_path": tmp_path / "result.jsonl",
                "num_prompts": 1,
                "random_input_len": 4,
                "random_output_len": 2,
                "streaming": False,
            }
        )


def test_existing_server_probe_uses_only_read_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Response:
        status = 200

        def __init__(self) -> None:
            self._read = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, *_args: object) -> bytes:
            if self._read:
                return b""
            self._read = True
            return json.dumps({"data": [{"id": "model"}]}).encode()

    def fake_open_probe(request: object, timeout: float) -> Response:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr("flameox.application.inference_providers._open_probe", fake_open_probe)

    assert probe_existing_vllm_server("http://127.0.0.1:8000").model_ids == ("model",)
    assert calls == ["http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1/models"]


def test_aiperf_discovery_rejects_unsupported_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "aiperf"
    executable.write_bytes(b"executable")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _tool, path=None: str(executable),
    )
    monkeypatch.setattr("flameox.application.inference_providers.version", lambda _tool: "0.13.0")

    result = discover_inference_tool(InferenceTool.AIPERF)

    assert result.available is False
    assert result.compatible is False
    assert result.executable_digest is not None
    assert result.compatibility_reason is not None


def test_sglang_bench_uses_fixed_module_random_workload_and_safe_output(tmp_path: Path) -> None:
    request = SglangBenchServingRequest(
        benchmark_python=Path("/opt/sglang/bin/python"),
        base_url="http://127.0.0.1:8000",
        model="model",
        tokenizer="tokenizer",
        result_path=tmp_path / "result.jsonl",
        num_prompts=3,
        random_input_len=16,
        random_output_len=8,
        random_range_ratio=0.5,
        endpoint_type=InferenceEndpointType.CHAT,
    )

    argv = request.argv()

    assert argv[:3] == ("/opt/sglang/bin/python", "-m", "sglang.bench_serving")
    assert "sglang-oai-chat" in argv
    assert "--output-details" not in argv
    assert argv[argv.index("--output-file") + 1] == str(tmp_path / "result.jsonl")
    assert argv[argv.index("--warmup-requests") + 1] == "0"


def test_sglang_bench_rejects_base_url_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="root base_url"):
        SglangBenchServingRequest(
            benchmark_python=Path("/opt/sglang/bin/python"),
            base_url="http://127.0.0.1:8000/api",
            model="model",
            result_path=tmp_path / "result.jsonl",
            num_prompts=1,
            random_input_len=4,
            random_output_len=2,
        )


def test_sglang_discovery_uses_the_bounded_broker(tmp_path: Path) -> None:
    from flameox.domain import ProcessResult, process_termination_from_returncode
    from flameox.execution import (
        ExecutionOutcome,
        ExecutionRequest,
        ProcessContainment,
        SubprocessBroker,
    )

    executable = tmp_path / "python"
    executable.write_text("launcher")
    executable.chmod(0o755)

    class Broker(SubprocessBroker):
        def __init__(self) -> None:
            self.request: ExecutionRequest | None = None

        def run_sync(self, request: ExecutionRequest, **_kwargs: object) -> ExecutionOutcome:
            self.request = request
            return ExecutionOutcome(
                process=ProcessResult(termination=process_termination_from_returncode(0)),
                stdout=b"0.5.16\n",
                stderr=b"",
                resolved_executable=executable,
                executable_binding=request.executable_binding,
                containment=ProcessContainment.PROCESS_GROUP,
            )

    broker = Broker()
    discovery = discover_sglang(executable, broker=broker)

    assert discovery.available is True
    assert discovery.remediation == ()
    assert broker.request is not None
    assert broker.request.timeout_seconds == 5
    assert broker.request.max_output_bytes == 1024
