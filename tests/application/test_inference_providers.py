from __future__ import annotations

from pathlib import Path

import httpx
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
    probe_existing_vllm_server,
    probe_existing_vllm_server_async,
)
from flameox.application.provider_runtime import ProviderRuntime, ProviderRuntimeReceipt
from flameox.domain import CapabilityExtra, DomainError, ErrorCode
from flameox.http_transport import BoundedHttpClient

pytestmark = pytest.mark.unit


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
                "executable": Path("/opt/sglang/bin/python"),
                "base_url": "http://127.0.0.1:8000",
                "model": "model",
                "result_path": tmp_path / "result.jsonl",
                "num_prompts": 1,
                "random_input_len": 4,
                "random_output_len": 2,
                "streaming": False,
            }
        )


def test_existing_server_probe_uses_only_read_endpoints() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        body = b'{"data":[{"id":"model"}]}' if request.url.path == "/v1/models" else b""
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(body),
        )

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        probe = probe_existing_vllm_server(
            "http://127.0.0.1:8000",
            http_client=client,
        )

    assert probe.model_ids == ("model",)
    assert calls == ["http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1/models"]


@pytest.mark.anyio
async def test_existing_server_probe_has_native_async_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = b'{"data":[{"id":"model"}]}' if request.url.path == "/v1/models" else b""
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(body),
        )

    async with BoundedHttpClient(async_transport=httpx.MockTransport(handler)) as client:
        probe = await probe_existing_vllm_server_async(
            "http://127.0.0.1:8000",
            http_client=client,
        )

    assert probe.model_ids == ("model",)


def test_aiperf_discovery_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "bin" / "aiperf"
    executable.parent.mkdir()
    executable.write_bytes(b"executable")
    executable.chmod(0o755)
    python = tmp_path / "bin" / "python"
    python.write_bytes(b"python")
    python.chmod(0o755)
    runtime = ProviderRuntime(
        tmp_path,
        ProviderRuntimeReceipt(
            environment_id="sha256:" + "b" * 64,
            flameox_version="0.1.13",
            flameox_package_source="index",
            extra=CapabilityExtra.INFERENCE,
            requirement="aiperf>=0.12,<0.13",
            python_requirement="3.12",
            platform="linux",
            architecture="x86_64",
            uv_version="0.9.0",
            uv_sha256="sha256:" + "c" * 64,
            python_relative_path="bin/python",
            python_sha256="sha256:" + "d" * 64,
            distributions={"flameox": "0.1.13", "aiperf": "0.13.0"},
            executable_relative_path="bin/aiperf",
            executable_sha256="sha256:" + "e" * 64,
        ),
    )

    with pytest.raises(DomainError, match="outside Flameox") as exc_info:
        discover_inference_tool(InferenceTool.AIPERF, provider_runtime=runtime)

    assert exc_info.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert exc_info.value.next_action is not None


def test_sglang_bench_uses_fixed_module_random_workload_and_safe_output(tmp_path: Path) -> None:
    request = SglangBenchServingRequest(
        executable=Path("/opt/sglang/bin/python"),
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

    assert argv[:3] == ("/opt/sglang/bin/python", "-m", "sglang.benchmark.serving")
    assert "sglang-oai-chat" in argv
    assert "--output-details" not in argv
    assert argv[argv.index("--output-file") + 1] == str(tmp_path / "result.jsonl")
    assert argv[argv.index("--warmup-requests") + 1] == "0"


def test_sglang_bench_rejects_base_url_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="without URL extras"):
        SglangBenchServingRequest(
            executable=Path("/opt/sglang/bin/python"),
            base_url="http://127.0.0.1:8000/api",
            model="model",
            result_path=tmp_path / "result.jsonl",
            num_prompts=1,
            random_input_len=4,
            random_output_len=2,
        )


@pytest.mark.parametrize("reported_version", ("0.5.16", "0.5.18"))
def test_sglang_discovery_qualifies_supported_launcher_versions(
    tmp_path: Path,
    reported_version: str,
) -> None:
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
            stdout = (
                b"\n".join(
                    option.encode()
                    for option in (
                        "--backend",
                        "--host",
                        "--port",
                        "--model",
                        "--dataset-name",
                        "--random-input-len",
                        "--random-output-len",
                        "--random-range-ratio",
                        "--num-prompts",
                        "--seed",
                        "--output-file",
                        "--warmup-requests",
                    )
                )
                if request.argv[-1] == "--help"
                else f"{reported_version}\n".encode()
            )
            return ExecutionOutcome(
                process=ProcessResult(termination=process_termination_from_returncode(0)),
                stdout=stdout,
                stderr=b"",
                resolved_executable=executable,
                executable_binding=request.executable_binding,
                containment=ProcessContainment.PROCESS_GROUP,
            )

    broker = Broker()
    discovery = discover_sglang(executable, broker=broker)

    assert discovery.version == reported_version
    assert discovery.executable_binding.canonical_target == executable
    assert discovery.benchmark_capabilities == (
        "--backend",
        "--host",
        "--port",
        "--model",
        "--dataset-name",
        "--random-input-len",
        "--random-output-len",
        "--random-range-ratio",
        "--num-prompts",
        "--seed",
        "--output-file",
        "--warmup-requests",
    )
    assert broker.request is not None
    assert broker.request.timeout_seconds == 5
    assert broker.request.max_output_bytes == 1024


def test_sglang_discovery_refuses_an_incomplete_launcher_interface(tmp_path: Path) -> None:
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
        def run_sync(self, request: ExecutionRequest, **_kwargs: object) -> ExecutionOutcome:
            return ExecutionOutcome(
                process=ProcessResult(termination=process_termination_from_returncode(0)),
                stdout=b"--backend\n--host\n--port\n",
                stderr=b"",
                resolved_executable=executable,
                executable_binding=request.executable_binding,
                containment=ProcessContainment.PROCESS_GROUP,
            )

    with pytest.raises(DomainError, match=r"supported sglang\.benchmark\.serving") as exc_info:
        discover_sglang(executable, broker=Broker())

    assert exc_info.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert exc_info.value.details["module"] == "sglang.benchmark.serving"
    assert "--model" in exc_info.value.details["missing_options"]
    assert exc_info.value.next_action is not None
