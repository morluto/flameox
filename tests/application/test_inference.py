from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.inference import InferenceArtifactExtractor
from flameox.analysis.inference_protocol import InferenceProtocolIdentity
from flameox.application.environment import collect_environment
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.inference import (
    InferenceReplayResult,
    InferenceReplayService,
)
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ProcessResult,
    ValidationStatus,
)
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker
from flameox.storage import RunStore, Workspace

DIGEST = "sha256:" + "a" * 64


def _write_project(tmp_path: Path, *, server_mode: str = "existing_local") -> None:
    workload = (
        'argv = ["python", "-c", "import time; time.sleep(60)"]'
        if server_mode == "managed"
        else 'argv = ["python", "-c", "print(\'replay\')"]'
    )
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1

[workloads.replay]
{workload}

[inference_servers.local]
provider = "vllm"
mode = "{server_mode}"
{"workload = 'replay'" if server_mode == "managed" else ""}
base_url = "http://127.0.0.1:8000"
model = "test-model"

[inference_scenarios.aiperf_replay]
server = "local"
provider = "aiperf"
endpoint_type = "chat"
streaming = true
num_prompts = 10
concurrency = 2
speedup_ratio = 1.0

[inference_scenarios.vllm_bench_replay]
server = "local"
provider = "vllm_bench"
endpoint_type = "completions"
streaming = true
num_prompts = 5
"""
    )


class RecordingBroker(SubprocessBroker):
    def __init__(self, *, exit_code: int = 0, stdout: bytes = b"{}", stderr: bytes = b"") -> None:
        self.requests: list[ExecutionRequest] = []
        self._exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        self.requests.append(request)
        return self._outcome(request)

    def run_sync(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        self.requests.append(request)
        return self._outcome(request)

    def _outcome(self, request: ExecutionRequest) -> ExecutionOutcome:
        return ExecutionOutcome(
            process=ProcessResult(exit_code=self._exit_code, cleanup_complete=True),
            stdout=self._stdout,
            stderr=self._stderr,
            resolved_executable=Path(request.argv[0]),
            containment="process_group",
        )


def _patch_providers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    executable: Path | None = Path("/tools/aiperf"),
    health_ready: bool = True,
    model_ids: tuple[str, ...] = ("test-model",),
) -> None:
    from flameox.application import inference as inference_module
    from flameox.application.inference_providers import (
        ExistingServerProbe,
        InferenceToolDiscovery,
    )

    def fake_discover(tool: str) -> InferenceToolDiscovery:
        return InferenceToolDiscovery(
            tool=tool,  # type: ignore[arg-type]
            executable=executable,
            available=executable is not None,
            remediation=() if executable else ("Install the inference extra.",),
        )

    def fake_probe(base_url: str, *, timeout_seconds: float = 2.0) -> ExistingServerProbe:
        del base_url, timeout_seconds
        return ExistingServerProbe(
            base_url="http://127.0.0.1:8000",
            health_ready=health_ready,
            model_ids=model_ids,
        )

    monkeypatch.setattr(inference_module, "discover_inference_tool", fake_discover)
    monkeypatch.setattr(inference_module, "probe_existing_vllm_server", fake_probe)


def test_plan_existing_local_aiperf_builds_typed_argv_and_records_exploratory_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    plan = service.plan("aiperf_replay")

    assert plan.server_mode == "existing_local"
    assert plan.provider == "aiperf"
    assert plan.tool_available is True
    assert plan.tool_executable == "/tools/aiperf"
    assert plan.argv[0] == "/tools/aiperf"
    assert "--fixed-schedule" not in plan.argv  # no trace_artifact_id
    assert plan.exploratory_reason.startswith("Single replay run is exploratory")
    assert plan.timeout_seconds == 1800.0  # aiperf default deadline
    assert plan.output_path is not None
    assert plan.configuration_id.startswith("sha256:")


def test_plan_existing_local_vllm_bench_uses_shorter_default_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/vllm"))
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    plan = service.plan("vllm_bench_replay")

    assert plan.provider == "vllm_bench"
    assert plan.argv[0] == "/tools/vllm"
    assert "bench" in plan.argv and "serve" in plan.argv
    assert plan.timeout_seconds == 600.0  # vllm_bench default deadline


def test_sglang_protocol_identity_binds_random_shape_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace.initialize(tmp_path)
    launcher = tmp_path / "sglang-python"
    launcher.write_text("launcher")
    launcher.chmod(0o755)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[inference_servers.local]\n"
        'provider = "sglang"\n'
        f'benchmark_python = "{launcher}"\n'
        'mode = "existing_local"\nbase_url = "http://127.0.0.1:8000"\nmodel = "model"\n'
        "[inference_scenarios.replay]\n"
        'server = "local"\nprovider = "sglang_bench"\n'
        "random_input_len = 4\nrandom_output_len = 2\n"
    )
    from flameox.application import inference as inference_module
    from flameox.application.inference_providers import ExistingServerProbe, InferenceToolDiscovery

    discovery = InferenceToolDiscovery(
        tool="sglang",
        executable=launcher,
        available=True,
        compatible=True,
        version="0.5.16",
        executable_digest="sha256:" + "c" * 64,
    )
    monkeypatch.setattr(inference_module, "discover_sglang", lambda _launcher, *, broker: discovery)
    monkeypatch.setattr(
        inference_module,
        "probe_existing_vllm_server",
        lambda *_args, **_kwargs: ExistingServerProbe(
            base_url="http://127.0.0.1:8000", health_ready=True, model_ids=("model",)
        ),
    )
    service = InferenceReplayService(workspace, broker=RecordingBroker())
    plan = service.plan("replay")
    reshaped = plan.model_copy(update={"random_input_len": 8})

    identity = service._protocol_identity(plan, environment=collect_environment())
    reshaped_identity = service._protocol_identity(reshaped, environment=collect_environment())

    assert identity.trace.producer == "sglang.bench_serving"
    assert identity.server.cache_backend == "custom"
    assert plan.random_range_ratio == 1.0
    assert identity.trace.artifact_digest != reshaped_identity.trace.artifact_digest


def test_each_replay_plan_uses_an_isolated_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    first = service.plan("aiperf_replay")
    second = service.plan("aiperf_replay")

    assert first.output_path != second.output_path
    assert first.output_path is not None
    assert second.output_path is not None
    assert Path(first.output_path).parent.parent == Path(second.output_path).parent.parent


def test_plan_accepts_managed_server_without_probing_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path, server_mode="managed")
    _patch_providers(monkeypatch)
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    plan = service.plan("aiperf_replay")

    assert plan.server_mode == "managed"
    assert plan.tool_available is True


@pytest.mark.anyio
async def test_run_managed_server_cleans_up_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path, server_mode="managed")
    provider = tmp_path / "aiperf"
    provider.write_text("#!/bin/sh\nexit 0\n")
    provider.chmod(0o755)
    _patch_providers(monkeypatch, executable=provider)

    service = InferenceReplayService(workspace)
    result = await service.run(service.plan("aiperf_replay", timeout_seconds=5))

    assert result.exit_code == 0
    assert result.server_cleanup_complete is True


def test_plan_unknown_scenario_raises_workspace_invalid(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    with pytest.raises(DomainError, match="not declared") as exc_info:
        service.plan("missing")

    assert exc_info.value.code is ErrorCode.WORKSPACE_INVALID


def test_plan_unavailable_tool_records_remediation_and_empty_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=None)
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    plan = service.plan("vllm_bench_replay")

    assert plan.tool_available is False
    assert plan.tool_executable is None
    assert plan.argv == ()
    assert plan.tool_remediation == ("Install the inference extra.",)


@pytest.mark.anyio
async def test_run_executes_through_broker_and_preserves_argv_and_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    broker = RecordingBroker(exit_code=0, stdout=b'{"result": "ok"}')
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))
    service = InferenceReplayService(workspace, broker=broker)

    plan = service.plan("aiperf_replay")
    assert plan.output_path is not None
    output = Path(plan.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"aggregate": true}')
    result = await service.run(plan)

    assert isinstance(result, InferenceReplayResult)
    assert result.plan_id == plan.plan_id
    assert result.argv == plan.argv
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.health_ready is True
    assert result.probed_model_ids == ("test-model",)
    assert result.containment == "process_group"
    assert result.output_path == plan.output_path
    assert result.exploratory_reason == plan.exploratory_reason
    # The exploratory reason is always included in limitations
    assert any("exploratory" in lim for lim in result.limitations)
    assert len(broker.requests) == 1
    # The remaining-time enforcement clamps to min(timeout_seconds, remaining),
    # so the actual request timeout may be slightly less than the plan's.
    assert broker.requests[0].timeout_seconds <= plan.timeout_seconds
    assert broker.requests[0].timeout_seconds > plan.timeout_seconds - 5
    assert len(result.artifact_ids) == 1
    assert len(result.artifact_run_ids) == 1
    manifest = service.runs.read(result.run_id)
    assert manifest.execution_status is ExecutionStatus.SUCCEEDED
    assert manifest.capture_status is CaptureStatus.REGISTERED
    assert manifest.inference_protocol_identity_id == manifest.measurement_protocol_id
    assert manifest.inference_protocol_identity_json is not None
    assert manifest.finished_at is not None


@pytest.mark.anyio
async def test_run_refuses_unavailable_tool_before_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    broker = RecordingBroker()
    _patch_providers(monkeypatch, executable=None)
    service = InferenceReplayService(workspace, broker=broker)

    plan = service.plan("vllm_bench_replay")
    with pytest.raises(DomainError, match="unavailable") as exc_info:
        await service.run(plan)

    assert exc_info.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert len(broker.requests) == 0


@pytest.mark.anyio
async def test_run_preserves_partial_provider_artifacts_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))

    class FailingBroker(RecordingBroker):
        async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
            output_dir = Path(request.argv[request.argv.index("--output-artifact-dir") + 1])
            nested = output_dir / "failed-run"
            nested.mkdir(parents=True)
            (nested / "profile_export.jsonl").write_text('{"partial":true}\n')
            raise DomainError(ErrorCode.PROCESS_FAILED, "provider failed")

    service = InferenceReplayService(workspace, broker=FailingBroker())

    with pytest.raises(DomainError) as caught:
        await service.run(service.plan("aiperf_replay"))

    assert caught.value.code is ErrorCode.PROCESS_FAILED
    assert caught.value.run_id is not None
    failed_run = service.runs.read(caught.value.run_id)
    assert failed_run.execution_status is ExecutionStatus.FAILED
    assert failed_run.capture_status is CaptureStatus.REGISTERED
    assert len(caught.value.details["partial_artifact_ids"]) == 1
    artifact = service.artifacts.get(caught.value.details["partial_artifact_ids"][0])
    assert artifact.payload_path.read_text() == '{"partial":true}\n'
    assert len(caught.value.details["partial_artifact_run_ids"]) == 1


@pytest.mark.anyio
async def test_run_finalizes_canonical_run_after_unexpected_broker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))

    class BrokenBroker(RecordingBroker):
        async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
            del request
            raise RuntimeError("broker implementation failed")

    service = InferenceReplayService(workspace, broker=BrokenBroker())

    with pytest.raises(DomainError) as caught:
        await service.run(service.plan("aiperf_replay"))

    assert caught.value.code is ErrorCode.INTERNAL_ERROR
    assert caught.value.run_id is not None
    failed_run = service.runs.read(caught.value.run_id)
    assert failed_run.execution_status is ExecutionStatus.FAILED
    assert failed_run.finished_at is not None


@pytest.mark.anyio
async def test_run_records_unhealthy_server_as_limitation_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    broker = RecordingBroker()
    _patch_providers(monkeypatch, executable=Path("/tools/vllm"), health_ready=False, model_ids=())
    service = InferenceReplayService(workspace, broker=broker)

    plan = service.plan("vllm_bench_replay")
    result = await service.run(plan)

    assert result.health_ready is False
    assert result.probed_model_ids == ()
    assert any("health probe was not ready" in lim for lim in result.limitations)
    assert any("no model ids" in lim for lim in result.limitations)


def test_run_sync_executes_and_returns_typed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    broker = RecordingBroker(exit_code=0)
    _patch_providers(monkeypatch, executable=Path("/tools/vllm"))
    service = InferenceReplayService(workspace, broker=broker)

    plan = service.plan("vllm_bench_replay")
    result = service.run_sync(plan)

    assert isinstance(result, InferenceReplayResult)
    assert result.exit_code == 0
    assert result.argv == plan.argv
    assert len(broker.requests) == 1


def test_run_sync_executes_semantic_oracle_and_persists_safe_receipt_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    config_path = tmp_path / "flameox.toml"
    config_path.write_text(
        config_path.read_text().replace(
            "[inference_scenarios.vllm_bench_replay]",
            """[workloads.semantic]
argv = ["python", "-c", "pass"]

[workloads.semantic.oracle]
strength = "contract_check"
argv = ["python", "-c", "pass"]
receipt_schema = "flameox.oracle-receipt.v1"

[inference_scenarios.vllm_bench_replay]""",
        )
        + '\nsemantic_oracle_workload = "semantic"\n'
    )
    _patch_providers(monkeypatch, executable=Path("/tools/vllm"))

    class OracleBroker(RecordingBroker):
        def run_sync(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
            self.requests.append(request)
            receipt_path = request.environment_overrides.get("FLAMEOX_ORACLE_RECEIPT")
            if receipt_path is not None:
                Path(receipt_path).write_text(
                    json.dumps(
                        {
                            "schema_version": "flameox.oracle-receipt.v1",
                            "status": "pass",
                            "reason": "response_contract_valid",
                        }
                    )
                )
                return ExecutionOutcome(
                    process=ProcessResult(exit_code=0, cleanup_complete=True),
                    stdout=b"native oracle diagnostics",
                    stderr=b"",
                    resolved_executable=Path(request.argv[0]),
                    containment="process_group",
                )
            return self._outcome(request)

    service = InferenceReplayService(workspace, broker=OracleBroker())
    plan = service.plan("vllm_bench_replay")
    result = service.run_sync(plan)

    assert result.oracle_status == "pass"
    assert len(service.broker.requests) == 2  # type: ignore[attr-defined]
    manifest = service.runs.read(result.run_id)
    assert manifest.validation_status is ValidationStatus.PASSED
    assert manifest.inference_protocol_identity_json is not None
    identity = InferenceProtocolIdentity.model_validate_json(
        manifest.inference_protocol_identity_json
    )
    assert identity.oracle.kind == "contract_check"
    assert identity.oracle.command_digest is not None
    assert identity.oracle_result is not None
    assert identity.oracle_result.reason == "response_contract_valid"
    assert "native oracle diagnostics" not in manifest.inference_protocol_identity_json
    assert any(
        service.artifacts.get(artifact_id).payload_path.read_bytes() == b"native oracle diagnostics"
        for artifact_id in result.artifact_ids
    )


def test_run_sync_publishes_vllm_measurements_under_canonical_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/vllm"))
    service = InferenceReplayService(workspace, broker=RecordingBroker(exit_code=0))
    plan = service.plan("vllm_bench_replay")
    assert plan.output_path is not None
    output = Path(plan.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metrics": {
                    "completed": 2,
                    "total_input": 20,
                    "total_output": 10,
                    "request_throughput": 2.0,
                    "output_throughput": 10.0,
                    "total_token_throughput": 30.0,
                    "mean_ttft_ms": 1.0,
                    "median_ttft_ms": 1.0,
                    "std_ttft_ms": 0.0,
                    "mean_tpot_ms": 2.0,
                    "median_tpot_ms": 2.0,
                    "std_tpot_ms": 0.0,
                    "mean_itl_ms": 2.0,
                    "median_itl_ms": 2.0,
                    "std_itl_ms": 0.0,
                    "mean_e2el_ms": 5.0,
                    "median_e2el_ms": 5.0,
                    "std_e2el_ms": 0.0,
                },
                "successful_requests": 2,
                "failed_requests": 0,
                "total_requests": 2,
                "actual_duration": 1.0,
            }
        )
    )

    result = service.run_sync(plan)

    measurements = EvidenceQueryService(workspace).measurements(run_id=result.run_id, limit=100)
    assert measurements.measurements
    assert {row.run_id for row in measurements.measurements} == {result.run_id}
    assert any(row.name == "vllm.request_throughput" for row in measurements.measurements)


def test_run_sync_correlates_aiperf_outputs_under_canonical_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))
    service = InferenceReplayService(workspace, broker=RecordingBroker(exit_code=0))
    plan = service.plan("aiperf_replay")
    assert plan.output_path is not None
    output_dir = Path(plan.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "session_id": "conversation-a",
                        "payloads": [{"prompt": "native only"}],
                    }
                ]
            }
        )
    )
    (output_dir / "profile_export.jsonl").write_text(
        json.dumps(
            {
                "metadata": {
                    "session_num": 0,
                    "conversation_id": "conversation-a",
                    "turn_index": 0,
                    "x_request_id": "provider-1",
                    "request_start_ns": 100,
                    "request_end_ns": 1_000_100,
                    "was_cancelled": False,
                },
                "metrics": {
                    "input_sequence_length": {"value": 12, "unit": "tokens"},
                    "output_sequence_length": {"value": 3, "unit": "tokens"},
                    "time_to_first_token": {"value": 0.5, "unit": "ms"},
                    "request_latency": {"value": 1, "unit": "ms"},
                },
                "raw_prompt": "must not enter normalized evidence",
            }
        )
        + "\n"
    )

    result = service.run_sync(plan)

    requests = EvidenceQueryService(workspace).inference_requests(run_id=result.run_id, limit=100)
    assert requests.returned == 1
    assert requests.requests[0].run_id == result.run_id
    assert requests.requests[0].source_request_id == "conversation-a:0"
    assert requests.requests[0].provider_request_id == "provider-1"
    assert "native only" not in requests.model_dump_json()
    assert "must not enter" not in requests.model_dump_json()
    measurements = EvidenceQueryService(workspace).measurements(
        run_id=result.run_id, name_prefix="aiperf.", limit=100
    )
    names = {row.name for row in measurements.measurements}
    assert "aiperf.request_throughput" in names
    assert "aiperf.ttft.median_ms" in names
    assert "aiperf.end_to_end_latency.p95_ms" in names
    canonical = RunStore(workspace).read(result.run_id)
    assert {artifact.display_name for artifact in canonical.artifacts} >= {
        "inputs.json",
        "profile_export.jsonl",
    }
    reextracted = InferenceArtifactExtractor(workspace).extract_aiperf_result(result.run_id)
    assert reextracted.evidence_run_id == result.run_id
    requests_after = EvidenceQueryService(workspace).inference_requests(
        run_id=result.run_id, limit=100
    )
    measurements_after = EvidenceQueryService(workspace).measurements(
        run_id=result.run_id, name_prefix="aiperf.", limit=100
    )
    assert requests_after.total == requests.total == 1
    assert measurements_after.total == measurements.total


def test_success_retains_staging_when_native_artifact_import_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/vllm"))
    service = InferenceReplayService(workspace, broker=RecordingBroker(exit_code=0))
    plan = service.plan("vllm_bench_replay")
    assert plan.output_path is not None
    output = Path(plan.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}")

    def fail_import(*_args: object, **_kwargs: object) -> object:
        raise DomainError(ErrorCode.ARTIFACT_PARSE_FAILED, "injected import failure")

    monkeypatch.setattr("flameox.application.inference.ImportService.import_artifact", fail_import)

    result = service.run_sync(plan)

    assert output.read_text() == "{}"
    assert result.output_path_retained is True
    assert any("staging was retained" in item for item in result.limitations)


def test_output_discovery_reports_file_limit_instead_of_claiming_complete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-output"
    root.mkdir()
    for index in range(129):
        (root / f"result-{index}.json").write_text("{}")

    candidates, limitations = InferenceReplayService._bounded_output_candidates(root)

    assert len(candidates) == 128
    assert limitations == ("Provider output discovery stopped at 128 files.",)


def test_plan_honors_explicit_timeout_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    plan = service.plan("aiperf_replay", timeout_seconds=42.0)

    assert plan.timeout_seconds == 42.0


def test_plan_rejects_out_of_range_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch)
    service = InferenceReplayService(workspace, broker=RecordingBroker())

    with pytest.raises(DomainError, match="timeout_seconds") as exc_info:
        service.plan("aiperf_replay", timeout_seconds=0.0)

    assert exc_info.value.code is ErrorCode.EXECUTION_REFUSED


def test_run_sync_rejects_stale_inference_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_project(tmp_path)
    _patch_providers(monkeypatch, executable=Path("/tools/aiperf"))
    service = InferenceReplayService(workspace, broker=RecordingBroker())
    plan = service.plan("aiperf_replay")
    config_path = tmp_path / "flameox.toml"
    config_path.write_text(config_path.read_text().replace("num_prompts = 10", "num_prompts = 11"))

    with pytest.raises(DomainError, match="changed after this plan") as caught:
        service.run_sync(plan)

    assert caught.value.code is ErrorCode.REVISION_CONFLICT
