from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.application.workloads import (
    ConfigurationOperation,
    ConfigureInferenceScenarioRequest,
    ConfigureInferenceServerRequest,
    InferenceScenarioConfig,
    InferenceServerConfig,
    ProjectConfig,
    WorkloadService,
    parse_inference_scenario_config,
    parse_inference_server_config,
)
from flameox.storage import Workspace


def _server(**config: object) -> InferenceServerConfig:
    return parse_inference_server_config(config)


def _scenario(**config: object) -> InferenceScenarioConfig:
    return parse_inference_scenario_config(config)


@pytest.mark.parametrize(
    "config",
    (
        {"mode": "managed", "workload": "serve", "model": "model"},
        {"mode": "existing_local", "model": "model"},
        {
            "provider": "sglang",
            "benchmark_python": "/opt/sglang/bin/python",
            "mode": "managed",
            "workload": "serve",
            "model": "model",
        },
        {
            "provider": "sglang",
            "benchmark_python": "/opt/sglang/bin/python",
            "mode": "existing_local",
            "model": "model",
        },
    ),
)
def test_inference_server_parser_round_trips_each_legal_case(
    config: dict[str, object],
) -> None:
    parsed = parse_inference_server_config(config)

    assert parse_inference_server_config(parsed.model_dump(mode="python")) == parsed


@pytest.mark.parametrize(
    "config",
    (
        {"server": "local", "provider": "aiperf"},
        {"server": "local", "provider": "vllm_bench"},
        {
            "server": "local",
            "provider": "sglang_bench",
            "random_input_len": 4,
            "random_output_len": 2,
        },
    ),
)
def test_inference_scenario_parser_round_trips_each_provider(
    config: dict[str, object],
) -> None:
    parsed = parse_inference_scenario_config(config)

    assert parse_inference_scenario_config(parsed.model_dump(mode="python")) == parsed


def test_inference_configuration_references_managed_workload() -> None:
    config = ProjectConfig.model_validate(
        {
            "workloads": {"serve": {"argv": ["vllm", "serve", "model"]}},
            "inference_servers": {
                "local": {
                    "provider": "vllm",
                    "mode": "managed",
                    "workload": "serve",
                    "model": "model",
                }
            },
            "inference_scenarios": {
                "replay": {
                    "server": "local",
                    "provider": "aiperf",
                    "endpoint_type": "chat",
                    "streaming": True,
                }
            },
        }
    )

    assert config.inference_servers["local"].base_url == "http://127.0.0.1:8000"
    assert config.inference_scenarios["replay"].provider == "aiperf"


@pytest.mark.parametrize("base_url", ["https://127.0.0.1:8000", "http://example.test:8000"])
def test_existing_inference_server_must_be_loopback_http(base_url: str) -> None:
    with pytest.raises(ValidationError, match=r"loopback|unauthenticated http"):
        ProjectConfig.model_validate(
            {
                "inference_servers": {
                    "local": {"mode": "existing_local", "base_url": base_url, "model": "model"}
                }
            }
        )


def test_inference_scenario_requires_declared_server() -> None:
    with pytest.raises(ValidationError, match="unknown servers"):
        ProjectConfig.model_validate(
            {"inference_scenarios": {"replay": {"server": "missing", "provider": "aiperf"}}}
        )


def test_managed_inference_server_rejects_localhost_name() -> None:
    with pytest.raises(ValidationError, match="IP-literal loopback"):
        _server(
            mode="managed",
            workload="serve",
            base_url="http://localhost:8000",
            model="model",
        )


@pytest.mark.parametrize("launcher", [None, "python", "relative/python"])
def test_sglang_server_requires_an_absolute_benchmark_launcher(launcher: str | None) -> None:
    with pytest.raises(ValidationError, match="benchmark_python"):
        _server(
            provider="sglang",
            benchmark_python=launcher,
            mode="existing_local",
            model="model",
        )


def test_sglang_server_rejects_non_root_base_url() -> None:
    with pytest.raises(ValidationError, match="root base_url"):
        _server(
            provider="sglang",
            benchmark_python="/opt/sglang/bin/python",
            mode="existing_local",
            base_url="http://127.0.0.1:8000/api",
            model="model",
        )


def test_sglang_scenario_requires_random_workload_and_sglang_server() -> None:
    with pytest.raises(ValidationError, match="random_input_len"):
        _scenario(server="local", provider="sglang_bench")
    with pytest.raises(ValidationError, match="require an sglang inference server"):
        ProjectConfig.model_validate(
            {
                "inference_servers": {"local": {"mode": "existing_local", "model": "model"}},
                "inference_scenarios": {
                    "replay": {
                        "server": "local",
                        "provider": "sglang_bench",
                        "random_input_len": 4,
                        "random_output_len": 2,
                    }
                },
            }
        )


def test_sglang_scenario_rejects_dropped_burstiness() -> None:
    with pytest.raises(ValidationError, match="burstiness"):
        _scenario(
            server="local",
            provider="sglang_bench",
            random_input_len=4,
            random_output_len=2,
            request_rate=1,
            burstiness=0.5,
        )


@pytest.mark.parametrize(
    ("provider", "trace_artifact_id", "random_input_len", "random_output_len", "message"),
    [
        ("aiperf", None, None, None, "requires an aiperf trace_artifact_id"),
        ("vllm_bench", None, None, None, "speedup_ratio"),
        ("sglang_bench", None, 4, 2, "speedup_ratio"),
    ],
)
def test_inference_scenario_rejects_speedup_when_provider_cannot_apply_it(
    provider: str,
    trace_artifact_id: str | None,
    random_input_len: int | None,
    random_output_len: int | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _scenario(
            server="local",
            provider=provider,
            trace_artifact_id=trace_artifact_id,
            random_input_len=random_input_len,
            random_output_len=random_output_len,
            speedup_ratio=2.0,
        )


def test_sglang_config_rejects_non_cuda_v1_escape_hatches() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        parse_inference_server_config(
            {
                "provider": "sglang",
                "benchmark_python": "/opt/sglang/bin/python",
                "mode": "existing_local",
                "model": "model",
                "rocm": True,
            }
        )


def test_structured_inference_configuration_preserves_existing_sections(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        'schema_version = 1\n[workloads.serve]\nargv = ["python", "-c", "pass"]\n'
    )
    service = WorkloadService(workspace)

    server = service.configure_inference_server(
        ConfigureInferenceServerRequest(
            name="local",
            operation=ConfigurationOperation.CREATE,
            config=_server(
                mode="managed",
                workload="serve",
                model="model",
                model_revision="model-rev",
                tokenizer="tokenizer",
                tokenizer_revision="tokenizer-rev",
                quantization="none",
            ),
        )
    )
    scenario = service.configure_inference_scenario(
        ConfigureInferenceScenarioRequest(
            name="replay",
            operation=ConfigurationOperation.CREATE,
            config=_scenario(
                server="local",
                provider="aiperf",
                request_rate=12.5,
                burstiness=0.8,
                warmup_request_count=3,
                seed=17,
            ),
        )
    )

    loaded = service.load()
    assert server.action == "created"
    assert scenario.action == "created"
    assert loaded.workloads["serve"].argv == ("python", "-c", "pass")
    assert loaded.inference_servers["local"].tokenizer == "tokenizer"
    assert loaded.inference_servers["local"].model_revision == "model-rev"
    assert loaded.inference_servers["local"].tokenizer_revision == "tokenizer-rev"
    assert loaded.inference_servers["local"].quantization == "none"
    assert loaded.inference_scenarios["replay"].streaming is True
    assert loaded.inference_scenarios["replay"].request_rate == 12.5
    assert loaded.inference_scenarios["replay"].burstiness == 0.8
    assert loaded.inference_scenarios["replay"].warmup_request_count == 3
    assert loaded.inference_scenarios["replay"].seed == 17


def test_inference_semantic_oracle_requires_receipt_strength() -> None:
    with pytest.raises(ValidationError, match="contract-check receipt oracle"):
        ProjectConfig.model_validate(
            {
                "workloads": {
                    "oracle": {
                        "argv": ["python", "-c", "pass"],
                        "oracle": {
                            "strength": "execution_check",
                            "argv": ["python", "-c", "pass"],
                        },
                    }
                },
                "inference_servers": {"local": {"mode": "existing_local", "model": "model"}},
                "inference_scenarios": {
                    "replay": {
                        "server": "local",
                        "provider": "aiperf",
                        "semantic_oracle_workload": "oracle",
                    }
                },
            }
        )


def test_structured_inference_replace_requires_current_digest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    created = service.configure_inference_server(
        ConfigureInferenceServerRequest(
            name="local",
            operation=ConfigurationOperation.CREATE,
            config=_server(mode="existing_local", model="model", base_url="http://127.0.0.1:8000"),
        )
    )

    replaced = service.configure_inference_server(
        ConfigureInferenceServerRequest(
            name="local",
            operation=ConfigurationOperation.REPLACE,
            expected_configuration_id=created.configuration_id,
            config=_server(
                mode="existing_local", model="model-2", base_url="http://127.0.0.1:8000"
            ),
        )
    )

    assert replaced.action == "updated"
