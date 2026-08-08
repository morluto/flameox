from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.application.workloads import (
    ConfigureInferenceScenarioRequest,
    ConfigureInferenceServerRequest,
    InferenceScenarioConfig,
    InferenceServerConfig,
    ProjectConfig,
    WorkloadService,
)
from flameox.storage import Workspace


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
        InferenceServerConfig(
            mode="managed",
            workload="serve",
            base_url="http://localhost:8000",
            model="model",
        )


@pytest.mark.parametrize("launcher", [None, "python", "relative/python"])
def test_sglang_server_requires_an_absolute_benchmark_launcher(launcher: str | None) -> None:
    with pytest.raises(ValidationError, match="absolute benchmark_python"):
        InferenceServerConfig(
            provider="sglang",
            benchmark_python=launcher,
            mode="existing_local",
            model="model",
        )


def test_sglang_scenario_requires_random_workload_and_sglang_server() -> None:
    with pytest.raises(ValidationError, match="requires random_input_len"):
        InferenceScenarioConfig(server="local", provider="sglang_bench")
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
    with pytest.raises(ValidationError, match="burstiness is unsupported"):
        InferenceScenarioConfig(
            server="local",
            provider="sglang_bench",
            random_input_len=4,
            random_output_len=2,
            request_rate=1,
            burstiness=0.5,
        )


def test_sglang_config_rejects_non_cuda_v1_escape_hatches() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        InferenceServerConfig.model_validate(
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
            operation="create",
            config=InferenceServerConfig(
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
            operation="create",
            config=InferenceScenarioConfig(
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
            operation="create",
            config=InferenceServerConfig(
                mode="existing_local", model="model", base_url="http://127.0.0.1:8000"
            ),
        )
    )

    replaced = service.configure_inference_server(
        ConfigureInferenceServerRequest(
            name="local",
            operation="replace",
            expected_configuration_id=created.configuration_id,
            config=InferenceServerConfig(
                mode="existing_local", model="model-2", base_url="http://127.0.0.1:8000"
            ),
        )
    )

    assert replaced.action == "updated"
