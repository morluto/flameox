from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    digest_model,
    missing_artifact_input,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace
from flameox.workers.nsight_compute_contract import (
    NSIGHT_COMPUTE_WORKER,
    NsightComputeWorkerRequest,
)

_NCU_INSTALL_ROOTS = (
    Path("/opt/nvidia/nsight-compute"),
    Path("/usr/local/NVIDIA-Nsight-Compute"),
)
_WORKER_RESPONSE_OVERHEAD_BYTES = 64 * 1024
_WORKER_RESPONSE_BYTES_PER_ROW = 16 * 1024
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1


class NsightComputeExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    producer_version: str | None
    report_version: str
    range_count: int
    action_count: int
    metric_count: int
    observation_count: int
    truncated: bool
    report_interface_sha256: str
    schema_fingerprint: str
    corpus_commit_id: str
    limitations: tuple[str, ...]


def find_ncu_report_interface(
    *,
    executable: str | Path | None = None,
    producer_version: str | None = None,
) -> Path | None:
    """Find the Python interface shipped with Nsight Compute, never a PyPI substitute."""

    resolved_executable = Path(executable).resolve() if executable else None
    if resolved_executable is not None:
        for parent in resolved_executable.parents:
            candidate = parent / "extras" / "python" / "ncu_report.py"
            if candidate.is_file():
                return candidate

    candidates: set[Path] = set()
    for base in _NCU_INSTALL_ROOTS:
        if base.is_dir():
            candidates.update(base.glob("*/extras/python/ncu_report.py"))
            direct = base / "extras" / "python" / "ncu_report.py"
            if direct.is_file():
                candidates.add(direct)
    if not candidates:
        return None

    def parsed_version(value: str) -> Version:
        match = re.search(r"\d+(?:\.\d+)+", value)
        if match is None:
            return Version("0")
        try:
            return Version(match.group())
        except InvalidVersion:
            return Version("0")

    producer = parsed_version(producer_version) if producer_version else Version("0")

    def key(path: Path) -> tuple[int, Version, str]:
        version_text = path.parents[2].name
        candidate = parsed_version(version_text)
        exact = int(producer != Version("0") and candidate == producer)
        return exact, candidate, path.as_posix()

    return max(candidates, key=key)


class NsightComputeExtractor:
    name = "nsight.compute.report"
    version = "1"
    compatibility_family = "nsight-compute.ncu-report-api.v1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> NsightComputeExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        matches = tuple(item for item in run.artifacts if item.kind is ArtifactKind.KERNEL_PROFILE)
        if not matches:
            raise missing_artifact_input(
                run_id=run_id,
                requirement="Nsight Compute kernel-profile",
                artifact_kinds=(ArtifactKind.KERNEL_PROFILE.value,),
                capture_adapters=("nsight.compute",),
                import_producers=("nsight.compute",),
            )
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one Nsight Compute report.",
                run_id=run_id,
            )
        registration = matches[0]
        if registration.producer not in {"nsight.compute", "ncu", "flameox.import"}:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The kernel profile is not registered as Nsight Compute output.",
                run_id=run_id,
                details={"registered_producer": registration.producer},
            )
        if not registration.display_name.casefold().endswith((".ncu-rep", ".ncu-repz")):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Only unchanged .ncu-rep and .ncu-repz reports are supported.",
            )
        binding = ExecutableResolver().resolve_host_tool("ncu")
        interface = find_ncu_report_interface(
            executable=str(binding.invocation_path) if binding is not None else None,
            producer_version=registration.producer_version,
        )
        if interface is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The official ncu_report Python interface was not found.",
                remediation=(
                    "Install Nsight Compute with its extras/python interface, then retry; "
                    "FlameOx does not decode NVIDIA report binaries.",
                ),
            )
        with interface.open("rb") as stream:
            report_interface_sha256 = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        maximum = self.workspace.config.storage.max_rows_per_generation
        if maximum < 3:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Nsight Compute extraction requires room for metrics, observations, "
                "and provenance.",
            )
        response_bytes = self.workspace.config.execution.max_output_bytes
        response_rows = max(
            0,
            (response_bytes - _WORKER_RESPONSE_OVERHEAD_BYTES) // _WORKER_RESPONSE_BYTES_PER_ROW,
        )
        normalized_rows = min(maximum - 1, response_rows)
        if normalized_rows < 2:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Nsight Compute extraction requires a worker response budget of at least "
                f"{_WORKER_RESPONSE_OVERHEAD_BYTES + 2 * _WORKER_RESPONSE_BYTES_PER_ROW} bytes.",
            )
        max_metrics = normalized_rows // 2
        max_observations = normalized_rows - max_metrics
        response = IsolatedWorkerHarness(self.workspace).run_typed_sync(
            NSIGHT_COMPUTE_WORKER,
            NsightComputeWorkerRequest(
                artifact_path=str(artifact.payload_path),
                interface_path=str(interface),
                max_ranges=min(1_000, max_observations),
                max_actions=min(10_000, max_observations),
                max_metrics=max_metrics,
                max_observations=max_observations,
            ),
        )
        measurements = [dict(item) for item in response.measurements]
        observations = [dict(item) for item in response.observations]
        metric_ids = list(response.metric_ids)
        section_ids = list(response.section_ids)
        limitations = list(response.limitations)
        report_version = response.report_version
        schema_fingerprint = digest_model(
            {
                "compatibility_family": self.compatibility_family,
                "report_interface_sha256": report_interface_sha256,
                "report_version": report_version,
                "metric_ids": metric_ids,
                "section_ids": section_ids,
            }
        )
        measurement_rows = [
            self._measurement_row(run_id, registration.artifact_id, index, item)
            for index, item in enumerate(measurements)
        ]
        observation_rows = [
            self._observation_row(run_id, registration.artifact_id, index, item)
            for index, item in enumerate(observations)
        ]
        observation_rows.append(
            self._observation_row(
                run_id,
                registration.artifact_id,
                len(observations),
                {
                    "kind": "profile.extraction",
                    "name": self.compatibility_family,
                    "value": {
                        "metric_ids": metric_ids,
                        "producer_version": registration.producer_version,
                        "report_interface_sha256": report_interface_sha256,
                        "report_version": report_version,
                        "schema_fingerprint": schema_fingerprint,
                        "section_ids": section_ids,
                        "range_count": response.range_count,
                        "action_count": response.action_count,
                        "truncated": response.truncated,
                    },
                },
            )
        )
        published = self.publisher.publish_rows_idempotent(
            {"measurements": measurement_rows, "observations": observation_rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "compatibility_family": self.compatibility_family,
                "report_interface_sha256": report_interface_sha256,
                "report_version": report_version,
                "schema_fingerprint": schema_fingerprint,
                "max_metrics": max_metrics,
                "max_observations": max_observations,
            },
        )
        return NsightComputeExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer_version=registration.producer_version,
            report_version=report_version,
            range_count=response.range_count,
            action_count=response.action_count,
            metric_count=len(measurements),
            observation_count=len(observations),
            truncated=response.truncated,
            report_interface_sha256=report_interface_sha256,
            schema_fingerprint=schema_fingerprint,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _measurement_row(
        run_id: str,
        artifact_id: str,
        index: int,
        metric: dict[str, Any],
    ) -> dict[str, object]:
        value = metric.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DomainError(ErrorCode.ARTIFACT_PARSE_FAILED, "Invalid numeric metric value.")
        provider_value_kind = metric.get("provider_value_kind")
        if not isinstance(provider_value_kind, str):
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The ncu_report worker omitted the provider numeric value kind.",
            )
        value_int: int | None = None
        value_float: float | None = None
        value_uint: int | None = None
        normalized_value_kind: str
        if provider_value_kind in {"uint32", "uint64"}:
            maximum = 2**32 - 1 if provider_value_kind == "uint32" else _UINT64_MAX
            if not isinstance(value, int) or not 0 <= value <= maximum:
                raise DomainError(
                    ErrorCode.ADAPTER_INCOMPATIBLE,
                    "Unsigned ncu_report metric value contradicts its provider kind.",
                )
            value_uint = value
            normalized_value_kind = "unsigned_integer"
        elif provider_value_kind == "integer":
            if not isinstance(value, int) or not _INT64_MIN <= value <= _INT64_MAX:
                raise DomainError(
                    ErrorCode.ADAPTER_INCOMPATIBLE,
                    "Integer ncu_report metric cannot be represented as a signed int64.",
                )
            value_int = value
            normalized_value_kind = "integer"
        elif provider_value_kind in {"float", "float32", "float64"}:
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise DomainError(
                    ErrorCode.ADAPTER_INCOMPATIBLE,
                    "Floating ncu_report metric is not finite.",
                )
            value_float = float(value)
            normalized_value_kind = "floating"
        else:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The ncu_report worker returned an unsupported numeric value kind.",
            )
        name = metric.get("name")
        unit = metric.get("unit")
        if not isinstance(name, str) or not isinstance(unit, str):
            raise DomainError(ErrorCode.ARTIFACT_PARSE_FAILED, "Invalid metric identity.")
        dimensions = {
            "action_index": str(metric.get("action_index")),
            "action_name": str(metric.get("action_name")),
            "range_index": str(metric.get("range_index")),
            "report_provider": "nsight.compute",
            "provider_value_kind": provider_value_kind,
        }
        return {
            "measurement_id": digest_model(
                {"artifact_id": artifact_id, "metric_index": index, "metric": metric}
            ),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": name,
            "value_int": value_int,
            "value_float": value_float,
            "value_uint": value_uint,
            "value_kind": normalized_value_kind,
            "unit": unit or "unknown",
            "aggregation": "reported",
            "scope": "device",
            "trial_id": None,
            "worker_id": None,
            "worker_run_index": None,
            "value_index": index,
            "loop_count": None,
            "is_warmup": False,
            "block_id": None,
            "variant_id": None,
            "order_in_block": None,
            "phase": None,
            "dimensions": dimensions,
            "evidence_level": "observed",
        }

    @staticmethod
    def _observation_row(
        run_id: str,
        artifact_id: str,
        index: int,
        observation: dict[str, Any],
    ) -> dict[str, object]:
        kind = observation.get("kind")
        name = observation.get("name")
        if not isinstance(kind, str) or not isinstance(name, str):
            raise DomainError(ErrorCode.ARTIFACT_PARSE_FAILED, "Invalid profile observation.")
        return {
            "observation_id": digest_model(
                {"artifact_id": artifact_id, "observation_index": index, "value": observation}
            ),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "kind": kind,
            "name": name,
            "value_json": json.dumps(
                observation.get("value"), allow_nan=False, separators=(",", ":"), sort_keys=True
            ),
            "file": None,
            "line_from": None,
            "line_to": None,
            "context": "extractor_provenance" if kind == "profile.extraction" else None,
            "evidence_level": "derived" if kind == "profile.extraction" else "observed",
        }
