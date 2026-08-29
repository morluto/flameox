from __future__ import annotations

import heapq
import json
import re
from typing import cast

from pydantic import JsonValue

from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    DomainError,
    ErrorCode,
    digest_model,
    missing_artifact_input,
)
from flameox.evidence import GenerationPublisher
from flameox.execution import SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace
from flameox.workers.nsight_systems_contract import (
    NSIGHT_SYSTEMS_WORKER,
    NsightSystemsWorkerRequest,
    NsightSystemsWorkerResult,
)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str | bytes | bytearray):
        raise ValueError(f"expected an integer-compatible event value, got {type(value).__name__}")
    return int(value)


def _version_release(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    match = re.search(r"(?<!\d)(\d+(?:\.\d+){1,3})(?!\d)", value)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _range_assignments(
    indexed_events: list[tuple[int, dict[str, object]]],
    ranges: list[dict[str, object]],
) -> dict[int, tuple[str, int]]:
    assignments: dict[int, tuple[str, int]] = {}
    ordered_ranges = sorted(ranges, key=lambda item: _integer(item["start_ns"]))
    active: list[tuple[int, int, str]] = []
    cursor = 0
    for index, event in sorted(
        indexed_events,
        key=lambda item: _integer(item[1]["start_ns"]),
    ):
        start = _integer(event["start_ns"])
        end = start + _integer(event["duration_ns"])
        while (
            cursor < len(ordered_ranges) and _integer(ordered_ranges[cursor]["start_ns"]) <= start
        ):
            selected = ordered_ranges[cursor]
            duration = _integer(selected["duration_ns"])
            heapq.heappush(
                active,
                (
                    duration,
                    _integer(selected["start_ns"]) + duration,
                    str(selected["name"]),
                ),
            )
            cursor += 1
        while active and active[0][1] < end:
            heapq.heappop(active)
        if active:
            duration, _, name = active[0]
            assignments[index] = (name, duration)
    return assignments


def _nvtx_phase_assignments(
    events: list[dict[str, object]],
) -> dict[int, tuple[str, str]]:
    def lane(event: dict[str, object]) -> str | None:
        if event.get("thread") is not None:
            return f"thread:{event['thread']}"
        if event.get("process") is not None:
            return f"process:{event['process']}"
        return None

    ranges_by_lane: dict[str | None, list[dict[str, object]]] = {}
    events_by_lane: dict[str | None, list[tuple[int, dict[str, object]]]] = {}
    lane_by_correlation = {
        str(event["correlation_id"]): event_lane
        for event in events
        if (event_lane := lane(event)) is not None
        if event.get("category") in {"cuda_runtime", "cuda_driver"}
        and event.get("correlation_id") is not None
    }
    for index, event in enumerate(events):
        event_lane = lane(event)
        if event_lane is None and event.get("correlation_id") is not None:
            event_lane = lane_by_correlation.get(str(event["correlation_id"]))
        if event.get("category") == "nvtx" and _integer(event.get("duration_ns", 0)) > 0:
            ranges_by_lane.setdefault(event_lane, []).append(event)
        elif event.get("category") != "nvtx":
            events_by_lane.setdefault(event_lane, []).append((index, event))
    global_ranges = ranges_by_lane.get(None, [])
    assignments: dict[int, tuple[str, str]] = {}
    for event_lane, indexed_events in events_by_lane.items():
        candidates = _range_assignments(indexed_events, global_ranges)
        if event_lane is not None:
            specific = _range_assignments(
                indexed_events,
                ranges_by_lane.get(event_lane, []),
            )
            for index, candidate in specific.items():
                if index not in candidates or candidate[1] < candidates[index][1]:
                    candidates[index] = candidate
        event_by_index = dict(indexed_events)
        for index, (name, _) in candidates.items():
            evidence = "derived_from_containing_nvtx_range"
            if lane(event_by_index[index]) is None:
                evidence = "derived_from_correlation_and_containing_nvtx_range"
            assignments[index] = (name, evidence)
    return assignments


class NsightSystemsExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    asserted_producer_version: str | None
    observed_product_name: str | None
    observed_product_version: str | None
    export_schema_version: str | None
    export_schema_checksum: str | None
    export_settings: dict[str, JsonValue]
    export_format: str
    compatibility_family: str
    query_version: str
    schema_fingerprint: str
    observed_tables: tuple[str, ...]
    event_count: int
    runtime_event_count: int
    driver_event_count: int
    kernel_event_count: int
    nvtx_event_count: int
    memory_copy_event_count: int
    memory_set_event_count: int
    coverage: dict[str, bool]
    corpus_commit_id: str
    limitations: tuple[str, ...]


class NsightSystemsExtractor:
    """Import one supported official Nsight Systems SQLite export."""

    name = "nsight.systems.sqlite"
    version = "1"
    query_version = "flameox.nsight-systems.cuda-events.v1"
    compatibility_family = "nsight-systems.sqlite.cuda.v1"

    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()
        self.publisher = GenerationPublisher(workspace)

    async def extract(self, run_id: str) -> NsightSystemsExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registrations = [
            item for item in run.artifacts if item.kind is ArtifactKind.EXECUTION_TRACE
        ]
        if not registrations:
            raise missing_artifact_input(
                run_id=run_id,
                requirement="Nsight Systems execution-trace",
                artifact_kinds=(ArtifactKind.EXECUTION_TRACE.value,),
                capture_adapters=(),
                import_producers=("auto",),
            )
        registration = self._structured_registration(registrations, run_id=run_id)
        if registration.producer not in {"nsight.systems", "nsys"}:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The execution trace is not registered as an Nsight Systems export.",
                run_id=run_id,
                details={"registered_producer": registration.producer},
                remediation=(
                    "Import an official `nsys export --type sqlite` output with "
                    "producer='nsight.systems'.",
                ),
            )
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        try:
            with artifact.payload_path.open("rb") as stream:
                header = stream.read(16)
        except OSError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Nsight Systems export cannot be read.",
            ) from exc
        if header != b"SQLite format 3\x00":
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Only official Nsight Systems SQLite structured exports are supported.",
                remediation=(
                    "Use `nsys export --type sqlite --output <path> <report>.nsys-rep`, then "
                    "import the SQLite file; Flameox does not parse .nsys-rep directly.",
                ),
            )

        maximum = self.workspace.config.storage.max_rows_per_generation
        if maximum < 7:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Nsight Systems normalization requires room for provenance and one event table.",
            )
        max_rows_per_table = (maximum - 1) // 6
        response = await self._run_worker(
            NsightSystemsWorkerRequest(
                artifact_path=str(artifact.payload_path),
                max_rows_per_table=max_rows_per_table,
            )
        )
        if response.product_name not in {None, "NVIDIA Nsight Systems"}:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The structured export identifies another product.",
                details={"observed_product_name": response.product_name},
            )
        if (
            response.export_schema_version is not None
            and re.fullmatch(r"3\.\d+\.\d+", response.export_schema_version) is None
        ):
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The Nsight Systems export schema major is unsupported.",
                details={"observed_export_schema_version": response.export_schema_version},
                remediation=("Re-export with a supported Nsight Systems release.",),
            )
        asserted_release = _version_release(registration.producer_version)
        observed_release = _version_release(response.product_version)
        releases_match = (
            asserted_release is None
            or observed_release is None
            or asserted_release
            == observed_release[: len(asserted_release)]
        )
        if not releases_match:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The asserted Nsight Systems version conflicts with the export metadata.",
                details={
                    "asserted_producer_version": registration.producer_version,
                    "observed_product_version": response.product_version,
                },
                remediation=(
                    "Import the export without a producer-version claim or provide the observed "
                    "Nsight Systems product version.",
                ),
            )
        raw_events = response.events
        raw_coverage = response.coverage
        raw_tables = response.tables
        if len(raw_events) > maximum:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Nsight Systems events exceed the workspace generation row limit.",
            )
        events = cast(list[dict[str, object]], [dict(event) for event in raw_events])

        phase_assignments = _nvtx_phase_assignments(events)
        rows: list[dict[str, object]] = []
        for index, event in enumerate(events):
            phase: str | None = None
            phase_evidence: str | None = None
            if event.get("category") == "nvtx":
                phase = str(event["name"])
                phase_evidence = "observed"
            elif index in phase_assignments:
                phase, phase_evidence = phase_assignments[index]
            values = {
                **event,
                "phase": phase,
                "phase_evidence": phase_evidence,
                "artifact_role": registration.role,
                "clock_domain": "nsight_systems_export",
                "timestamp_unit": "ns",
            }
            rows.append(
                {
                    "observation_id": digest_model(
                        {
                            "artifact_id": registration.artifact_id,
                            "event_index": index,
                            "kind": "trace.event",
                            "run_id": run_id,
                        }
                    ),
                    "run_id": run_id,
                    "artifact_id": registration.artifact_id,
                    "kind": "trace.event",
                    "name": str(event["name"]),
                    "value_json": json.dumps(
                        values,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "file": None,
                    "line_from": None,
                    "line_to": None,
                    "context": phase,
                    "evidence_level": "observed",
                }
            )

        coverage = {str(key): bool(value) for key, value in raw_coverage.items()}
        schema_fingerprint = response.schema_fingerprint
        rows.append(
            {
                "observation_id": digest_model(
                    {
                        "artifact_id": registration.artifact_id,
                        "kind": "trace.extraction",
                        "query_version": self.query_version,
                        "run_id": run_id,
                        "schema_fingerprint": schema_fingerprint,
                    }
                ),
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "kind": "trace.extraction",
                "name": self.query_version,
                "value_json": json.dumps(
                    {
                        "compatibility_family": self.compatibility_family,
                        "coverage": coverage,
                        "asserted_producer_version": registration.producer_version,
                        "observed_product_name": response.product_name,
                        "observed_product_version": response.product_version,
                        "export_schema_version": response.export_schema_version,
                        "export_schema_checksum": response.export_schema_checksum,
                        "export_settings": response.export_settings,
                        "truncated_tables": list(response.truncated_tables),
                        "query_version": self.query_version,
                        "schema_fingerprint": schema_fingerprint,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "file": None,
                "line_from": None,
                "line_to": None,
                "context": "extractor_provenance",
                "evidence_level": "observed",
            }
        )
        truncated_tables = response.truncated_tables
        limitations = [
            "The SQLite export is authoritative; normalized rows cover only the declared "
            "compatibility schema.",
            "NVTX containment is a derived temporal association and does not prove causality.",
            "Timestamps use the Nsight Systems export clock; cross-profiler clock alignment is "
            "unknown.",
            *response.limitations,
        ]
        if response.product_version is None or response.export_schema_version is None:
            limitations.append(
                "Provider-declared product or export-schema identity is unavailable; "
                "compatibility is limited to the observed table fingerprint."
            )
        for capability, present in coverage.items():
            if not present:
                limitations.append(
                    f"Optional Nsight Systems evidence is unavailable: {capability}."
                )
        if truncated_tables:
            limitations.append(
                "Structured extraction reached the row budget for: "
                + ", ".join(str(item) for item in truncated_tables)
                + "."
            )
        published = self.publisher.publish_rows_idempotent(
            {"observations": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "compatibility_family": self.compatibility_family,
                "max_rows_per_table": max_rows_per_table,
                "asserted_producer_version": registration.producer_version,
                "observed_product_version": response.product_version,
                "export_schema_version": response.export_schema_version,
                "export_schema_checksum": response.export_schema_checksum,
                "query_version": self.query_version,
                "schema_fingerprint": schema_fingerprint,
            },
        )
        counts = {
            category: sum(event.get("category") == category for event in events)
            for category in (
                "cuda_runtime",
                "cuda_driver",
                "kernel",
                "nvtx",
                "memcpy",
                "memset",
            )
        }
        return NsightSystemsExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            asserted_producer_version=registration.producer_version,
            observed_product_name=response.product_name,
            observed_product_version=response.product_version,
            export_schema_version=response.export_schema_version,
            export_schema_checksum=response.export_schema_checksum,
            export_settings=dict(response.export_settings),
            export_format="sqlite",
            compatibility_family=self.compatibility_family,
            query_version=self.query_version,
            schema_fingerprint=schema_fingerprint,
            observed_tables=tuple(raw_tables),
            event_count=len(events),
            runtime_event_count=counts["cuda_runtime"],
            driver_event_count=counts["cuda_driver"],
            kernel_event_count=counts["kernel"],
            nvtx_event_count=counts["nvtx"],
            memory_copy_event_count=counts["memcpy"],
            memory_set_event_count=counts["memset"],
            coverage=coverage,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _structured_registration(
        registrations: list[ArtifactRegistration], *, run_id: str
    ) -> ArtifactRegistration:
        structured = [
            item
            for item in registrations
            if item.role == "sqlite_export"
        ]
        if len(structured) == 1:
            return structured[0]
        if (
            len(registrations) == 1
            and registrations[0].display_name.endswith(".sqlite")
            and not registrations[0].role.startswith("partial_")
        ):
            return registrations[0]
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            "The run must contain exactly one Nsight Systems structured export.",
            run_id=run_id,
        )

    async def _run_worker(
        self,
        request: NsightSystemsWorkerRequest,
    ) -> NsightSystemsWorkerResult:
        return await IsolatedWorkerHarness(self.workspace, broker=self.broker).run_typed(
            NSIGHT_SYSTEMS_WORKER,
            request,
        )
