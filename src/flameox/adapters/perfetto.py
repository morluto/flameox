from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import Field, JsonValue

from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CursorCodec,
    DomainError,
    ErrorCode,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.evidence_status import EvidenceAvailability, available_availability, empty_availability
from flameox.execution import SubprocessBroker
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import ArtifactStore, RunStore, Workspace

type _AggregateKey = tuple[str, str | None, str | None]


class PerfettoExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    trace_processor_path: str
    trace_processor_sha256: str
    query_version: str
    slice_count: int
    frame_count: int
    call_edge_count: int
    representative_stack_count: int
    trace_event_count: int
    truncated: bool = False
    corpus_commit_id: str
    limitations: tuple[str, ...]


class TraceEvent(ContractModel):
    slice_id: int
    parent_id: int | None
    name: str
    category: str | None
    start_ns: int
    duration_ns: int
    track_id: int


class TraceWindowResult(CursorPageContract):
    page_items_field = "events"

    schema_version: int = 1
    artifact_id: str
    start_ns: int
    end_ns: int
    events: tuple[TraceEvent, ...]
    total: int
    coverage: float
    trace_processor_path: str
    limitations: tuple[str, ...] = (
        "The window includes slices that overlap the requested interval.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class PerfettoExtractor:
    """Curated Trace Processor queries; callers cannot supply SQL."""

    name = "perfetto"
    version = "1"
    query_version = "flameox.perfetto.trace-events.v1"

    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()
        self.publisher = GenerationPublisher(workspace)

    async def extract(
        self,
        run_id: str,
        *,
        artifact_id: str | None = None,
    ) -> PerfettoExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registration = self._trace_registration(run.artifacts, artifact_id, run_id=run_id)

        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)

        binary = self._trace_processor_path()
        with binary.open("rb") as stream:
            trace_processor_sha256 = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        maximum = self.workspace.config.storage.max_rows_per_generation
        if maximum < 7:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Perfetto normalization requires room for provenance and one bounded slice.",
            )
        max_slices = (maximum - 1) // 6
        response = await self._run_worker(
            {
                "operation": "extract",
                "artifact_path": str(artifact.payload_path),
                "binary_path": str(binary),
                "max_rows": max_slices,
            }
        )
        rows = response.get("rows")
        if not isinstance(rows, list):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Perfetto worker returned no slice rows.",
                run_id=run_id,
            )
        normalized_phase_by_id = self._normalized_phases(rows)
        frame_by_id: dict[str, dict[str, Any]] = {}
        aggregates: dict[_AggregateKey, tuple[int, int]] = {}
        phases_by_aggregate: dict[_AggregateKey, set[str]] = {}
        operator_observation_rows: list[dict[str, Any]] = []
        trace_event_observation_rows: list[dict[str, Any]] = []
        torch_source = "torch" in (run.collector or "").lower() or (
            registration.producer is not None and "torch" in registration.producer.lower()
        )
        events: dict[
            int,
            tuple[
                int | None,
                str,
                int,
                int,
                int,
                str | None,
                str | None,
            ],
        ] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Perfetto worker returned an invalid slice row.",
                )
            name = str(row["name"])
            filename = str(row["filename"]) if row["filename"] is not None else None
            line = int(row["line"]) if row["line"] is not None else None
            category = str(row["category"]) if row["category"] is not None else None
            thread_name = str(row["thread_name"]) if row.get("thread_name") is not None else None
            process_name = str(row["process_name"]) if row.get("process_name") is not None else None
            frame_id = digest_model(
                {
                    "artifact_id": registration.artifact_id,
                    "function": name,
                    "category": category,
                    "file": filename,
                    "line": line,
                    "symbolization": "partial",
                }
            )
            frame_by_id.setdefault(
                frame_id,
                {
                    "frame_id": frame_id,
                    "language": ("Python" if run.collector == "py-spy" else None),
                    "function": name,
                    "module": category,
                    "file": filename,
                    "line": line,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": run.source_state_id,
                    "artifact_id": registration.artifact_id,
                    "inlined": None,
                    "symbolization": "partial",
                },
            )
            duration = int(row["dur"])
            aggregate_key = (frame_id, thread_name, process_name)
            count, inclusive = aggregates.get(aggregate_key, (0, 0))
            aggregates[aggregate_key] = (count + 1, inclusive + duration)
            phase = normalized_phase_by_id.get(int(row["id"]))
            if phase is not None:
                phases_by_aggregate.setdefault(aggregate_key, set()).add(phase)
            if torch_source:
                values = {
                    "frame_id": frame_id,
                    "category": category,
                    "duration_ns": duration,
                    "input_shapes": (
                        str(row["input_shapes"]) if row.get("input_shapes") is not None else None
                    ),
                    "allocation_bytes": (
                        int(row["allocation_bytes"])
                        if row.get("allocation_bytes") is not None
                        else None
                    ),
                    "phase": phase,
                }
                operator_observation_rows.append(
                    {
                        "observation_id": digest_model(
                            {
                                "artifact_id": registration.artifact_id,
                                "slice_id": int(row["id"]),
                                "kind": "pytorch.operator",
                            }
                        ),
                        "run_id": run_id,
                        "artifact_id": registration.artifact_id,
                        "kind": "pytorch.operator",
                        "name": name,
                        "value_json": json.dumps(
                            values,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "file": filename,
                        "line_from": line,
                        "line_to": line,
                        "context": phase,
                        "evidence_level": "observed",
                    }
                )
            trace_event_values = {
                "slice_id": int(row["id"]),
                "parent_id": int(row["parent_id"]) if row["parent_id"] is not None else None,
                "category": category,
                "start_ns": int(row["ts"]),
                "duration_ns": duration,
                "track_id": int(row["track_id"]),
                "thread_name": thread_name,
                "process_name": process_name,
                "phase": phase,
                "correlation_id": (
                    str(row["correlation_id"]) if row.get("correlation_id") is not None else None
                ),
                "device": str(row["device"]) if row.get("device") is not None else None,
                "stream": str(row["stream"]) if row.get("stream") is not None else None,
                "artifact_role": registration.role,
            }
            trace_event_observation_rows.append(
                {
                    "observation_id": digest_model(
                        {
                            "artifact_id": registration.artifact_id,
                            "slice_id": int(row["id"]),
                            "kind": "trace.event",
                        }
                    ),
                    "run_id": run_id,
                    "artifact_id": registration.artifact_id,
                    "kind": "trace.event",
                    "name": name,
                    "value_json": json.dumps(
                        trace_event_values,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "file": filename,
                    "line_from": line,
                    "line_to": line,
                    "context": phase,
                    "evidence_level": "observed",
                }
            )
            events[int(row["id"])] = (
                int(row["parent_id"]) if row["parent_id"] is not None else None,
                frame_id,
                int(row["ts"]),
                duration,
                int(row["track_id"]),
                thread_name,
                process_name,
            )
        child_duration_by_event: dict[int, int] = {}
        for parent_id, _, _, duration, _, _, _ in events.values():
            if parent_id is not None and parent_id in events:
                child_duration_by_event[parent_id] = (
                    child_duration_by_event.get(parent_id, 0) + duration
                )
        self_duration_by_aggregate: dict[_AggregateKey, int] = {}
        for event_id, (
            _,
            frame_id,
            _,
            duration,
            _,
            thread_name,
            process_name,
        ) in events.items():
            aggregate_key = (frame_id, thread_name, process_name)
            self_duration_by_aggregate[aggregate_key] = self_duration_by_aggregate.get(
                aggregate_key, 0
            ) + max(0, duration - child_duration_by_event.get(event_id, 0))
        measurement_rows = [
            {
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "frame_id": frame_id,
                "metric": "trace.slice_duration",
                "self_value": self_duration_by_aggregate[(frame_id, thread_name, process_name)],
                "inclusive_value": duration,
                "unit": "ns",
                "sample_count": count,
                "thread_name": thread_name,
                "process_name": process_name,
                "phase": (
                    next(iter(phases_by_aggregate[(frame_id, thread_name, process_name)]))
                    if len(
                        phases_by_aggregate.get(
                            (frame_id, thread_name, process_name),
                            (),
                        )
                    )
                    == 1
                    else None
                ),
            }
            for (
                frame_id,
                thread_name,
                process_name,
            ), (count, duration) in aggregates.items()
        ]
        edge_aggregates: dict[tuple[str, str], tuple[int, int]] = {}
        parent_ids: set[int] = set()
        for parent_id, child_frame_id, _, duration, _, _, _ in events.values():
            if parent_id is None or parent_id not in events:
                continue
            parent_ids.add(parent_id)
            parent_frame_id = events[parent_id][1]
            edge_key = (parent_frame_id, child_frame_id)
            count, total = edge_aggregates.get(edge_key, (0, 0))
            edge_aggregates[edge_key] = (count + 1, total + duration)
        edge_rows = [
            {
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "parent_frame_id": parent_frame_id,
                "child_frame_id": child_frame_id,
                "sample_count": count,
                "duration_ns": duration,
            }
            for (parent_frame_id, child_frame_id), (
                count,
                duration,
            ) in edge_aggregates.items()
        ]
        stack_rows: list[dict[str, Any]] = []
        for event_id, (
            _,
            leaf_frame_id,
            start,
            duration,
            track_id,
            _,
            _,
        ) in events.items():
            if event_id in parent_ids:
                continue
            frame_ids: list[str] = []
            cursor: int | None = event_id
            seen: set[int] = set()
            while cursor is not None and cursor in events and cursor not in seen:
                seen.add(cursor)
                parent_id, frame_id, _, _, _, _, _ = events[cursor]
                frame_ids.append(frame_id)
                cursor = parent_id
            frame_ids.reverse()
            stack_id = digest_model(
                {
                    "artifact_id": registration.artifact_id,
                    "event_id": event_id,
                    "frame_ids": frame_ids,
                    "start_ns": start,
                }
            )
            stack_rows.append(
                {
                    "stack_id": stack_id,
                    "run_id": run_id,
                    "artifact_id": registration.artifact_id,
                    "frame_ids": frame_ids,
                    "leaf_frame_id": leaf_frame_id,
                    "start_ns": start,
                    "duration_ns": duration,
                    "track_id": track_id,
                }
            )
        evidence_rows: dict[str, list[dict[str, Any]]] = {
            "frames": list(frame_by_id.values()),
            "frame_measurements": measurement_rows,
            "call_edges": edge_rows,
            "stacks": stack_rows,
        }
        extraction_observation = {
            "observation_id": digest_model(
                {
                    "artifact_id": registration.artifact_id,
                    "kind": "trace.extraction",
                    "query_version": self.query_version,
                    "trace_processor_sha256": trace_processor_sha256,
                }
            ),
            "run_id": run_id,
            "artifact_id": registration.artifact_id,
            "kind": "trace.extraction",
            "name": self.query_version,
            "value_json": json.dumps(
                {
                    "query_version": self.query_version,
                    "trace_processor_sha256": trace_processor_sha256,
                    "normalized_slice_count": len(events),
                    "truncated": response.get("truncated") is True,
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
        if operator_observation_rows or trace_event_observation_rows:
            evidence_rows["observations"] = [
                extraction_observation,
                *operator_observation_rows,
                *trace_event_observation_rows,
            ]
        else:
            evidence_rows["observations"] = [extraction_observation]
        published = self.publisher.publish_rows_idempotent(
            evidence_rows,
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "query_version": self.query_version,
                "trace_processor_sha256": trace_processor_sha256,
                "max_slices": max_slices,
            },
        )
        return PerfettoExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            trace_processor_path=str(binary),
            trace_processor_sha256=trace_processor_sha256,
            query_version=self.query_version,
            slice_count=len(events),
            frame_count=len(frame_by_id),
            call_edge_count=len(edge_rows),
            representative_stack_count=len(stack_rows),
            trace_event_count=len(trace_event_observation_rows),
            truncated=response.get("truncated") is True,
            corpus_commit_id=published.commit.commit_id,
            limitations=(
                "Slice duration is inclusive and nested slices can overlap.",
                "Parent-child edges reflect trace nesting, not causal dependence.",
                "Complete temporal detail remains in the native trace.",
                *(
                    (
                        "Normalized extraction reached the generation row budget; later slices "
                        "remain only in the native trace.",
                    )
                    if response.get("truncated") is True
                    else ()
                ),
            ),
        )

    @staticmethod
    def _trace_registration(
        artifacts: tuple[ArtifactRegistration, ...],
        artifact_id: str | None,
        *,
        run_id: str,
    ) -> ArtifactRegistration:
        registrations = [
            item
            for item in artifacts
            if item.kind in {ArtifactKind.EXECUTION_TRACE, ArtifactKind.SAMPLE_PROFILE}
        ]
        available_artifact_ids = [item.artifact_id for item in registrations]
        if artifact_id is not None:
            registrations = [item for item in registrations if item.artifact_id == artifact_id]
        if len(registrations) == 1:
            return registrations[0]
        if artifact_id is not None:
            message = "The selected Perfetto-compatible trace artifact was not found on the run."
        elif registrations:
            message = "The run contains multiple Perfetto-compatible traces; select one."
        else:
            message = "The run contains no Perfetto-compatible trace."
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            message,
            run_id=run_id,
            details={
                "artifact_ids": available_artifact_ids,
                "next_tool": "extract_perfetto",
            },
            remediation=("Pass one execution-trace artifact_id from the run artifact list.",),
        )

    @staticmethod
    def _normalized_phases(rows: list[object]) -> dict[int, str | None]:
        row_by_id = {
            int(row["id"]): row
            for row in rows
            if isinstance(row, dict) and row.get("id") is not None
        }
        normalized: dict[int, str | None] = {}
        for event_id, row in row_by_id.items():
            explicit = row.get("phase")
            selected: str | None = str(explicit) if explicit is not None else None
            cursor: int | None = event_id
            seen: set[int] = set()
            while selected is None and cursor is not None and cursor not in seen:
                seen.add(cursor)
                ancestor = row_by_id.get(cursor)
                if ancestor is None:
                    break
                ancestor_name = str(ancestor.get("name") or "")
                if ancestor_name.startswith("flameox.phase:"):
                    candidate = ancestor_name.removeprefix("flameox.phase:")
                    selected = candidate or None
                    break
                parent_id = ancestor.get("parent_id")
                cursor = int(parent_id) if parent_id is not None else None
            normalized[event_id] = selected
        return normalized

    async def trace_window(
        self,
        artifact_id: str,
        *,
        start_ns: int,
        end_ns: int,
        limit: int = 100,
        cursor: str | None = None,
    ) -> TraceWindowResult:
        if start_ns < 0 or end_ns <= start_ns:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Trace window requires 0 <= start_ns < end_ns.",
            )
        maximum = self.workspace.config.analysis.max_row_limit
        if limit < 1 or limit > maximum:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Limit must be between 1 and {maximum}.",
            )
        artifact = ArtifactStore(self.workspace).get(artifact_id)
        binary = self._trace_processor_path()
        scope_digest = digest_model(
            {
                "artifact_id": artifact_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
        )
        after_ts: int | None = None
        after_id: int | None = None
        if cursor is not None:
            position = CursorCodec.decode(
                cursor,
                namespace="trace_window",
                snapshot_id=artifact_id,
                scope_digest=scope_digest,
            )
            if (
                len(position) != 2
                or not isinstance(position[0], int)
                or not isinstance(position[1], int)
            ):
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
            timestamp_value, id_value = position
            assert isinstance(timestamp_value, int)
            assert isinstance(id_value, int)
            after_ts, after_id = timestamp_value, id_value
        response = await self._run_worker(
            {
                "operation": "window",
                "artifact_path": str(artifact.payload_path),
                "binary_path": str(binary),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "limit": limit,
                "after_ts": after_ts,
                "after_id": after_id,
            }
        )
        total = int(response.get("total", 0))
        rows = response.get("rows")
        if not isinstance(rows, list):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Perfetto worker returned no trace-window rows.",
            )
        selected = rows[:limit]
        events = tuple(
            TraceEvent(
                slice_id=int(row["id"]),
                parent_id=(int(row["parent_id"]) if row["parent_id"] is not None else None),
                name=str(row["name"]),
                category=(str(row["category"]) if row["category"] is not None else None),
                start_ns=int(row["ts"]),
                duration_ns=int(row["dur"]),
                track_id=int(row["track_id"]),
            )
            for row in selected
            if isinstance(row, dict)
        )
        has_more = len(rows) > limit
        next_cursor = (
            CursorCodec.encode(
                namespace="trace_window",
                snapshot_id=artifact_id,
                scope_digest=scope_digest,
                position=(events[-1].start_ns, events[-1].slice_id),
            )
            if has_more and events
            else None
        )
        return TraceWindowResult(
            artifact_id=artifact_id,
            start_ns=start_ns,
            end_ns=end_ns,
            events=events,
            total=total,
            coverage=(len(events) / total if total else 0.0),
            next_cursor=next_cursor,
            trace_processor_path=str(binary),
            evidence=(
                empty_availability("no_events_in_window")
                if not events
                else available_availability()
            ),
        )

    async def _run_worker(
        self,
        request: dict[str, JsonValue],
    ) -> dict[str, Any]:
        return await IsolatedWorkerHarness(self.workspace, broker=self.broker).run(
            "flameox.workers.perfetto",
            request,
            name="Perfetto",
            timeout_seconds=120,
            consume=lambda response, _root: response,
        )

    def _trace_processor_path(self) -> Path:
        configured = self.workspace.config.analysis.trace_processor_path
        candidate: str | None
        if configured is not None:
            path = Path(configured)
            candidate = str(path if path.is_absolute() else self.workspace.project_root / path)
        else:
            binding = ExecutableResolver().resolve_host_tool("trace_processor_shell")
            binding = binding or ExecutableResolver().resolve_host_tool("trace_processor")
            candidate = str(binding.invocation_path) if binding is not None else None
        if candidate is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "A local Perfetto Trace Processor binary is required.",
                remediation=(
                    "Install trace_processor_shell on PATH or set "
                    "analysis.trace_processor_path in workspace policy.",
                ),
            )
        path = Path(candidate).absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Trace Processor is not executable: {path}",
            )
        return path
