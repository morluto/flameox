from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    AcceleratorLaunchAnalysisResult,
    AcceleratorLaunchComparison,
    AcceleratorLaunchRegion,
    AcceleratorStreamSummary,
    KernelNameCount,
)
from flameox.catalog import Snapshot
from flameox.domain import DomainError, ErrorCode
from flameox.evidence_scope import resolve_evidence_scope
from flameox.evidence_status import (
    available_availability,
    empty_availability,
    partial_availability,
)


def _trace_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _accelerator_stream_key(value: dict[str, Any]) -> str:
    stream = value.get("stream")
    if stream not in {None, ""}:
        return "/".join(
            (
                f"device:{value.get('device', 'unknown')}",
                f"context:{value.get('context', 'unknown')}",
                f"stream:{stream}",
            )
        )
    track = value.get("track_id")
    return f"track:{track if track not in {None, ''} else 'unknown'}"


def _runtime_track_key(value: dict[str, Any]) -> str:
    for field in ("track_id", "thread", "process", "thread_name"):
        identity = value.get(field)
        if identity not in {None, ""}:
            return f"{field}:{identity}"
    return "runtime:unknown"


def _positive_gaps(events: list[tuple[int, int]]) -> list[int]:
    gaps: list[int] = []
    cursor: int | None = None
    for start, duration in sorted(events):
        if cursor is not None and start > cursor:
            gaps.append(start - cursor)
        cursor = max(cursor if cursor is not None else start, start + duration)
    return gaps


class AcceleratorRecipes(RecipeContext):
    def accelerator_launches(
        self,
        input_id: str,
        *,
        comparison_input_id: str | None = None,
        phase: str | None = None,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> AcceleratorLaunchAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            regions, total, coverage, limitations = self._accelerator_launch_regions(
                snapshot,
                input_id,
                phase=phase,
                limit=bounded,
            )
            comparison_regions: tuple[AcceleratorLaunchRegion, ...] = ()
            comparison_coverage: dict[str, bool] | None = None
            if comparison_input_id is not None:
                (
                    comparison_regions,
                    _,
                    comparison_coverage,
                    comparison_limitations,
                ) = self._accelerator_launch_regions(
                    snapshot,
                    comparison_input_id,
                    phase=phase,
                    limit=bounded,
                )
                limitations = (
                    *limitations,
                    *(f"Comparison input: {item}" for item in comparison_limitations),
                )
        primary_by_region = {item.region: item for item in regions}
        comparison_by_region = {item.region: item for item in comparison_regions}
        comparisons = tuple(
            AcceleratorLaunchComparison(
                region=region,
                direct_launch_count_delta=(
                    comparison_by_region[region].direct_launch_count
                    - primary_by_region[region].direct_launch_count
                ),
                graph_launch_count_delta=(
                    comparison_by_region[region].graph_launch_count
                    - primary_by_region[region].graph_launch_count
                ),
                kernel_count_delta=(
                    comparison_by_region[region].kernel_count
                    - primary_by_region[region].kernel_count
                ),
                kernel_duration_delta_ns=(
                    comparison_by_region[region].kernel_duration_ns
                    - primary_by_region[region].kernel_duration_ns
                ),
                runtime_launch_gap_total_delta_ns=(
                    comparison_by_region[region].runtime_launch_gap_total_ns
                    - primary_by_region[region].runtime_launch_gap_total_ns
                ),
                idle_gap_total_delta_ns=(
                    comparison_by_region[region].idle_gap_total_ns
                    - primary_by_region[region].idle_gap_total_ns
                ),
            )
            for region in sorted(primary_by_region.keys() & comparison_by_region.keys())
        )
        if comparison_input_id is not None:
            limitations = (
                *limitations,
                "Launch-structure deltas are descriptive; they do not prove semantic "
                "equivalence or causality.",
            )
        return AcceleratorLaunchAnalysisResult(
            corpus_commit_id=corpus_commit_id,
            input_id=input_id,
            comparison_input_id=comparison_input_id,
            phase_filter=phase,
            regions=regions,
            comparison_regions=comparison_regions,
            comparisons=comparisons,
            total=total,
            returned=len(regions),
            truncated=total > len(regions),
            coverage=coverage,
            comparison_coverage=comparison_coverage,
            limitations=tuple(dict.fromkeys(limitations)),
            evidence=(
                empty_availability("no_matching_accelerator_launch_events")
                if total == 0
                else (
                    partial_availability("runtime_or_accelerator_tracks_missing")
                    if not (
                        coverage["runtime_launches"]
                        and coverage["accelerator_kernels"]
                        and (
                            comparison_coverage is None
                            or (
                                comparison_coverage["runtime_launches"]
                                and comparison_coverage["accelerator_kernels"]
                            )
                        )
                    )
                    else available_availability()
                )
            ),
        )

    def _accelerator_launch_regions(
        self,
        snapshot: Snapshot,
        input_id: str,
        *,
        phase: str | None,
        limit: int,
    ) -> tuple[
        tuple[AcceleratorLaunchRegion, ...],
        int,
        dict[str, bool],
        tuple[str, ...],
    ]:
        scope = resolve_evidence_scope(snapshot, input_id)
        where, parameters = scope.predicate(
            run_column="run_id",
            artifact_column="artifact_id",
        )
        rows = snapshot.execute(
            "SELECT name, value_json, context FROM observations WHERE ("
            + where
            + ") AND kind = 'trace.event' ORDER BY observation_id",
            parameters,
        ).fetchall()
        if not rows:
            producer_rows = (
                snapshot.execute(
                    "SELECT DISTINCT producer FROM artifact_registrations "
                    "WHERE artifact_id IN (" + ", ".join("?" for _ in scope.artifact_ids) + ")",
                    scope.artifact_ids,
                ).fetchall()
                if scope.artifact_ids
                else ()
            )
            producers = {str(row[0]).casefold() for row in producer_rows if row[0] is not None}
            next_tool = (
                "extract_nsight_systems"
                if producers & {"nsight.systems", "nsys"}
                else "extract_perfetto"
            )
            details: dict[str, object] = {"next_tool": next_tool}
            if scope.run_ids:
                details["run_id"] = scope.run_ids[0]
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Accelerator launch analysis requires current normalized trace-event evidence.",
                details=details,
                remediation=(
                    f"Call {next_tool} for the reported run, then retry "
                    "analyze_accelerator_launches.",
                ),
            )
        events_by_region: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        trace_phase_present = False
        correlation_present = False
        stream_present = False
        for name, value_json, context in rows:
            try:
                value = json.loads(str(value_json))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            phase_value = context or value.get("phase")
            normalized_phase = str(phase_value) if phase_value not in {None, ""} else "<unscoped>"
            trace_phase_present = trace_phase_present or normalized_phase != "<unscoped>"
            if phase is not None and normalized_phase != phase:
                continue
            artifact_role = str(value.get("artifact_role") or "")
            cycle = (
                artifact_role if artifact_role.startswith(("cycle_", "partial_cycle_")) else None
            )
            region = f"{cycle}/{normalized_phase}" if cycle is not None else normalized_phase
            correlation_present = correlation_present or value.get("correlation_id") not in {
                None,
                "",
            }
            stream_present = stream_present or value.get("stream") not in {None, ""}
            events_by_region[region].append((str(name), value))
        summaries = tuple(
            self._summarize_accelerator_region(region, events, limit=limit)
            for region, events in sorted(events_by_region.items())
        )
        runtime_present = any(
            item.direct_launch_count or item.graph_launch_count for item in summaries
        )
        accelerator_present = any(item.kernel_count for item in summaries)
        kernel_count = sum(item.kernel_count for item in summaries)
        correlated_kernel_count = sum(item.correlated_kernel_count for item in summaries)
        matched_correlation_present = correlated_kernel_count > 0
        limitations = [
            "Launch and kernel counts are observed trace events; matching counts or names do "
            "not prove equivalent computation.",
            "A gap is the positive uncovered interval between consecutive observed slices on "
            "the same runtime track or accelerator stream; zero and overlap are not gaps.",
        ]
        if not trace_phase_present:
            limitations.append(
                "The trace contains no phase annotations; events are grouped as <unscoped>."
            )
        elif phase is not None and not summaries:
            limitations.append(f"No normalized trace events matched phase {phase!r}.")
        if not correlation_present:
            limitations.append(
                "Runtime-to-kernel correlation identifiers were unavailable in normalized evidence."
            )
        elif correlated_kernel_count < kernel_count:
            limitations.append(
                f"Correlation identifiers covered {correlated_kernel_count} of "
                f"{kernel_count} selected kernels."
            )
        if not runtime_present:
            limitations.append(
                "No recognized direct or CUDA Graph runtime launch events were found."
            )
        if not accelerator_present:
            limitations.append("No recognized accelerator kernel events were found.")
        return (
            summaries[:limit],
            len(summaries),
            {
                "runtime_launches": runtime_present,
                "accelerator_kernels": accelerator_present,
                "phase_annotations": trace_phase_present,
                "correlation_ids": correlation_present,
                "host_to_device_correlation": matched_correlation_present,
                "stream_identity": stream_present,
            },
            tuple(limitations),
        )

    @staticmethod
    def _summarize_accelerator_region(
        region: str,
        events: list[tuple[str, dict[str, Any]]],
        *,
        limit: int,
    ) -> AcceleratorLaunchRegion:
        direct: list[dict[str, Any]] = []
        graph: list[dict[str, Any]] = []
        kernels: list[tuple[str, dict[str, Any]]] = []
        for name, value in events:
            normalized_name = "".join(
                character for character in name.casefold() if character.isalnum()
            )
            category = str(value.get("category") or "").casefold()
            runtime_api = "runtime" in category or "driver" in category
            if runtime_api:
                if "graphlaunch" in normalized_name:
                    graph.append(value)
                elif any(
                    token in normalized_name
                    for token in (
                        "cudalaunchkernel",
                        "cudalaunchcooperativekernel",
                        "culaunchkernel",
                        "culaunchcooperativekernel",
                        "hiplaunchkernel",
                        "hipmodulelaunchkernel",
                    )
                ):
                    direct.append(value)
            if "kernel" in category and "runtime" not in category:
                kernels.append((name, value))

        kernel_names = Counter(name for name, _ in kernels)
        selected_names = tuple(
            KernelNameCount(name=name, count=count)
            for name, count in sorted(
                kernel_names.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
        )
        kernels_by_stream: dict[str, list[tuple[int, int]]] = defaultdict(list)
        stream_identity: dict[str, dict[str, str | None]] = {}
        for _, value in kernels:
            start = _trace_integer(value.get("start_ns"))
            duration = _trace_integer(value.get("duration_ns"))
            if start is None or duration is None:
                continue
            stream_key = _accelerator_stream_key(value)
            kernels_by_stream[stream_key].append((start, duration))
            stream_identity.setdefault(
                stream_key,
                {
                    field: (str(value[field]) if value.get(field) not in {None, ""} else None)
                    for field in ("device", "context", "stream", "track_id")
                },
            )
        stream_summaries: list[AcceleratorStreamSummary] = []
        for stream_key, stream_events in sorted(kernels_by_stream.items()):
            stream_gaps = _positive_gaps(stream_events)
            identity = stream_identity[stream_key]
            stream_summaries.append(
                AcceleratorStreamSummary(
                    identity=stream_key,
                    device=identity["device"],
                    context=identity["context"],
                    stream=identity["stream"],
                    track_id=identity["track_id"],
                    kernel_count=len(stream_events),
                    kernel_duration_ns=sum(duration for _, duration in stream_events),
                    idle_gap_count=len(stream_gaps),
                    idle_gap_total_ns=sum(stream_gaps),
                    idle_gap_max_ns=max(stream_gaps, default=0),
                )
            )
        gaps = [
            gap
            for stream_events in kernels_by_stream.values()
            for gap in _positive_gaps(stream_events)
        ]
        runtime_by_track: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for value in (*direct, *graph):
            start = _trace_integer(value.get("start_ns"))
            duration = _trace_integer(value.get("duration_ns"))
            if start is None or duration is None:
                continue
            runtime_by_track[_runtime_track_key(value)].append((start, duration))
        runtime_gaps = [
            gap
            for track_events in runtime_by_track.values()
            for gap in _positive_gaps(track_events)
        ]
        runtime_correlation_ids = {
            str(value["correlation_id"])
            for value in (*direct, *graph)
            if value.get("correlation_id") not in {None, ""}
        }
        starts_and_ends = [
            (start, start + duration)
            for _, value in events
            if (start := _trace_integer(value.get("start_ns"))) is not None
            and (duration := _trace_integer(value.get("duration_ns"))) is not None
        ]
        region_start = min((start for start, _ in starts_and_ends), default=0)
        region_end = max((end for _, end in starts_and_ends), default=region_start)
        return AcceleratorLaunchRegion(
            region=region,
            region_start_ns=region_start,
            region_end_ns=region_end,
            region_duration_ns=max(0, region_end - region_start),
            selection_rule="exact normalized phase or cycle/phase grouping",
            direct_launch_count=len(direct),
            direct_launch_duration_ns=sum(
                _trace_integer(item.get("duration_ns")) or 0 for item in direct
            ),
            graph_launch_count=len(graph),
            graph_launch_duration_ns=sum(
                _trace_integer(item.get("duration_ns")) or 0 for item in graph
            ),
            kernel_count=len(kernels),
            kernel_duration_ns=sum(
                _trace_integer(item.get("duration_ns")) or 0 for _, item in kernels
            ),
            kernel_names=selected_names,
            kernel_names_truncated=len(kernel_names) > len(selected_names),
            correlated_kernel_count=sum(
                str(item["correlation_id"]) in runtime_correlation_ids
                for _, item in kernels
                if item.get("correlation_id") not in {None, ""}
            ),
            runtime_launch_gap_count=len(runtime_gaps),
            runtime_launch_gap_total_ns=sum(runtime_gaps),
            runtime_launch_gap_max_ns=max(runtime_gaps, default=0),
            idle_gap_count=len(gaps),
            idle_gap_total_ns=sum(gaps),
            idle_gap_max_ns=max(gaps, default=0),
            stream_count=len(kernels_by_stream),
            streams=tuple(stream_summaries[:limit]),
            streams_truncated=len(stream_summaries) > limit,
        )
