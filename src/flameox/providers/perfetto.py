from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flameox.canonical import sha256_id
from flameox.command_binding import ExecutableResolver
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.workers.harness import IsolatedWorkerHarness
from flameox.workers.perfetto_contract import (
    PERFETTO_WORKER,
    PerfettoExtractRequest,
    PerfettoExtractResult,
    PerfettoWindowRequest,
    PerfettoWindowResult,
)


class PerfettoProvider:
    """Bounded Perfetto analysis over one explicit native trace path."""

    def __init__(self, harness: IsolatedWorkerHarness) -> None:
        self.harness = harness

    def analyze(
        self,
        capability_id: str,
        path: Path,
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis:
        binary = self._binary()
        version = self._identity(binary)
        if capability_id == "trace.window":
            start_ns = arguments.get("start_ns")
            end_ns = arguments.get("end_ns")
            if not isinstance(start_ns, int) or not isinstance(end_ns, int):
                raise ProviderFailure("INVALID_INPUT", "trace.window requires window arguments")
            response = self.harness.run_typed_sync(
                PERFETTO_WORKER,
                PerfettoWindowRequest(
                    operation="window",
                    artifact_path=str(path),
                    binary_path=str(binary),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    limit=max_rows,
                ),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            if not isinstance(response, PerfettoWindowResult):
                raise ProviderFailure("DECODE_FAILURE", "Perfetto returned another operation")
            rows = [row.model_dump(mode="json") for row in response.rows[:max_rows]]
            return ProviderAnalysis(
                provider_id="perfetto",
                provider_version=version,
                blocks=[
                    {"type": "metrics", "values": {"matching_slice_count": response.total}},
                    {"type": "table", "rows": rows},
                ],
                rows_observed=response.total,
                complete=len(rows) >= response.total,
                limitations=["The window includes slices overlapping the requested interval."],
            )
        response = self.harness.run_typed_sync(
            PERFETTO_WORKER,
            PerfettoExtractRequest(
                operation="extract",
                artifact_path=str(path),
                binary_path=str(binary),
                max_rows=max_rows,
                projection=(
                    "call_graph"
                    if capability_id == "trace.call_graph"
                    else "pytorch"
                    if capability_id == "trace.pytorch"
                    else "slices"
                ),
            ),
            timeout_seconds=timeout_seconds,
            maximum_rss_bytes=maximum_rss_bytes,
            maximum_writable_growth_bytes=maximum_output_bytes,
        )
        if not isinstance(response, PerfettoExtractResult):
            raise ProviderFailure("DECODE_FAILURE", "Perfetto returned another operation")
        if capability_id == "trace.call_graph":
            rows = [row.model_dump(mode="json") for row in response.call_graph_rows]
            slice_count = 0
        else:
            slices = [row.model_dump(mode="json") for row in response.rows]
            rows = self._project(capability_id, slices)
            slice_count = len(slices)
        metrics: dict[str, Any] = {"slice_count": slice_count}
        limitations = [
            "Slice duration is inclusive and nested slices can overlap.",
            "Trace nesting does not by itself prove causal dependence.",
        ]
        if capability_id == "trace.pytorch":
            metrics = {
                "source_slice_count": slice_count,
                "pytorch_event_count": len(rows),
            }
            if not rows:
                limitations.append("no_pytorch_activity_observed")
        return ProviderAnalysis(
            provider_id="perfetto",
            provider_version=version,
            blocks=[
                {"type": "metrics", "values": metrics},
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=(
                response.projected_total
                if capability_id == "trace.call_graph" and response.projected_total is not None
                else len(rows) + int(response.truncated)
            ),
            complete=not response.truncated and len(rows) <= max_rows,
            limitations=limitations,
        )

    @staticmethod
    def _project(capability_id: str, slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if capability_id == "trace.call_graph":
            by_id = {int(row["id"]): row for row in slices}
            edges: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
            for row in slices:
                parent_id = row.get("parent_id")
                if parent_id is None or int(parent_id) not in by_id:
                    continue
                parent = by_id[int(parent_id)]
                aggregate = edges[(str(parent["name"]), str(row["name"]))]
                aggregate[0] += 1
                aggregate[1] += int(row["dur"])
            return [
                {
                    "parent": parent,
                    "child": child,
                    "sample_count": values[0],
                    "inclusive_duration_ns": values[1],
                }
                for (parent, child), values in sorted(
                    edges.items(), key=lambda item: (-item[1][1], item[0])
                )
            ]
        if capability_id == "trace.pytorch":
            rows = []
            for row in slices:
                name = str(row["name"])
                category = str(row.get("category") or "").casefold()
                if not (
                    "pytorch" in category
                    or "torch" in category
                    or name.startswith(("aten::", "autograd::", "torch::", "ProfilerStep#"))
                    or row.get("input_shapes") is not None
                    or row.get("allocation_bytes") is not None
                ):
                    continue
                event_kind = (
                    "operator"
                    if name.startswith(("aten::", "autograd::", "torch::"))
                    else "profiler_step"
                    if name.startswith("ProfilerStep#")
                    else "runtime"
                )
                rows.append({"event_kind": event_kind, **row})
            return rows
        return slices

    @staticmethod
    def _binary() -> Path:
        resolver = ExecutableResolver()
        configured = os.environ.get("FLAMEOX_TRACE_PROCESSOR")
        binding = resolver.resolve_host_tool(configured) if configured else None
        binding = binding or resolver.resolve_host_tool("trace_processor_shell")
        binding = binding or resolver.resolve_host_tool("trace_processor")
        if binding is None:
            raise ProviderFailure(
                "UNAVAILABLE_CAPABILITY",
                "A local Perfetto Trace Processor executable is required.",
                details={"missing_executable": "trace_processor_shell"},
            )
        return binding.invocation_path

    @staticmethod
    def _identity(path: Path) -> str:
        with path.open("rb") as stream:
            return sha256_id(hashlib.file_digest(stream, "sha256").hexdigest())
