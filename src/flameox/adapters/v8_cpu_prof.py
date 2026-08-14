from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flameox.domain import ArtifactKind, DomainError, ErrorCode, digest_model
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class V8CpuProfExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    node_count: int
    sample_count: int
    frame_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class V8CpuProfExtractor:
    name = "node-cpu-prof"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> V8CpuProfExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registrations = [
            item for item in run.artifacts if item.kind is ArtifactKind.SAMPLE_PROFILE
        ]
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one V8 CPU profile artifact.",
                run_id=run_id,
            )
        registration = registrations[0]
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        try:
            payload = json.loads(artifact.payload_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported V8 CPU profile (.cpuprofile).",
                run_id=run_id,
            ) from exc
        if not isinstance(payload, dict) or "nodes" not in payload:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact does not contain a V8 CPU profile node tree.",
                run_id=run_id,
            )
        nodes = payload.get("nodes")
        samples = payload.get("samples") or []
        if not isinstance(nodes, list) or not isinstance(samples, list):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The V8 CPU profile nodes or samples are malformed.",
                run_id=run_id,
            )
        nodes_by_id: dict[int, dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nodes_by_id[int(node["id"])] = node
        frame_rows: dict[str, dict[str, Any]] = {}
        aggregates: dict[tuple[str, str], dict[str, int]] = {}
        for node in nodes_by_id.values():
            self._aggregate_node(
                node,
                nodes_by_id=nodes_by_id,
                frame_rows=frame_rows,
                aggregates=aggregates,
                artifact_id=registration.artifact_id,
            )
        measurement_rows: list[dict[str, Any]] = [
            {
                "measurement_id": digest_model(
                    {
                        "run_id": run_id,
                        "artifact_id": registration.artifact_id,
                        "name": name,
                    }
                ),
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "name": name,
                "value_int": value,
                "value_float": None,
                "unit": unit,
                "aggregation": aggregation,
                "scope": "process",
                "trial_id": None,
                "worker_id": None,
                "worker_run_index": None,
                "value_index": None,
                "loop_count": None,
                "is_warmup": False,
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "phase": None,
                "dimensions": {},
                "evidence_level": "observed",
            }
            for name, value, unit, aggregation in (
                ("cpu.samples", len(samples), "count", "total"),
                ("cpu.nodes", len(nodes), "count", "total"),
            )
        ]
        frame_measurements = [
            {
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "frame_id": frame_id,
                "metric": metric,
                "self_value": values["self"],
                "inclusive_value": values["inclusive"],
                "unit": "count",
                "sample_count": values["samples"],
                "thread_name": None,
                "process_name": None,
                "phase": None,
            }
            for (metric, frame_id), values in sorted(aggregates.items())
        ]
        published = self.publisher.publish_rows(
            {
                "measurements": measurement_rows,
                "frames": list(frame_rows.values()),
                "frame_measurements": frame_measurements,
            },
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
        )
        return V8CpuProfExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            node_count=len(nodes),
            sample_count=len(samples),
            frame_count=len(frame_rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=(
                "V8 CPU samples represent execution time, not allocation or memory evidence.",
                "The CPU profile contains sampled stack locations; source-map resolution "
                "is not applied by this extractor.",
            ),
        )

    def _aggregate_node(
        self,
        node: dict[str, Any],
        *,
        nodes_by_id: dict[int, dict[str, Any]],
        frame_rows: dict[str, dict[str, Any]],
        aggregates: dict[tuple[str, str], dict[str, int]],
        artifact_id: str,
    ) -> None:
        call_frame = node.get("callFrame") or {}
        function = str(call_frame.get("functionName") or "(anonymous)")
        url = str(call_frame.get("url") or "")
        line = int(call_frame.get("lineNumber") or 0)
        column = int(call_frame.get("columnNumber") or 0)
        script_id = str(call_frame.get("scriptId") or "")
        normalized = self._normalize(url)
        frame_id = digest_model(
            {
                "language": "JavaScript",
                "function": function,
                "file": normalized,
                "line": line,
                "column": column,
                "script_id": script_id,
            }
        )
        frame_rows.setdefault(
            frame_id,
            {
                "frame_id": frame_id,
                "language": "JavaScript",
                "function": function,
                "module": None,
                "file": normalized,
                "line": line,
                "column": column,
                "address": None,
                "build_id": None,
                "module_relative_address": None,
                "inline_chain_id": None,
                "source_state_id": None,
                "artifact_id": artifact_id,
                "inlined": False,
                "symbolization": "complete",
            },
        )
        hit_count = int(node.get("hitCount", 0))
        key = ("cpu.hit_count", frame_id)
        values = aggregates.setdefault(
            key, {"self": 0, "inclusive": 0, "samples": 0}
        )
        values["self"] += hit_count
        children = node.get("children") or []
        for child_id in children:
            child = nodes_by_id.get(int(child_id))
            if child is not None:
                self._aggregate_node(
                    child,
                    nodes_by_id=nodes_by_id,
                    frame_rows=frame_rows,
                    aggregates=aggregates,
                    artifact_id=artifact_id,
                )
        values["inclusive"] += hit_count
        values["samples"] += hit_count

    def _normalize(self, url: str) -> str:
        if not url:
            return ""
        if url.startswith("node:") or not Path(url).is_absolute():
            return url
        try:
            resolved = Path(url).resolve()
            return resolved.relative_to(self.workspace.project_root).as_posix()
        except (ValueError, OSError):
            return url
