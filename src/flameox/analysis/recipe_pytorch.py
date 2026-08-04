from __future__ import annotations

import json

import numpy as np

from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import OperatorSummary, PyTorchAnalysisResult
from flameox.catalog import Snapshot
from flameox.domain import DomainError, ErrorCode
from flameox.evidence_scope import resolve_evidence_scope
from flameox.evidence_status import available_availability, empty_availability


class PyTorchRecipes(RecipeContext):
    def pytorch(
        self,
        input_id: str,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> PyTorchAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            self._require_pytorch_source(snapshot, scope.run_ids, scope.artifact_ids)
            where, parameters = scope.predicate(
                run_column="fm.run_id",
                artifact_column="fm.artifact_id",
            )
            count_row = snapshot.execute(
                "SELECT count(DISTINCT fm.frame_id) FROM frame_measurements fm WHERE " + where,
                parameters,
            ).fetchone()
            assert count_row is not None
            total = int(count_row[0])
            if total == 0:
                if scope.run_ids:
                    run_ids = scope.run_ids
                elif scope.artifact_ids:
                    run_rows = snapshot.execute(
                        "SELECT DISTINCT run_id FROM artifact_registrations WHERE artifact_id IN ("
                        + ", ".join("?" for _ in scope.artifact_ids)
                        + ") ORDER BY run_id",
                        scope.artifact_ids,
                    ).fetchall()
                    run_ids = tuple(str(row[0]) for row in run_rows)
                else:
                    run_ids = ()
                details: dict[str, object] = {"next_tool": "extract_perfetto"}
                if run_ids:
                    details["run_id"] = run_ids[0]
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "PyTorch operator analysis requires Perfetto extraction for this "
                    "imported trace.",
                    details=details,
                    remediation=(
                        "Call extract_perfetto with the reported run_id, then retry "
                        "analyze_pytorch.",
                        "If Trace Processor is unavailable, call prepare_capabilities with "
                        "adapter='perfetto'.",
                    ),
                )
            rows = snapshot.execute(
                "SELECT fm.frame_id, coalesce(f.function, '<unnamed>'), f.module, "
                "sum(coalesce(fm.self_value, 0)), "
                "sum(coalesce(fm.inclusive_value, 0)), "
                "sum(coalesce(fm.sample_count, 0)) "
                "FROM frame_measurements fm JOIN frames f "
                "ON f.frame_id = fm.frame_id WHERE "
                + where
                + " GROUP BY fm.frame_id, f.function, f.module "
                "ORDER BY sum(coalesce(fm.inclusive_value, 0)) DESC, fm.frame_id "
                "LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
            observation_where, observation_parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            metadata_rows = snapshot.execute(
                "SELECT name, value_json, context FROM observations WHERE "
                + observation_where
                + " AND kind = 'pytorch.operator'",
                observation_parameters,
            ).fetchall()
        metadata_by_operator: dict[tuple[str, str], list[dict[str, object]]] = {}
        for name, value_json, context in metadata_rows:
            try:
                value = json.loads(str(value_json))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            frame_id = value.get("frame_id")
            if not isinstance(frame_id, str):
                continue
            if context is not None and value.get("phase") is None:
                value["phase"] = str(context)
            metadata_by_operator.setdefault((frame_id, str(name)), []).append(value)
        operators_list: list[OperatorSummary] = []
        device_time_present = False
        synchronization_present = False
        for row in rows:
            category = str(row[2]) if row[2] is not None else None
            category_lower = (category or "").lower()
            operator = str(row[1])
            operator_lower = operator.lower()
            frame_id = str(row[0])
            metadata = metadata_by_operator.get((frame_id, operator), [])
            is_device = any(
                token in category_lower
                for token in ("kernel", "gpu", "device", "xpu", "hip", "mps")
            )
            synchronization = any(
                token in operator_lower
                for token in (
                    "synchronize",
                    "cudadevicesynchronize",
                    "cudastreamsynchronize",
                    "event_synchronize",
                )
            )
            device_time_present = device_time_present or is_device
            synchronization_present = synchronization_present or synchronization
            inclusive = int(row[4])
            shapes = tuple(
                sorted(
                    {
                        str(value["input_shapes"])
                        for value in metadata
                        if value.get("input_shapes") not in {None, ""}
                    }
                )
            )
            allocations = [
                allocation
                for value in metadata
                if isinstance(allocation := value.get("allocation_bytes"), int)
            ]
            phases = {
                str(value["phase"]).lower()
                for value in metadata
                if value.get("phase") not in {None, ""}
            }
            warmup_phases = {phase for phase in phases if "warm" in phase}
            warmup = (
                True
                if phases and warmup_phases == phases
                else False
                if phases and not warmup_phases
                else None
            )
            operators_list.append(
                OperatorSummary(
                    frame_id=frame_id,
                    operator=operator,
                    category=category,
                    self_cpu_ns=None if is_device else int(row[3]),
                    total_cpu_ns=None if is_device else inclusive,
                    device_ns=inclusive if is_device else None,
                    inclusive_ns=inclusive,
                    event_count=int(row[5]),
                    input_shapes=shapes,
                    allocation_bytes=sum(allocations) if allocations else None,
                    synchronization=synchronization,
                    warmup=warmup,
                )
            )
        operators = tuple(operators_list)
        synchronization_time_ns = sum(
            item.inclusive_ns for item in operators if item.synchronization
        )
        compilation_time_ns = sum(
            item.inclusive_ns
            for item in operators
            if any(
                token in item.operator.lower()
                for token in ("compile", "dynamo", "inductor", "graph_executor")
            )
        )
        warmup_time_ns = sum(
            duration
            for metadata in metadata_by_operator.values()
            for value in metadata
            if isinstance(duration := value.get("duration_ns"), int)
            and "warm" in str(value.get("phase", "")).lower()
        )
        allocation_bytes = sum(item.allocation_bytes or 0 for item in operators) or None
        per_event_times = [
            operator.inclusive_ns / max(operator.event_count, 1) for operator in operators
        ]
        typical_event_ns = max(
            1.0,
            float(np.percentile(per_event_times, 25)) if per_event_times else 1.0,
        )
        repeated_small = tuple(
            sorted(
                (
                    item
                    for item in operators
                    if item.event_count >= 3
                    and item.inclusive_ns / item.event_count <= typical_event_ns
                ),
                key=lambda item: (-item.event_count, item.inclusive_ns, item.frame_id),
            )[:bounded]
        )
        limitations = [
            "Operator categories and durations come from the exported torch.profiler trace.",
            "Nested operator durations can overlap; self time subtracts direct nested slices.",
        ]
        if not device_time_present:
            limitations.append("The trace contains no recognized accelerator kernel categories.")
        shapes_present = any(item.input_shapes for item in operators)
        allocations_present = any(item.allocation_bytes is not None for item in operators)
        warmup_present = any(item.warmup is not None for item in operators)
        if not shapes_present:
            limitations.append("Input shapes were not present in normalized trace evidence.")
        if not allocations_present:
            limitations.append(
                "Per-operator allocation bytes were not present in normalized trace evidence."
            )
        if not warmup_present:
            limitations.append("Warm-up separation requires profiler phase annotations.")
        return PyTorchAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            operators=operators,
            total=total,
            returned=len(operators),
            truncated=total > len(operators),
            coverage={
                "self_cpu_time": True,
                "total_cpu_time": True,
                "device_time": device_time_present,
                "input_shapes": shapes_present,
                "memory_allocations": allocations_present,
                "synchronization": synchronization_present,
                "warmup_phases": warmup_present,
            },
            repeated_small_operations=repeated_small,
            synchronization_time_ns=synchronization_time_ns,
            compilation_time_ns=compilation_time_ns,
            warmup_time_ns=warmup_time_ns,
            allocation_bytes=allocation_bytes,
            limitations=tuple(limitations),
            evidence=(
                empty_availability("no_normalized_torch_operators")
                if total == 0
                else available_availability()
            ),
        )

    def _require_pytorch_source(
        self,
        snapshot: Snapshot,
        run_ids: tuple[str, ...],
        artifact_ids: tuple[str, ...],
    ) -> None:
        if run_ids:
            placeholders = ", ".join("?" for _ in run_ids)
            rows = snapshot.execute(
                "SELECT lower(coalesce(collector, '')) FROM ("
                "SELECT *, row_number() OVER (PARTITION BY run_id "
                "ORDER BY published_at DESC) AS revision_order FROM runs"
                f") WHERE revision_order = 1 AND run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
            if len(rows) == len(set(run_ids)) and all("torch" in str(row[0]) for row in rows):
                return
            producer_rows = snapshot.execute(
                "SELECT DISTINCT run_id FROM artifact_registrations "
                f"WHERE run_id IN ({placeholders}) "
                "AND lower(coalesce(producer, '')) LIKE '%torch%'",
                run_ids,
            ).fetchall()
            if {str(row[0]) for row in producer_rows} == set(run_ids):
                return
        if artifact_ids:
            placeholders = ", ".join("?" for _ in artifact_ids)
            rows = snapshot.execute(
                "SELECT DISTINCT lower(coalesce(producer, '')) "
                "FROM artifact_registrations "
                f"WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            ).fetchall()
            if any("torch" in str(row[0]) for row in rows):
                return
        raise DomainError(
            ErrorCode.COMPARISON_INVALID,
            "PyTorch operator analysis requires a torch.profiler-produced trace.",
            details={
                "next_tool": "import_artifact",
                "required_kind": "execution_trace",
                "required_producer": "torch.profiler",
            },
            remediation=(
                "Re-import the trace with kind='execution_trace'; Torch markers are detected "
                "automatically.",
                "If detection is ambiguous, set producer='torch.profiler' on import_artifact.",
            ),
        )
