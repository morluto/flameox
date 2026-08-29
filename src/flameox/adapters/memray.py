from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from flameox.adapters.compatibility import require_supported_producer_major
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


class MemrayExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    peak_memory_bytes: int
    retained_end_bytes: int
    total_allocations: int
    frame_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class MemrayExtractor:
    name = "memray"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(
        self,
        run_id: str,
        *,
        cancel_check: Callable[[], None] | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> MemrayExtractionResult:
        self._check_cancelled(cancel_check)
        run = RunStore(self.workspace).read(run_id)
        registrations = [item for item in run.artifacts if item.kind is ArtifactKind.MEMORY_PROFILE]
        if not registrations:
            raise missing_artifact_input(
                run_id=run_id,
                requirement="Memray memory-profile",
                artifact_kinds=(ArtifactKind.MEMORY_PROFILE.value,),
                capture_adapters=("memray",),
                import_producers=("memray",),
            )
        try:
            import memray
        except ImportError as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Memray is not installed.",
                remediation=("Install flameox's memory optional dependencies.",),
            ) from exc
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one Memray artifact.",
                run_id=run_id,
            )
        registration = registrations[0]
        compatibility_limitations = require_supported_producer_major(
            registration,
            package="memray",
            producer_tokens=("memray",),
        )
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        try:
            self._check_cancelled(cancel_check)
            reader = memray.FileReader(str(artifact.payload_path))
            high_watermark = self._materialize_records(
                reader.get_high_watermark_allocation_records(),
                phase="reading_high_watermark",
                cancel_check=cancel_check,
                progress=progress,
            )
            retained = self._materialize_records(
                reader.get_leaked_allocation_records(),
                phase="reading_retained_end",
                cancel_check=cancel_check,
                progress=progress,
            )
            metadata = reader.metadata
        except (OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported Memray capture.",
                run_id=run_id,
            ) from exc
        frame_rows: dict[str, dict[str, Any]] = {}
        aggregates: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"self": 0, "inclusive": 0, "samples": 0}
        )
        self._aggregate(
            high_watermark,
            metric="memory.high_watermark",
            frame_rows=frame_rows,
            aggregates=aggregates,
            artifact_id=registration.artifact_id,
            cancel_check=cancel_check,
            progress=progress,
        )
        self._aggregate(
            retained,
            metric="memory.retained_end",
            frame_rows=frame_rows,
            aggregates=aggregates,
            artifact_id=registration.artifact_id,
            cancel_check=cancel_check,
            progress=progress,
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
                ("memory.peak", int(metadata.peak_memory), "bytes", "peak"),
                (
                    "memory.retained_end",
                    sum(int(record.size) for record in retained),
                    "bytes",
                    "total",
                ),
                (
                    "memory.total_allocations",
                    int(metadata.total_allocations),
                    "count",
                    "total",
                ),
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
                "unit": "bytes",
                "sample_count": values["samples"],
                "thread_name": None,
                "process_name": None,
                "phase": None,
            }
            for (metric, frame_id), values in sorted(aggregates.items())
        ]
        self._check_cancelled(cancel_check)
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
        limitations = [
            *compatibility_limitations,
            "Frame aggregates expose bounded callers; complete stacks remain in Memray.",
        ]
        if not metadata.has_native_traces:
            limitations.append("The capture does not contain native stack traces.")
        return MemrayExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            peak_memory_bytes=int(metadata.peak_memory),
            retained_end_bytes=sum(int(record.size) for record in retained),
            total_allocations=int(metadata.total_allocations),
            frame_count=len(frame_rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    def _aggregate(
        self,
        records: list[Any],
        *,
        metric: str,
        frame_rows: dict[str, dict[str, Any]],
        aggregates: dict[tuple[str, str], dict[str, int]],
        artifact_id: str,
        cancel_check: Callable[[], None] | None,
        progress: Callable[[str, int, int | None], None] | None,
    ) -> None:
        total = len(records)
        phase = f"aggregating_{metric.removeprefix('memory.')}"
        for record_index, record in enumerate(records, start=1):
            if record_index == 1 or record_index % 256 == 0:
                self._check_cancelled(cancel_check)
            stack = list(record.stack_trace())
            for index, (function, filename, line) in enumerate(stack):
                normalized = self._normalize(filename)
                frame_id = digest_model(
                    {
                        "language": "Python",
                        "function": function,
                        "file": normalized,
                        "line": line,
                    }
                )
                frame_rows.setdefault(
                    frame_id,
                    {
                        "frame_id": frame_id,
                        "language": "Python",
                        "function": function,
                        "module": None,
                        "file": normalized,
                        "line": line,
                        "column": None,
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
                values = aggregates[(metric, frame_id)]
                values["inclusive"] += int(record.size)
                values["samples"] += int(record.n_allocations)
                if index == 0:
                    values["self"] += int(record.size)
            if progress is not None and (record_index == total or record_index % 1_024 == 0):
                progress(phase, record_index, total)

    @staticmethod
    def _check_cancelled(cancel_check: Callable[[], None] | None) -> None:
        if cancel_check is not None:
            cancel_check()

    def _materialize_records(
        self,
        records: Iterable[Any],
        *,
        phase: str,
        cancel_check: Callable[[], None] | None,
        progress: Callable[[str, int, int | None], None] | None,
    ) -> list[Any]:
        materialized: list[Any] = []
        for record_index, record in enumerate(records, start=1):
            if record_index == 1 or record_index % 256 == 0:
                self._check_cancelled(cancel_check)
            materialized.append(record)
            if progress is not None and record_index % 1_024 == 0:
                progress(phase, record_index, None)
        self._check_cancelled(cancel_check)
        if progress is not None:
            progress(phase, len(materialized), len(materialized))
        return materialized

    def _normalize(self, filename: str) -> str:
        if filename.startswith("<") and filename.endswith(">"):
            return filename
        path = Path(filename).resolve()
        try:
            return path.relative_to(self.workspace.project_root).as_posix()
        except ValueError:
            return str(path)
