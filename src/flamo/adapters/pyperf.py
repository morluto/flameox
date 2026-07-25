from __future__ import annotations

from pathlib import Path
from typing import Any

import pyperf
from pydantic import BaseModel, ConfigDict

from flamo.domain.errors import DomainError, ErrorCode
from flamo.domain.identity import digest_model
from flamo.domain.models import ArtifactKind, RunManifest
from flamo.evidence import GenerationPublisher
from flamo.storage import ArtifactStore, RunStore, Workspace


class PyPerfExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    run_id: str
    artifact_id: str
    benchmark_names: tuple[str, ...]
    measurement_count: int
    warmup_count: int
    corpus_commit_id: str


class PyPerfExtractor:
    name = "pyperf"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> PyPerfExtractionResult:
        run = self.runs.read(run_id)
        registration = self._benchmark_registration(run)
        stored = self.artifacts.get(registration.artifact_id)
        suite = self._load_suite(stored.payload_path)
        rows: list[dict[str, Any]] = []
        measured_count = 0
        warmup_count = 0

        for benchmark in suite.get_benchmarks():
            benchmark_name = benchmark.get_name()
            unit = benchmark.get_unit()
            for run_index, benchmark_run in enumerate(benchmark.get_runs()):
                worker_id = f"{benchmark_name}:{run_index}"
                for warmup_index, (loops, value) in enumerate(benchmark_run.warmups):
                    rows.append(
                        self._measurement_row(
                            run=run,
                            artifact_id=registration.artifact_id,
                            benchmark_name=benchmark_name,
                            unit=unit,
                            worker_id=worker_id,
                            worker_run_index=run_index,
                            value_index=warmup_index,
                            value=value,
                            loop_count=loops,
                            is_warmup=True,
                        )
                    )
                    warmup_count += 1
                loops = benchmark_run.get_loops()
                for value_index, value in enumerate(benchmark_run.values):
                    rows.append(
                        self._measurement_row(
                            run=run,
                            artifact_id=registration.artifact_id,
                            benchmark_name=benchmark_name,
                            unit=unit,
                            worker_id=worker_id,
                            worker_run_index=run_index,
                            value_index=value_index,
                            value=value,
                            loop_count=loops,
                            is_warmup=False,
                        )
                    )
                    measured_count += 1

        published = self.publisher.publish_rows(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
        )
        return PyPerfExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            benchmark_names=tuple(suite.get_benchmark_names()),
            measurement_count=measured_count,
            warmup_count=warmup_count,
            corpus_commit_id=published.commit.commit_id,
        )

    def _benchmark_registration(self, run: RunManifest) -> Any:
        matches = [
            registration
            for registration in run.artifacts
            if registration.kind is ArtifactKind.BENCHMARK_SAMPLES
        ]
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one pyperf benchmark artifact.",
                run_id=run.run_id,
            )
        return matches[0]

    def _load_suite(self, path: Path) -> pyperf.BenchmarkSuite:
        try:
            return pyperf.BenchmarkSuite.load(str(path))
        except (OSError, ValueError, TypeError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported pyperf benchmark suite.",
            ) from exc

    def _measurement_row(
        self,
        *,
        run: RunManifest,
        artifact_id: str,
        benchmark_name: str,
        unit: str,
        worker_id: str,
        worker_run_index: int,
        value_index: int,
        value: float,
        loop_count: int,
        is_warmup: bool,
    ) -> dict[str, Any]:
        value_int: int | None = None
        value_float: float | None = None
        normalized_unit = unit
        if unit == "second":
            value_int = round(value * 1_000_000_000)
            normalized_unit = "ns"
        elif unit == "byte":
            value_int = round(value)
            normalized_unit = "bytes"
        elif unit == "integer":
            value_int = round(value)
            normalized_unit = "count"
        else:
            value_float = float(value)
        identity = {
            "run_id": run.run_id,
            "artifact_id": artifact_id,
            "benchmark": benchmark_name,
            "worker_id": worker_id,
            "worker_run_index": worker_run_index,
            "value_index": value_index,
            "is_warmup": is_warmup,
        }
        return {
            "measurement_id": digest_model(identity),
            "run_id": run.run_id,
            "artifact_id": artifact_id,
            "name": f"pyperf.{benchmark_name}",
            "value_int": value_int,
            "value_float": value_float,
            "unit": normalized_unit,
            "aggregation": "sample",
            "scope": "workload",
            "trial_id": None,
            "worker_id": worker_id,
            "worker_run_index": worker_run_index,
            "value_index": value_index,
            "loop_count": loop_count,
            "is_warmup": is_warmup,
            "block_id": None,
            "variant_id": None,
            "order_in_block": None,
            "phase": "warmup" if is_warmup else "steady_state",
            "dimensions": {
                "benchmark": benchmark_name,
                "pyperf_unit": unit,
            },
            "evidence_level": "observed",
        }
