from __future__ import annotations

import math
import re
from typing import Any

import pyperf

from flameox.adapters.compatibility import require_supported_producer_major
from flameox.adapters.pyperf import PyPerfSample, iter_pyperf_samples, load_pyperf_suite
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import ArtifactKind, ArtifactRegistration, RunManifest
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.startup_profile import PYTHON_STARTUP_PROFILE
from flameox.storage import ArtifactStore, RunStore, Workspace

_IMPORT_TIME = re.compile(
    r"^import time:\s+(?P<self>\d+)\s+\|\s+(?P<cumulative>\d+)\s+\|\s+(?P<module>.+)$"
)


class PythonStartupExtractionResult(ContractModel):
    schema_version: int = 2
    run_id: str
    artifact_id: str
    import_trace_artifact_id: str
    sample_count: int
    package_count: int
    measurement_count: int
    peak_rss_backends: tuple[str, ...]
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class PythonStartupExtractor:
    name = "python-startup"
    version = "2"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> PythonStartupExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        benchmark_registration, import_registration = self._registrations(run)
        limitations = [
            *require_supported_producer_major(
                benchmark_registration,
                package="pyperf",
                producer_tokens=("pyperf",),
            ),
            "The initial OS file-cache state was uncontrolled; no caches were dropped.",
            "Import-time instrumentation affects the separate trace process, not wall samples.",
        ]
        artifacts = ArtifactStore(self.workspace)
        benchmark_artifact = artifacts.get(benchmark_registration.artifact_id)
        import_artifact = artifacts.get(import_registration.artifact_id)
        suite = load_pyperf_suite(benchmark_artifact.payload_path)
        samples = self._startup_samples(suite, run_id=run_id)
        try:
            import_bytes = import_artifact.payload_path.read_bytes()
        except OSError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Python import-time artifact could not be read.",
                run_id=run_id,
            ) from exc
        import_text = import_bytes.decode("utf-8", errors="replace")
        if "\ufffd" in import_text:
            limitations.append("The import-time trace contained non-UTF-8 bytes.")
        _, package_rows, unparsed_lines = _group_imports(import_text)

        rows: list[dict[str, Any]] = []
        peak_rss_backends: set[str] = set()
        missing_rss = False
        for sample in samples:
            cache_semantics = (
                "uncontrolled_initial" if sample.worker_run_index == 0 else "warm_process_restart"
            )
            dimensions = {
                "cache_semantics": cache_semantics,
                "profile_id": PYTHON_STARTUP_PROFILE.profile_id,
                "pyperf_unit": sample.unit,
            }
            rows.append(
                self._row(
                    run_id,
                    benchmark_registration.artifact_id,
                    "python_startup.wall_time",
                    round(sample.value * 1_000_000_000),
                    "ns",
                    sample.value_index,
                    "startup",
                    dimensions,
                    worker_id=f"startup:{sample.worker_run_index}",
                    worker_run_index=sample.worker_run_index,
                    loop_count=sample.loop_count,
                )
            )
            peak_rss = sample.run_metadata.get("command_max_rss")
            if isinstance(peak_rss, int) and not isinstance(peak_rss, bool) and peak_rss >= 0:
                backend = "pyperf.command_max_rss"
                peak_rss_backends.add(backend)
                rows.append(
                    self._row(
                        run_id,
                        benchmark_registration.artifact_id,
                        "python_startup.peak_rss",
                        peak_rss,
                        "bytes",
                        sample.value_index,
                        "startup",
                        {**dimensions, "rss_backend": backend},
                        worker_id=f"startup:{sample.worker_run_index}",
                        worker_run_index=sample.worker_run_index,
                        loop_count=sample.loop_count,
                    )
                )
            else:
                missing_rss = True

        packages: set[str] = set()
        for package in package_rows:
            package_name = _text(package, "package")
            packages.add(package_name)
            package_dimensions = {"package": package_name}
            for name, field, unit in (
                ("python_startup.import.module_count", "module_count", "count"),
                ("python_startup.import.self_time", "self_us", "us"),
                (
                    "python_startup.import.max_cumulative_time",
                    "max_cumulative_us",
                    "us",
                ),
            ):
                rows.append(
                    self._row(
                        run_id,
                        import_registration.artifact_id,
                        name,
                        _integer(package, field),
                        unit,
                        0,
                        "import",
                        package_dimensions,
                    )
                )
        if missing_rss:
            limitations.append(
                "pyperf command_max_rss was unavailable for at least one startup sample."
            )
        if unparsed_lines:
            limitations.append(
                f"{unparsed_lines} import-time data lines used an unsupported representation."
            )
        maximum = self.workspace.config.storage.max_rows_per_generation
        if len(rows) > maximum:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Python startup extraction exceeded the {maximum}-row generation limit.",
            )
        published = self.publisher.publish_rows(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(
                benchmark_registration.artifact_id,
                import_registration.artifact_id,
            ),
        )
        return PythonStartupExtractionResult(
            run_id=run_id,
            artifact_id=benchmark_registration.artifact_id,
            import_trace_artifact_id=import_registration.artifact_id,
            sample_count=len(samples),
            package_count=len(packages),
            measurement_count=len(rows),
            peak_rss_backends=tuple(sorted(peak_rss_backends)),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _startup_samples(
        suite: pyperf.BenchmarkSuite,
        *,
        run_id: str,
    ) -> tuple[PyPerfSample, ...]:
        samples = tuple(iter_pyperf_samples(suite))
        expected_runs = tuple(range(PYTHON_STARTUP_PROFILE.process_count))
        valid = (
            tuple(suite.get_benchmark_names()) == (PYTHON_STARTUP_PROFILE.benchmark_name,)
            and len(samples) == PYTHON_STARTUP_PROFILE.sample_count
            and tuple(sample.worker_run_index for sample in samples) == expected_runs
            and all(
                sample.benchmark_name == PYTHON_STARTUP_PROFILE.benchmark_name
                and sample.unit == "second"
                and sample.value_index == 0
                and sample.loop_count == PYTHON_STARTUP_PROFILE.loops_per_value
                and not sample.is_warmup
                and math.isfinite(sample.value)
                and sample.value >= 0
                for sample in samples
            )
        )
        if not valid:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The pyperf artifact does not match the closed Python startup profile.",
                run_id=run_id,
            )
        return samples

    @staticmethod
    def _registrations(
        run: RunManifest,
    ) -> tuple[ArtifactRegistration, ArtifactRegistration]:
        benchmark = [
            item
            for item in run.artifacts
            if item.kind is ArtifactKind.BENCHMARK_SAMPLES and item.role == "startup_wall"
        ]
        import_trace = [
            item
            for item in run.artifacts
            if item.kind is ArtifactKind.PYTHON_STARTUP and item.role == "import_trace"
        ]
        if len(benchmark) != 1 or len(import_trace) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain one startup-wall pyperf artifact and one import-time trace.",
                run_id=run.run_id,
            )
        return benchmark[0], import_trace[0]

    @staticmethod
    def _row(
        run_id: str,
        artifact_id: str,
        name: str,
        value: int,
        unit: str,
        value_index: int,
        phase: str,
        dimensions: dict[str, str],
        *,
        worker_id: str | None = None,
        worker_run_index: int | None = None,
        loop_count: int | None = None,
    ) -> dict[str, Any]:
        identity = {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": name,
            "worker_run_index": worker_run_index,
            "value_index": value_index,
            "dimensions": dimensions,
        }
        return {
            "measurement_id": digest_model(identity),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": name,
            "value_int": value,
            "value_float": None,
            "unit": unit,
            "aggregation": "sample",
            "scope": "workload",
            "trial_id": None,
            "worker_id": worker_id,
            "worker_run_index": worker_run_index,
            "value_index": value_index,
            "loop_count": loop_count,
            "is_warmup": False,
            "block_id": None,
            "variant_id": None,
            "order_in_block": None,
            "phase": phase,
            "dimensions": dimensions,
            "evidence_level": "observed",
        }


def _group_imports(stderr: str) -> tuple[str, list[dict[str, Any]], int]:
    raw_lines: list[str] = []
    modules: dict[str, tuple[int, int]] = {}
    ignored_lines = 0
    for line in stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        raw_lines.append(line)
        match = _IMPORT_TIME.match(line)
        if match is None:
            if "self [us]" not in line:
                ignored_lines += 1
            continue
        module = match.group("module").strip()
        self_us = int(match.group("self"))
        cumulative_us = int(match.group("cumulative"))
        previous_self, previous_cumulative = modules.get(module, (0, 0))
        modules[module] = (
            previous_self + self_us,
            max(previous_cumulative, cumulative_us),
        )

    grouped: dict[str, dict[str, int]] = {}
    for module, (self_us, cumulative_us) in modules.items():
        normalized = module.lstrip(".")
        package = normalized.split(".", 1)[0] or module
        group = grouped.setdefault(
            package,
            {"module_count": 0, "self_us": 0, "max_cumulative_us": 0},
        )
        group["module_count"] += 1
        group["self_us"] += self_us
        group["max_cumulative_us"] = max(group["max_cumulative_us"], cumulative_us)
    return (
        "\n".join(raw_lines) + ("\n" if raw_lines else ""),
        [{"package": package, **values} for package, values in sorted(grouped.items())],
        ignored_lines,
    )


def _integer(value: Any, field: str) -> int:
    result = value.get(field) if isinstance(value, dict) else None
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Python startup field {field!r} is invalid.",
        )
    return result


def _text(value: Any, field: str) -> str:
    result = value.get(field) if isinstance(value, dict) else None
    if not isinstance(result, str) or not result:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Python startup field {field!r} is invalid.",
        )
    return result
