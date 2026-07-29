from __future__ import annotations

import json
from typing import Any

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import ArtifactKind, RunManifest
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class PythonStartupExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    sample_count: int
    package_count: int
    measurement_count: int
    peak_rss_backends: tuple[str, ...]
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class PythonStartupExtractor:
    name = "python-startup"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> PythonStartupExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registration = self._registration(run)
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        try:
            payload = json.loads(artifact.payload_path.read_text())
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != "flameox.python-startup.v1"
            ):
                raise ValueError("unsupported schema")
            samples = payload["samples"]
            if not isinstance(samples, list) or not samples:
                raise ValueError("samples are missing")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported Python startup capture.",
                run_id=run_id,
            ) from exc

        rows: list[dict[str, Any]] = []
        packages: set[str] = set()
        limitations = [
            "The initial OS file-cache state was uncontrolled; no caches were dropped.",
            "Import-time instrumentation affects the separate trace process, not wall samples.",
        ]
        missing_rss = False
        peak_rss_backends: set[str] = set()
        for sample in samples:
            index = _integer(sample, "index")
            cache_semantics = _text(sample, "cache_semantics")
            dimensions = {"cache_semantics": cache_semantics}
            rows.append(
                self._row(
                    run_id,
                    registration.artifact_id,
                    "python_startup.wall_time",
                    _integer(sample, "duration_ns"),
                    "ns",
                    index,
                    "startup",
                    dimensions,
                )
            )
            peak_rss = sample.get("peak_rss_bytes")
            if isinstance(peak_rss, int):
                peak_rss_backend = _text(sample, "peak_rss_backend")
                peak_rss_backends.add(peak_rss_backend)
                rows.append(
                    self._row(
                        run_id,
                        registration.artifact_id,
                        "python_startup.peak_rss",
                        peak_rss,
                        "bytes",
                        index,
                        "startup",
                        {**dimensions, "rss_backend": peak_rss_backend},
                    )
                )
            else:
                missing_rss = True
        import_trace = payload.get("import_trace")
        if not isinstance(import_trace, dict):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Python startup artifact has no import trace.",
                run_id=run_id,
            )
        package_rows = import_trace.get("packages")
        if not isinstance(package_rows, list):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Python startup import package list is invalid.",
                run_id=run_id,
            )
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
                        registration.artifact_id,
                        name,
                        _integer(package, field),
                        unit,
                        0,
                        "import",
                        package_dimensions,
                    )
                )
        if missing_rss:
            limitations.append("Peak RSS was unavailable for at least one short-lived sample.")
        unparsed_lines = import_trace.get("unparsed_importtime_lines", 0)
        if (
            not isinstance(unparsed_lines, int)
            or isinstance(unparsed_lines, bool)
            or unparsed_lines < 0
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Python startup unparsed-line count is invalid.",
                run_id=run_id,
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
            input_artifact_ids=(registration.artifact_id,),
        )
        return PythonStartupExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            sample_count=len(samples),
            package_count=len(packages),
            measurement_count=len(rows),
            peak_rss_backends=tuple(sorted(peak_rss_backends)),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    def _registration(self, run: RunManifest) -> Any:
        matches = [item for item in run.artifacts if item.kind is ArtifactKind.PYTHON_STARTUP]
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one Python startup artifact.",
                run_id=run.run_id,
            )
        return matches[0]

    def _row(
        self,
        run_id: str,
        artifact_id: str,
        name: str,
        value: int,
        unit: str,
        value_index: int,
        phase: str,
        dimensions: dict[str, str],
    ) -> dict[str, Any]:
        identity = {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": name,
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
            "worker_id": None,
            "worker_run_index": None,
            "value_index": value_index,
            "loop_count": None,
            "is_warmup": False,
            "block_id": None,
            "variant_id": None,
            "order_in_block": None,
            "phase": phase,
            "dimensions": dimensions,
            "evidence_level": "observed",
        }


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
