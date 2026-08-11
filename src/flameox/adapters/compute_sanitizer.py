from __future__ import annotations

import json
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

from flameox.application.artifact_workers import ArtifactWorker
from flameox.domain import ArtifactKind, DomainError, ErrorCode, digest_model
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

_SUPPORTED_PRODUCER_MAJOR = 2026


class ComputeSanitizerExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    producer_version: str | None
    status: Literal["clean", "findings", "inconclusive"]
    finding_count: int
    classifications: dict[str, int]
    schema_fingerprint: str
    corpus_commit_id: str
    limitations: tuple[str, ...]


class ComputeSanitizerInspection(ContractModel):
    records: tuple[dict[str, Any], ...]
    classifications: dict[str, int]
    limitations: tuple[str, ...]


def inspect_compute_sanitizer_report(
    workspace: Workspace,
    artifact_path: str,
    *,
    max_records: int,
    max_frames: int = 64,
) -> ComputeSanitizerInspection:
    response = ArtifactWorker(workspace).run_sync(
        "flameox.workers.compute_sanitizer",
        {
            "artifact_path": artifact_path,
            "project_root": str(workspace.project_root),
            "max_records": max_records,
            "max_frames": max_frames,
        },
        name="Compute Sanitizer",
        timeout_seconds=120,
    )
    records = response.get("records")
    classifications = response.get("classifications")
    raw_limitations = response.get("limitations")
    if (
        not isinstance(records, list)
        or any(not isinstance(item, dict) for item in records)
        or not isinstance(classifications, dict)
        or any(
            not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int)
            for key, value in classifications.items()
        )
        or not isinstance(raw_limitations, list)
        or any(not isinstance(item, str) for item in raw_limitations)
    ):
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            "Compute Sanitizer worker returned an invalid normalized result.",
        )
    limitations = list(raw_limitations)
    if any(record.get("classification") == "unknown" for record in records):
        limitations.append("One or more Compute Sanitizer record shapes were not classified.")
    return ComputeSanitizerInspection(
        records=tuple(records),
        classifications={str(key): int(value) for key, value in classifications.items()},
        limitations=tuple(dict.fromkeys(limitations)),
    )


def compute_sanitizer_compatibility_limitations(
    producer_version: str | None,
) -> tuple[str, ...]:
    """Describe whether a report belongs to the explicitly verified XML family."""
    if producer_version is None:
        return (
            "Compute Sanitizer producer version is unavailable; XML compatibility is unverified.",
        )
    normalized = producer_version.strip()
    if normalized.casefold().startswith("version "):
        normalized = normalized.split(maxsplit=2)[1]
    try:
        observed_major = Version(normalized).major
    except InvalidVersion:
        return (
            "Compute Sanitizer producer version is not identifiable; XML compatibility is "
            "unverified.",
        )
    if observed_major != _SUPPORTED_PRODUCER_MAJOR:
        return (
            f"Compute Sanitizer {observed_major} XML is outside the verified "
            f"{_SUPPORTED_PRODUCER_MAJOR} compatibility family.",
        )
    return ()


class ComputeSanitizerExtractor:
    name = "compute-sanitizer.xml"
    version = "1"
    compatibility_family = "compute-sanitizer.xml.2026.v1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> ComputeSanitizerExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registrations = tuple(
            item for item in run.artifacts if item.kind is ArtifactKind.SANITIZER_REPORT
        )
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one Compute Sanitizer XML report.",
                run_id=run_id,
            )
        registration = registrations[0]
        if registration.producer not in {"compute-sanitizer", "flameox.import"}:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The sanitizer report is not registered as Compute Sanitizer output.",
                run_id=run_id,
            )
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        maximum = self.workspace.config.storage.max_rows_per_generation
        if maximum < 2:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Compute Sanitizer extraction requires room for provenance and findings.",
            )
        inspection = inspect_compute_sanitizer_report(
            self.workspace,
            str(artifact.payload_path),
            max_records=maximum - 1,
        )
        records = inspection.records
        classifications = inspection.classifications
        limitations = [
            *compute_sanitizer_compatibility_limitations(registration.producer_version),
            *inspection.limitations,
        ]
        schema_fingerprint = digest_model(
            {
                "compatibility_family": self.compatibility_family,
                "producer_version": registration.producer_version,
                "record_kinds": sorted(
                    {str(item.get("kind")) for item in records if item.get("kind") is not None}
                ),
                "classifications": sorted(classifications),
            }
        )
        rows = [
            self._finding_row(
                run_id,
                registration.artifact_id,
                index,
                record,
            )
            for index, record in enumerate(records)
        ]
        rows.append(
            {
                "observation_id": digest_model(
                    {
                        "artifact_id": registration.artifact_id,
                        "kind": "sanitizer.extraction",
                        "schema_fingerprint": schema_fingerprint,
                    }
                ),
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "kind": "sanitizer.extraction",
                "name": self.compatibility_family,
                "value_json": _json(
                    {
                        "classifications": classifications,
                        "finding_count": len(records),
                        "producer_version": registration.producer_version,
                        "schema_fingerprint": schema_fingerprint,
                    }
                ),
                "file": None,
                "line_from": None,
                "line_to": None,
                "context": "extractor_provenance",
                "evidence_level": "observed",
            }
        )
        published = self.publisher.publish_rows_idempotent(
            {"observations": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "compatibility_family": self.compatibility_family,
                "max_records": maximum - 1,
                "max_frames": 64,
                "producer_version": registration.producer_version,
                "schema_fingerprint": schema_fingerprint,
            },
        )
        inconclusive = bool(limitations) and not records
        return ComputeSanitizerExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer_version=registration.producer_version,
            status=("findings" if records else "inconclusive" if inconclusive else "clean"),
            finding_count=len(records),
            classifications={str(key): int(value) for key, value in classifications.items()},
            schema_fingerprint=schema_fingerprint,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(dict.fromkeys(limitations)),
        )

    @staticmethod
    def _finding_row(
        run_id: str,
        artifact_id: str,
        index: int,
        record: dict[str, Any],
    ) -> dict[str, object]:
        path = record.get("path")
        line = record.get("line")
        return {
            "observation_id": digest_model(
                {
                    "artifact_id": artifact_id,
                    "record_index": index,
                    "record": record,
                }
            ),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "kind": "sanitizer.finding",
            "name": str(record.get("classification", "unknown")),
            "value_json": _json(record),
            "file": path if isinstance(path, str) else None,
            "line_from": line if isinstance(line, int) else None,
            "line_to": line if isinstance(line, int) else None,
            "context": record.get("function") if isinstance(record.get("function"), str) else None,
            "evidence_level": "observed",
        }


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
