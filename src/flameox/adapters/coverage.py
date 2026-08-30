from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    DomainError,
    ErrorCode,
    RunManifest,
    digest_model,
    missing_artifact_input,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

_CONTROL_READER_REQUIREMENT = Requirement("coverage>=7.14,<8")
_COVERAGE_PRODUCER = "coverage"


def qualified_control_coverage_reader_version() -> str:
    """Return the installed control-reader version if it supports coverage data extraction."""

    try:
        reader_version = Version(version(_CONTROL_READER_REQUIREMENT.name))
    except PackageNotFoundError as exc:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "The Flameox control environment has no coverage.py reader.",
            details={"reader_requirement": str(_CONTROL_READER_REQUIREMENT)},
            remediation=(
                "Install or update Flameox so its bundled coverage.py reader is available, then "
                "plan the coverage capture again.",
            ),
        ) from exc
    except InvalidVersion as exc:
        raise DomainError(
            ErrorCode.ADAPTER_INCOMPATIBLE,
            "The Flameox control coverage.py reader has an invalid version.",
            details={"reader_requirement": str(_CONTROL_READER_REQUIREMENT)},
            remediation=(
                "Install Flameox with a supported coverage.py reader, then plan the coverage "
                "capture again.",
            ),
        ) from exc
    if not _CONTROL_READER_REQUIREMENT.specifier.contains(reader_version, prereleases=True):
        raise DomainError(
            ErrorCode.ADAPTER_INCOMPATIBLE,
            "The Flameox control coverage.py reader is outside the supported data-reader range.",
            details={
                "reader_version": str(reader_version),
                "reader_requirement": str(_CONTROL_READER_REQUIREMENT),
            },
            remediation=(
                "Install Flameox with a supported coverage.py reader, then plan the coverage "
                "capture again.",
            ),
        )
    return str(reader_version)


class CoverageExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    producer: str
    producer_version: str
    reader_version: str
    file_count: int
    line_count: int
    arc_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class CoverageExtractor:
    name = "coverage"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> CoverageExtractionResult:
        reader_version = qualified_control_coverage_reader_version()
        try:
            from coverage import CoverageData
            from coverage.exceptions import DataError
        except ImportError as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "coverage.py is not installed.",
                remediation=("Install flameox's execution optional dependencies.",),
            ) from exc
        run = RunStore(self.workspace).read(run_id)
        registration = self._registration(run)
        producer_version = self._require_compatible_producer(registration, reader_version, run_id)
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        data = CoverageData(basename=str(artifact.payload_path))
        try:
            data.read()
            measured_files = sorted(data.measured_files())
        except (DataError, OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported coverage.py data file.",
                run_id=run_id,
            ) from exc
        rows: list[dict[str, Any]] = []
        limitations: list[str] = []
        line_count = 0
        arc_count = 0
        maximum = self.workspace.config.storage.max_rows_per_generation
        for filename in measured_files:
            normalized = self._normalize(filename)
            if normalized is None:
                limitations.append(f"Skipped coverage outside the project root: {filename}")
                continue
            contexts = data.contexts_by_lineno(filename)
            for line in sorted(data.lines(filename) or ()):
                line_contexts = sorted(contexts.get(line) or ("",))
                for context in line_contexts:
                    rows.append(
                        self._row(
                            run_id=run_id,
                            artifact_id=registration.artifact_id,
                            kind="line_hit",
                            name="coverage.line",
                            value=True,
                            filename=normalized,
                            line_from=line,
                            line_to=None,
                            context=context or None,
                        )
                    )
                    line_count += 1
                    self._require_budget(rows, maximum)
            if data.has_arcs():
                for line_from, line_to in sorted(data.arcs(filename) or ()):
                    rows.append(
                        self._row(
                            run_id=run_id,
                            artifact_id=registration.artifact_id,
                            kind="branch_arc",
                            name="coverage.arc",
                            value=True,
                            filename=normalized,
                            line_from=line_from,
                            line_to=line_to,
                            context=None,
                        )
                    )
                    arc_count += 1
                    self._require_budget(rows, maximum)
        published = self.publisher.publish_rows(
            {"observations": rows},
            publisher=self.name,
            publisher_version=reader_version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
        )
        return CoverageExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer=_COVERAGE_PRODUCER,
            producer_version=producer_version,
            reader_version=reader_version,
            file_count=len(measured_files),
            line_count=line_count,
            arc_count=arc_count,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    def _registration(self, run: RunManifest) -> ArtifactRegistration:
        matches = [item for item in run.artifacts if item.kind is ArtifactKind.EXECUTION_COVERAGE]
        if not matches:
            raise missing_artifact_input(
                run_id=run.run_id,
                requirement="coverage.py execution-coverage",
                artifact_kinds=(ArtifactKind.EXECUTION_COVERAGE.value,),
                capture_adapters=("coverage",),
            )
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one coverage.py artifact.",
                run_id=run.run_id,
            )
        return matches[0]

    @staticmethod
    def _require_compatible_producer(
        registration: ArtifactRegistration,
        reader_version: str,
        run_id: str,
    ) -> str:
        details = {
            "producer": registration.producer,
            "producer_version": registration.producer_version,
            "reader": _COVERAGE_PRODUCER,
            "reader_version": reader_version,
            "supported_requirement": str(_CONTROL_READER_REQUIREMENT),
        }
        if registration.producer != _COVERAGE_PRODUCER:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The coverage artifact does not identify coverage.py as its producer.",
                run_id=run_id,
                details=details,
                remediation=(
                    "Recapture with coverage>=7.14,<8 in the declared workload interpreter, "
                    "then extract the new run; Flameox cannot qualify an artifact without its "
                    "producer identity.",
                ),
            )
        if registration.producer_version is None:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The coverage artifact does not identify its coverage.py producer version.",
                run_id=run_id,
                details=details,
                remediation=(
                    "Recapture with coverage>=7.14,<8 so the run records the workload producer "
                    "version before extraction.",
                ),
            )
        try:
            producer_version = Version(registration.producer_version)
        except InvalidVersion as exc:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The coverage artifact has an invalid coverage.py producer version.",
                run_id=run_id,
                details=details,
                remediation=(
                    "Recapture with a supported coverage.py distribution and extract the new "
                    "run; do not relabel the existing artifact.",
                ),
            ) from exc
        if not _CONTROL_READER_REQUIREMENT.specifier.contains(producer_version, prereleases=True):
            details["producer_version"] = str(producer_version)
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The coverage artifact was produced by an unsupported coverage.py version.",
                run_id=run_id,
                details=details,
                remediation=(
                    "Recapture with coverage>=7.14,<8 in the declared workload interpreter, "
                    "then extract the new run; the native artifact remains preserved but is "
                    "not normalized by this reader.",
                ),
            )
        return str(producer_version)

    def _normalize(self, filename: str) -> str | None:
        path = Path(filename).resolve()
        try:
            return path.relative_to(self.workspace.project_root).as_posix()
        except ValueError:
            return None

    def _row(
        self,
        *,
        run_id: str,
        artifact_id: str,
        kind: str,
        name: str,
        value: bool,
        filename: str,
        line_from: int,
        line_to: int | None,
        context: str | None,
    ) -> dict[str, Any]:
        identity = {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "kind": kind,
            "file": filename,
            "line_from": line_from,
            "line_to": line_to,
            "context": context,
        }
        return {
            "observation_id": digest_model(identity),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "kind": kind,
            "name": name,
            "value_json": json.dumps(value),
            "file": filename,
            "line_from": line_from,
            "line_to": line_to,
            "context": context,
            "evidence_level": "observed",
        }

    def _require_budget(self, rows: list[dict[str, Any]], maximum: int) -> None:
        if len(rows) > maximum:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Coverage extraction exceeded the {maximum}-row generation limit.",
            )
