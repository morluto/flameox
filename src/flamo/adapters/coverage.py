from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from flamo.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    RunManifest,
    digest_model,
)
from flamo.evidence import GenerationPublisher
from flamo.storage import ArtifactStore, RunStore, Workspace


class CoverageExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    run_id: str
    artifact_id: str
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
        try:
            from coverage import CoverageData
        except ImportError as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "coverage.py is not installed.",
                remediation=("Install Flamo's execution optional dependencies.",),
            ) from exc
        run = RunStore(self.workspace).read(run_id)
        registration = self._registration(run)
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        data = CoverageData(basename=str(artifact.payload_path))
        try:
            data.read()
            measured_files = sorted(data.measured_files())
        except Exception as exc:
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
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
        )
        return CoverageExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            file_count=len(measured_files),
            line_count=line_count,
            arc_count=arc_count,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    def _registration(self, run: RunManifest) -> Any:
        matches = [item for item in run.artifacts if item.kind is ArtifactKind.EXECUTION_COVERAGE]
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one coverage.py artifact.",
                run_id=run.run_id,
            )
        return matches[0]

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
