from __future__ import annotations

import json
from typing import Any

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


class ObservationExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    observation_count: int
    corpus_commit_id: str


class ObservationExtractor:
    name = "flameox.sdk"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> ObservationExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registrations = [
            item for item in run.artifacts if item.kind is ArtifactKind.SEMANTIC_OBSERVATIONS
        ]
        if not registrations:
            raise missing_artifact_input(
                run_id=run_id,
                requirement="semantic-observation",
                artifact_kinds=(ArtifactKind.SEMANTIC_OBSERVATIONS.value,),
                capture_adapters=("command",),
                import_producers=("auto",),
            )
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one semantic-observation artifact.",
                run_id=run_id,
            )
        registration = registrations[0]
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        rows: list[dict[str, Any]] = []
        maximum = self.workspace.config.storage.max_rows_per_generation
        try:
            with artifact.payload_path.open("rb") as stream:
                for index, encoded in enumerate(stream):
                    if len(encoded) > 16 * 1024:
                        raise ValueError("observation line exceeds 16 KiB")
                    event = json.loads(encoded)
                    self._validate_event(event)
                    values = event["values"]
                    rows.append(
                        {
                            "observation_id": digest_model(
                                {
                                    "artifact_id": registration.artifact_id,
                                    "index": index,
                                    "event": event,
                                }
                            ),
                            "run_id": run_id,
                            "artifact_id": registration.artifact_id,
                            "kind": "annotation",
                            "name": event["name"],
                            "value_json": json.dumps(
                                values,
                                allow_nan=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            "file": None,
                            "line_from": None,
                            "line_to": None,
                            "context": event["phase"],
                            "evidence_level": "observed",
                        }
                    )
                    if len(rows) > maximum:
                        raise DomainError(
                            ErrorCode.QUERY_BUDGET_EXCEEDED,
                            f"Observation extraction exceeded {maximum} rows.",
                        )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The semantic-observation artifact is invalid.",
                run_id=run_id,
            ) from exc
        published = self.publisher.publish_rows(
            {"observations": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
        )
        return ObservationExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            observation_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
        )

    def _validate_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            raise ValueError("event must be an object")
        if set(event) != {"name", "phase", "monotonic_ns", "values"}:
            raise ValueError("event fields differ from the SDK observation contract")
        if (
            not isinstance(event["name"], str)
            or not isinstance(event["monotonic_ns"], int)
            or (event["phase"] is not None and not isinstance(event["phase"], str))
        ):
            raise ValueError("event field types are invalid")
