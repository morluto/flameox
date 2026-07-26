from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar, Literal, cast

from pydantic import JsonValue

from flameox.application.artifacts import ArtifactService
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel
from flameox.storage import GenerationManifest, RunStore, Workspace


class EvidenceLookupResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    ref_type: Literal[
        "analysis",
        "artifact",
        "comparison",
        "generation",
        "observation",
        "run",
        "run_set",
        "trial",
    ]
    ref_id: str
    data: dict[str, JsonValue]


class EvidenceLookupService:
    _TABLES: ClassVar[dict[str, tuple[str, str]]] = {
        "analysis": ("analyses", "analysis_id"),
        "comparison": ("comparisons", "comparison_id"),
        "observation": ("observations", "observation_id"),
        "run_set": ("run_sets", "run_set_id"),
        "trial": ("trials", "trial_id"),
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def get(
        self,
        ref_type: Literal[
            "analysis",
            "artifact",
            "comparison",
            "generation",
            "observation",
            "run",
            "run_set",
            "trial",
        ],
        ref_id: str,
    ) -> EvidenceLookupResult:
        head = self.workspace.corpus.read_head()
        if ref_type == "run":
            data = RunStore(self.workspace).read(ref_id).model_dump(mode="json")
        elif ref_type == "artifact":
            data = ArtifactService(self.workspace).get(ref_id).model_dump(mode="json")
        elif ref_type == "generation":
            path = self.workspace.paths.generations / ref_id / "manifest.json"
            try:
                data = GenerationManifest.model_validate_json(path.read_text()).model_dump(
                    mode="json"
                )
            except (OSError, ValueError) as exc:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Generation {ref_id!r} does not exist or is invalid.",
                ) from exc
        else:
            table, identifier = self._TABLES[ref_type]
            with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
                connection = snapshot.execute(
                    f'SELECT * FROM "{table}" WHERE "{identifier}" = ? '
                    "ORDER BY published_at DESC LIMIT 1",
                    (ref_id,),
                )
                row = connection.fetchone()
                columns = [item[0] for item in connection.description]
            if row is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"{ref_type} evidence {ref_id!r} does not exist.",
                )
            data = {name: _json_value(value) for name, value in zip(columns, row, strict=True)}
        return EvidenceLookupResult(
            corpus_commit_id=head.commit_id,
            ref_type=ref_type,
            ref_id=ref_id,
            data=cast(dict[str, JsonValue], data),
        )


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return cast(JsonValue, value.value)
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
