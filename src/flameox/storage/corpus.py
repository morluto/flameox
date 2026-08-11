from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    ModelWrapValidatorHandler,
    StringConstraints,
    ValidationError,
    computed_field,
    model_validator,
)

from flameox.atomic import atomic_write_json, atomic_write_text
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import utc_now
from flameox.models import ContractModel

Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class GenerationFile(ContractModel):
    path: str
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    byte_length: Annotated[int, Field(ge=0)]
    row_count: Annotated[int, Field(ge=0)]
    table: str
    schema_major: Annotated[int, Field(ge=1)]
    schema_minor: Annotated[int, Field(ge=0)]


class GenerationManifest(ContractModel):
    schema_version: Literal[1] = 1
    generation_id: str
    created_at: datetime
    input_corpus_commit_id: Digest
    input_run_ids: tuple[str, ...] = ()
    input_artifact_ids: tuple[Digest, ...] = ()
    publisher: str
    publisher_version: str
    operation_digest: Digest | None = None
    files: tuple[GenerationFile, ...]
    supersedes: tuple[str, ...] = ()


class _CorpusCommitIntegrityError(ValueError):
    """A persisted commit projection contradicts its canonical content."""


class CorpusCommit(ContractModel):
    schema_version: Literal[1] = 1
    parent_commit_id: Digest | None
    created_at: datetime
    generation_manifests: tuple[str, ...]

    @model_validator(mode="wrap")
    @classmethod
    def parse_digest_projections(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler[CorpusCommit],
    ) -> CorpusCommit:
        if not isinstance(value, Mapping):
            return handler(value)
        payload = dict(value)
        supplied_inventory = payload.pop("inventory_digest", None)
        supplied_commit = payload.pop("commit_id", None)
        commit = handler(payload)
        if supplied_inventory is not None and supplied_inventory != commit.inventory_digest:
            raise _CorpusCommitIntegrityError(
                "corpus inventory digest does not match its generation manifests"
            )
        if supplied_commit is not None and supplied_commit != commit.commit_id:
            raise _CorpusCommitIntegrityError("corpus commit digest does not match its content")
        return commit

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inventory_digest(self) -> Digest:
        return digest_model({"generation_manifests": self.generation_manifests})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def commit_id(self) -> Digest:
        return digest_model(self.content_without_id())

    def content_without_id(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"commit_id"})


def build_commit(
    *,
    parent_commit_id: str | None,
    generation_manifests: tuple[str, ...],
    created_at: datetime | None = None,
) -> CorpusCommit:
    generation_manifests = tuple(sorted(set(generation_manifests)))
    timestamp = created_at or utc_now()
    return CorpusCommit(
        parent_commit_id=parent_commit_id,
        created_at=timestamp,
        generation_manifests=generation_manifests,
    )


class CorpusStore:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.corpus_root = workspace_root / "corpus"
        self.commits_root = self.corpus_root / "commits"
        self.head_path = self.corpus_root / "HEAD"

    def initialize(self) -> CorpusCommit:
        self.commits_root.mkdir(parents=True, exist_ok=True)
        if self.head_path.exists():
            return self.read_head()
        commit = build_commit(parent_commit_id=None, generation_manifests=())
        self.write_commit(commit)
        self.publish_head(commit.commit_id)
        return commit

    def commit_path(self, commit_id: str) -> Path:
        digest = commit_id.removeprefix("sha256:")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Invalid corpus commit identifier.")
        return self.commits_root / f"{digest}.json"

    def write_commit(self, commit: CorpusCommit) -> None:
        path = self.commit_path(commit.commit_id)
        if path.exists():
            existing = self.read_commit(commit.commit_id)
            if existing != commit:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "A different corpus commit already uses this identifier.",
                )
            return
        atomic_write_json(path, commit.model_dump(mode="json"))

    def publish_head(self, commit_id: str) -> None:
        if not self.commit_path(commit_id).is_file():
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Cannot publish a corpus HEAD whose commit is missing.",
            )
        atomic_write_text(self.head_path, f"{commit_id}\n")

    def read_head(self) -> CorpusCommit:
        try:
            commit_id = self.head_path.read_text().strip()
        except FileNotFoundError as exc:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Corpus HEAD is missing.") from exc
        return self.read_commit(commit_id)

    def read_commit(self, commit_id: str) -> CorpusCommit:
        path = self.commit_path(commit_id)
        try:
            payload = json.loads(path.read_text())
            commit = CorpusCommit.model_validate(payload)
        except ValidationError as exc:
            integrity_failure = any(
                isinstance(error.get("ctx", {}).get("error"), _CorpusCommitIntegrityError)
                for error in exc.errors(include_url=False)
            )
            raise DomainError(
                (
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED
                    if integrity_failure
                    else ErrorCode.WORKSPACE_INVALID
                ),
                f"Corpus commit {commit_id!r} is missing or invalid.",
            ) from exc
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Corpus commit {commit_id!r} is missing or invalid.",
            ) from exc
        return commit
