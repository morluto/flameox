from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import (
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from flameox.atomic import atomic_write_json, atomic_write_text
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import utc_now
from flameox.models import ContractModel

Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GenerationFile(ContractModel):
    path: str
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    byte_length: Annotated[int, Field(ge=0)]
    row_count: Annotated[int, Field(ge=0)]
    table: str


class GenerationManifest(ContractModel):
    created_at: datetime
    input_corpus_commit_id: Digest
    input_run_ids: tuple[str, ...] = ()
    input_run_semantic_ids: tuple[Digest | None, ...]
    input_artifact_ids: tuple[Digest, ...] = ()
    publisher: str
    publisher_version: str
    operation_digest: Digest | None = None
    files: tuple[GenerationFile, ...]
    supersedes: tuple[Digest, ...] = ()

    @model_validator(mode="after")
    def run_semantic_ids_align_with_runs(self) -> GenerationManifest:
        if len(self.input_run_semantic_ids) != len(self.input_run_ids):
            raise ValueError("input run and semantic identities must be positionally aligned")
        return self

    @property
    def generation_id(self) -> Digest:
        """Content identity for this exact current manifest payload.

        The identifier is deliberately not serialized.  A persisted manifest is
        authoritative only when its strict payload hashes to the generation ID
        named by a corpus commit.
        """

        return digest_model(self.model_dump(mode="json"))


class CorpusCommit(ContractModel):
    parent_commit_id: Digest | None
    created_at: datetime
    generation_ids: tuple[Digest, ...]

    @model_validator(mode="after")
    def generation_ids_are_canonical(self) -> CorpusCommit:
        if self.generation_ids != tuple(sorted(set(self.generation_ids))):
            raise ValueError("generation IDs must be unique and sorted")
        return self

    @property
    def commit_id(self) -> Digest:
        return digest_model(self.model_dump(mode="json"))


def build_commit(
    *,
    parent_commit_id: str | None,
    generation_ids: tuple[str, ...],
    created_at: datetime | None = None,
) -> CorpusCommit:
    generation_ids = tuple(sorted(set(generation_ids)))
    timestamp = created_at or utc_now()
    return CorpusCommit(
        parent_commit_id=parent_commit_id,
        created_at=timestamp,
        generation_ids=generation_ids,
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
        commit = build_commit(parent_commit_id=None, generation_ids=())
        self.write_commit(commit)
        self.publish_head(commit.commit_id)
        return commit

    def commit_path(self, commit_id: str) -> Path:
        digest = self._digest_component(commit_id, kind="corpus commit")
        return self.commits_root / f"{digest}.json"

    def generation_path(self, generation_id: str) -> Path:
        digest = self._digest_component(generation_id, kind="generation")
        return self.workspace_root / "generations" / f"{digest}.json"

    @staticmethod
    def _digest_component(identifier: str, *, kind: str) -> str:
        digest = identifier.removeprefix("sha256:")
        if (
            not identifier.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, f"Invalid {kind} identifier.")
        return digest

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
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Corpus commit {commit_id!r} is missing or invalid.",
            ) from exc
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Corpus commit {commit_id!r} is missing or invalid.",
            ) from exc
        if commit.commit_id != commit_id:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Corpus commit {commit_id!r} does not match its content.",
            )
        return commit

    def read_generation(self, generation_id: str) -> GenerationManifest:
        path = self.generation_path(generation_id)
        try:
            manifest = GenerationManifest.model_validate_json(path.read_text())
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Generation {generation_id!r} is missing or invalid.",
            ) from exc
        if manifest.generation_id != generation_id:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Generation {generation_id!r} does not match its manifest content.",
            )
        return manifest
