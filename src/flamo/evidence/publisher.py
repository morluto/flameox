from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from flamo.domain.errors import DomainError, ErrorCode
from flamo.domain.models import utc_now
from flamo.evidence.schemas import SCHEMA_MAJOR, SCHEMA_MINOR, schema_for
from flamo.storage.atomic import atomic_write_json, fsync_directory
from flamo.storage.corpus import (
    CorpusCommit,
    GenerationFile,
    GenerationManifest,
    build_commit,
)
from flamo.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class PublishedGeneration:
    manifest: GenerationManifest
    commit: CorpusCommit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GenerationPublisher:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def publish_rows(
        self,
        rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        publisher: str,
        publisher_version: str,
        input_run_ids: tuple[str, ...] = (),
        input_artifact_ids: tuple[str, ...] = (),
        supersedes: tuple[str, ...] = (),
        expected_head: str | None = None,
    ) -> PublishedGeneration:
        attempts = 1 if expected_head is not None else 32
        last_error: DomainError | None = None
        for _ in range(attempts):
            try:
                return self._publish_once(
                    rows_by_table,
                    publisher=publisher,
                    publisher_version=publisher_version,
                    input_run_ids=input_run_ids,
                    input_artifact_ids=input_artifact_ids,
                    supersedes=supersedes,
                    expected_head=expected_head,
                )
            except DomainError as error:
                if error.code is not ErrorCode.REVISION_CONFLICT:
                    raise
                last_error = error
                staging = error.details.get("staging_path")
                if isinstance(staging, str):
                    path = Path(staging)
                    try:
                        path.relative_to(self.workspace.paths.staging)
                    except ValueError:
                        pass
                    else:
                        shutil.rmtree(path, ignore_errors=True)
        assert last_error is not None
        raise last_error

    def _publish_once(
        self,
        rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        publisher: str,
        publisher_version: str,
        input_run_ids: tuple[str, ...] = (),
        input_artifact_ids: tuple[str, ...] = (),
        supersedes: tuple[str, ...] = (),
        expected_head: str | None = None,
    ) -> PublishedGeneration:
        initial_head = self.workspace.corpus.read_head()
        if expected_head is not None and expected_head != initial_head.commit_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The corpus changed before generation staging began.",
                retryable=True,
            )
        generation_id = str(uuid4())
        published_at = utc_now()
        staging_root = self.workspace.paths.staging / generation_id
        staging_root.mkdir(parents=True, exist_ok=False)
        final_files: list[GenerationFile] = []
        staged_to_final: list[tuple[Path, Path]] = []

        for table_name, rows in sorted(rows_by_table.items()):
            schema = schema_for(table_name)
            augmented = [
                {
                    "schema_version": SCHEMA_MAJOR,
                    "evidence_generation_id": generation_id,
                    "published_at": published_at,
                    "extractor_name": publisher,
                    "extractor_version": publisher_version,
                    **dict(row),
                }
                for row in rows
            ]
            table = pa.Table.from_pylist(augmented, schema=schema)
            if not table.schema.equals(schema, check_metadata=True):
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    f"Evidence table {table_name!r} does not match its Arrow schema.",
                )
            staged_path = staging_root / f"{table_name}.parquet"
            pq.write_table(
                table,
                staged_path,
                compression="zstd",
                version="2.6",
                write_statistics=True,
            )
            with staged_path.open("rb") as stream:
                os.fsync(stream.fileno())
            metadata = pq.read_metadata(staged_path)
            if metadata.num_rows != len(rows):
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    f"Parquet row count changed while writing {table_name!r}.",
                )
            final_relative = (
                Path("evidence") / table_name / f"generation={generation_id}" / "part-00000.parquet"
            )
            final_files.append(
                GenerationFile(
                    path=final_relative.as_posix(),
                    sha256=_file_sha256(staged_path),
                    byte_length=staged_path.stat().st_size,
                    row_count=metadata.num_rows,
                    table=table_name,
                    schema_major=SCHEMA_MAJOR,
                    schema_minor=SCHEMA_MINOR,
                )
            )
            staged_to_final.append((staged_path, self.workspace.paths.root / final_relative))

        final_manifest_relative = Path("generations") / generation_id / "manifest.json"
        manifest = GenerationManifest(
            generation_id=generation_id,
            created_at=published_at,
            input_corpus_commit_id=initial_head.commit_id,
            input_run_ids=input_run_ids,
            input_artifact_ids=input_artifact_ids,
            publisher=publisher,
            publisher_version=publisher_version,
            files=tuple(final_files),
            supersedes=supersedes,
        )
        staged_manifest = staging_root / "manifest.json"
        atomic_write_json(staged_manifest, manifest.model_dump(mode="json"))
        GenerationManifest.model_validate(json.loads(staged_manifest.read_text()))

        with self.workspace.write_locked():
            current_head = self.workspace.corpus.read_head()
            if current_head.commit_id != initial_head.commit_id:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "The corpus changed while evidence was staged.",
                    retryable=True,
                    details={"staging_path": str(staging_root)},
                )

            for staged_path, final_path in staged_to_final:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if final_path.exists():
                    if _file_sha256(final_path) != _file_sha256(staged_path):
                        raise DomainError(
                            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                            f"Generation target already differs: {final_path}",
                        )
                    staged_path.unlink()
                else:
                    os.replace(staged_path, final_path)
                    fsync_directory(final_path.parent)

            final_manifest = self.workspace.paths.root / final_manifest_relative
            final_manifest.parent.mkdir(parents=True, exist_ok=True)
            if final_manifest.exists():
                existing = GenerationManifest.model_validate_json(final_manifest.read_text())
                if existing != manifest:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                        "Generation manifest target already contains different data.",
                    )
            else:
                os.replace(staged_manifest, final_manifest)
                fsync_directory(final_manifest.parent)

            retained = self._without_superseded(
                current_head.generation_manifests,
                supersedes,
            )
            manifest_paths = (*retained, final_manifest_relative.as_posix())
            commit = build_commit(
                parent_commit_id=current_head.commit_id,
                generation_manifests=manifest_paths,
            )
            self.workspace.corpus.write_commit(commit)
            self.workspace.corpus.publish_head(commit.commit_id)

        shutil.rmtree(staging_root, ignore_errors=True)
        return PublishedGeneration(manifest=manifest, commit=commit)

    def _without_superseded(
        self,
        manifest_paths: tuple[str, ...],
        supersedes: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not supersedes:
            return manifest_paths
        superseded = set(supersedes)
        retained: list[str] = []
        for manifest_path in manifest_paths:
            path = self.workspace.paths.root / manifest_path
            manifest = GenerationManifest.model_validate_json(path.read_text())
            if manifest.generation_id not in superseded:
                retained.append(manifest_path)
        missing = superseded - {
            GenerationManifest.model_validate_json(
                (self.workspace.paths.root / path).read_text()
            ).generation_id
            for path in manifest_paths
        }
        if missing:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Cannot supersede unknown evidence generations.",
                details={"generation_ids": sorted(missing)},
            )
        return tuple(retained)
