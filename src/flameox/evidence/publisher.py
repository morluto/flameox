from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from flameox.atomic import atomic_write_json, fsync_directory
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import utc_now
from flameox.evidence.schemas import SCHEMA_MAJOR, SCHEMA_MINOR, schema_for
from flameox.observability import OperationLogger, elapsed_ms
from flameox.storage.corpus import (
    CorpusCommit,
    GenerationFile,
    GenerationManifest,
    build_commit,
)
from flameox.storage.quotas import StorageQuota
from flameox.storage.workspace import Workspace


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


def publication_operation_digest(
    *,
    publisher: str,
    publisher_version: str,
    input_run_ids: tuple[str, ...],
    input_artifact_ids: tuple[str, ...],
    operation_identity: Mapping[str, Any],
) -> str:
    return digest_model(
        {
            "publisher": publisher,
            "publisher_version": publisher_version,
            "input_run_ids": input_run_ids,
            "input_artifact_ids": input_artifact_ids,
            "operation": dict(operation_identity),
        }
    )


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
        operation_digest: str | None = None,
        supersedes: tuple[str, ...] = (),
        expected_head: str | None = None,
    ) -> PublishedGeneration:
        quota = StorageQuota(self.workspace)
        quota.require_generation_rows(rows_by_table)
        quota.require_capacity(staging=True)
        attempts = 1 if expected_head is not None else 32
        last_error: DomainError | None = None
        for _ in range(attempts):
            generation_id = str(uuid4())
            try:
                return self._publish_once(
                    rows_by_table,
                    generation_id=generation_id,
                    publisher=publisher,
                    publisher_version=publisher_version,
                    input_run_ids=input_run_ids,
                    input_artifact_ids=input_artifact_ids,
                    operation_digest=operation_digest,
                    supersedes=supersedes,
                    expected_head=expected_head,
                )
            except BaseException as error:
                shutil.rmtree(
                    self.workspace.paths.staging / generation_id,
                    ignore_errors=True,
                )
                if (
                    not isinstance(error, DomainError)
                    or error.code is not ErrorCode.REVISION_CONFLICT
                ):
                    raise
                last_error = error
        assert last_error is not None
        raise last_error

    def publish_rows_idempotent(
        self,
        rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        publisher: str,
        publisher_version: str,
        input_run_ids: tuple[str, ...] = (),
        input_artifact_ids: tuple[str, ...] = (),
        operation_identity: Mapping[str, Any],
        supersede_matching: bool = True,
    ) -> PublishedGeneration:
        operation_digest = publication_operation_digest(
            publisher=publisher,
            publisher_version=publisher_version,
            input_run_ids=input_run_ids,
            input_artifact_ids=input_artifact_ids,
            operation_identity=operation_identity,
        )
        expected_tables = set(rows_by_table)
        last_conflict: DomainError | None = None
        for _ in range(32):
            head = self.workspace.corpus.read_head()
            matching: list[GenerationManifest] = []
            for relative_path in head.generation_manifests:
                manifest = GenerationManifest.model_validate_json(
                    (self.workspace.paths.root / relative_path).read_text()
                )
                if (
                    manifest.publisher == publisher
                    and manifest.input_run_ids == input_run_ids
                    and manifest.input_artifact_ids == input_artifact_ids
                ):
                    matching.append(manifest)
            for manifest in matching:
                if (
                    manifest.operation_digest == operation_digest
                    and {item.table for item in manifest.files} == expected_tables
                ):
                    return PublishedGeneration(manifest=manifest, commit=head)
            try:
                return self.publish_rows(
                    rows_by_table,
                    publisher=publisher,
                    publisher_version=publisher_version,
                    input_run_ids=input_run_ids,
                    input_artifact_ids=input_artifact_ids,
                    operation_digest=operation_digest,
                    supersedes=(
                        tuple(manifest.generation_id for manifest in matching)
                        if supersede_matching
                        else ()
                    ),
                    expected_head=head.commit_id,
                )
            except DomainError as error:
                if error.code is not ErrorCode.REVISION_CONFLICT:
                    raise
                last_conflict = error
        assert last_conflict is not None
        raise last_conflict

    def publish_prepared_parquet(
        self,
        prepare: Callable[[Path, str, datetime], Mapping[str, Path]],
        *,
        publisher: str,
        publisher_version: str,
        input_run_ids: tuple[str, ...] = (),
        input_artifact_ids: tuple[str, ...] = (),
        operation_digest: str | None = None,
        supersedes: tuple[str, ...] = (),
        expected_head: str | None = None,
    ) -> PublishedGeneration:
        """Publish producer-prepared Parquet without materializing rows in Python.

        The callback may write only beneath the supplied staging root. This class
        remains responsible for schema/integrity validation and atomic visibility.
        """
        StorageQuota(self.workspace).require_capacity(staging=True)
        attempts = 1 if expected_head is not None else 32
        last_error: DomainError | None = None
        for _ in range(attempts):
            generation_id = str(uuid4())
            staging_root = self.workspace.paths.staging / generation_id
            try:
                initial_head = self.workspace.corpus.read_head()
                if expected_head is not None and expected_head != initial_head.commit_id:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "The corpus changed before generation staging began.",
                        retryable=True,
                    )
                published_at = utc_now()
                staging_root.mkdir(parents=True, exist_ok=False)
                staged = prepare(staging_root, generation_id, published_at)
                return self._publish_staged_generation(
                    staged,
                    generation_id=generation_id,
                    initial_head=initial_head,
                    published_at=published_at,
                    publisher=publisher,
                    publisher_version=publisher_version,
                    input_run_ids=input_run_ids,
                    input_artifact_ids=input_artifact_ids,
                    operation_digest=operation_digest,
                    supersedes=supersedes,
                    expected_row_counts=None,
                )
            except BaseException as error:
                shutil.rmtree(staging_root, ignore_errors=True)
                if (
                    not isinstance(error, DomainError)
                    or error.code is not ErrorCode.REVISION_CONFLICT
                ):
                    raise
                last_error = error
        assert last_error is not None
        raise last_error

    def _publish_once(
        self,
        rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        generation_id: str,
        publisher: str,
        publisher_version: str,
        input_run_ids: tuple[str, ...] = (),
        input_artifact_ids: tuple[str, ...] = (),
        operation_digest: str | None = None,
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
        published_at = utc_now()
        staging_root = self.workspace.paths.staging / generation_id
        staging_root.mkdir(parents=True, exist_ok=False)
        staged: dict[str, Path] = {}

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
            staged[table_name] = staged_path

        return self._publish_staged_generation(
            staged,
            generation_id=generation_id,
            initial_head=initial_head,
            published_at=published_at,
            publisher=publisher,
            publisher_version=publisher_version,
            input_run_ids=input_run_ids,
            input_artifact_ids=input_artifact_ids,
            operation_digest=operation_digest,
            supersedes=supersedes,
            expected_row_counts={name: len(rows) for name, rows in rows_by_table.items()},
        )

    def _publish_staged_generation(
        self,
        staged: Mapping[str, Path],
        *,
        generation_id: str,
        initial_head: CorpusCommit,
        published_at: datetime,
        publisher: str,
        publisher_version: str,
        input_run_ids: tuple[str, ...],
        input_artifact_ids: tuple[str, ...],
        operation_digest: str | None,
        supersedes: tuple[str, ...],
        expected_row_counts: Mapping[str, int] | None,
    ) -> PublishedGeneration:
        quota = StorageQuota(self.workspace)
        operation_id = OperationLogger(self.workspace.paths.root).new_id()
        started = time.monotonic()
        staging_root = self.workspace.paths.staging / generation_id
        final_files: list[GenerationFile] = []
        staged_to_final: list[tuple[Path, Path]] = []
        total_rows = 0
        for table_name, staged_path in sorted(staged.items()):
            schema = schema_for(table_name)
            try:
                metadata = staged_path.lstat()
                resolved = staged_path.resolve(strict=True)
                resolved.relative_to(staging_root.resolve(strict=True))
            except (FileNotFoundError, OSError, ValueError) as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    f"Prepared evidence file for {table_name!r} escaped its staging root.",
                ) from error
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    f"Prepared evidence file for {table_name!r} is not an owned regular file.",
                )
            parquet_schema = pq.read_schema(resolved)
            if (
                not parquet_schema.equals(schema, check_metadata=False)
                or parquet_schema.metadata != schema.metadata
            ):
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    f"Prepared evidence table {table_name!r} does not match its Arrow schema.",
                )
            parquet_metadata = pq.read_metadata(resolved)
            expected_rows = (
                expected_row_counts.get(table_name) if expected_row_counts is not None else None
            )
            if expected_rows is not None and parquet_metadata.num_rows != expected_rows:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    f"Parquet row count changed while writing {table_name!r}.",
                )
            total_rows += parquet_metadata.num_rows
            quota.require_capacity(staging=True)
            final_relative = (
                Path("evidence") / table_name / f"generation={generation_id}" / "part-00000.parquet"
            )
            final_files.append(
                GenerationFile(
                    path=final_relative.as_posix(),
                    sha256=_file_sha256(resolved),
                    byte_length=metadata.st_size,
                    row_count=parquet_metadata.num_rows,
                    table=table_name,
                    schema_major=SCHEMA_MAJOR,
                    schema_minor=SCHEMA_MINOR,
                )
            )
            staged_to_final.append((resolved, self.workspace.paths.root / final_relative))
        quota.require_generation_row_count(total_rows)

        final_manifest_relative = Path("generations") / generation_id / "manifest.json"
        manifest = GenerationManifest(
            generation_id=generation_id,
            created_at=published_at,
            input_corpus_commit_id=initial_head.commit_id,
            input_run_ids=input_run_ids,
            input_artifact_ids=input_artifact_ids,
            publisher=publisher,
            publisher_version=publisher_version,
            operation_digest=operation_digest,
            files=tuple(final_files),
            supersedes=supersedes,
        )
        staged_manifest = staging_root / "manifest.json"
        atomic_write_json(staged_manifest, manifest.model_dump(mode="json"))
        GenerationManifest.model_validate(json.loads(staged_manifest.read_text()))

        lock_started = time.monotonic()
        with self.workspace.write_locked():
            lock_wait_ms = elapsed_ms(lock_started)
            quota.require_capacity(staging=True)
            current_head = self.workspace.corpus.read_head()
            if current_head.commit_id != initial_head.commit_id:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "The corpus changed while evidence was staged.",
                    retryable=True,
                    details={"staging_path": str(staging_root)},
                )

            final_manifest = self.workspace.paths.root / final_manifest_relative
            retained = self._without_superseded(
                current_head.generation_manifests,
                supersedes,
            )
            manifest_paths = (*retained, final_manifest_relative.as_posix())
            commit = build_commit(
                parent_commit_id=current_head.commit_id,
                generation_manifests=manifest_paths,
            )
            placed_evidence: list[Path] = []
            placed_manifest: Path | None = None
            head_publish_attempted = False
            try:
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
                        placed_evidence.append(final_path)
                        os.replace(staged_path, final_path)
                        fsync_directory(final_path.parent)

                final_manifest.parent.mkdir(parents=True, exist_ok=True)
                if final_manifest.exists():
                    existing = GenerationManifest.model_validate_json(final_manifest.read_text())
                    if existing != manifest:
                        raise DomainError(
                            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                            "Generation manifest target already contains different data.",
                        )
                else:
                    placed_manifest = final_manifest
                    os.replace(staged_manifest, final_manifest)
                    fsync_directory(final_manifest.parent)

                self.workspace.corpus.write_commit(commit)
                head_publish_attempted = True
                self.workspace.corpus.publish_head(commit.commit_id)
            except BaseException:
                rollback = not head_publish_attempted
                if head_publish_attempted:
                    try:
                        rollback = self.workspace.corpus.read_head().commit_id != commit.commit_id
                    except DomainError:
                        # An unreadable HEAD cannot prove that publication failed.
                        # Preserve the immutable files rather than risk deleting
                        # evidence referenced by a successfully replaced HEAD.
                        rollback = False
                if rollback:
                    self._rollback_placement(placed_evidence, placed_manifest)
                raise

        shutil.rmtree(staging_root, ignore_errors=True)
        OperationLogger(self.workspace.paths.root).emit(
            operation_id=operation_id,
            operation="evidence.publish",
            phase="corpus HEAD published",
            adapter=publisher,
            elapsed_ms=elapsed_ms(started),
            lock_wait_ms=lock_wait_ms,
            rows_returned=total_rows,
            bytes_returned=sum(file.byte_length for file in final_files),
        )
        return PublishedGeneration(manifest=manifest, commit=commit)

    @staticmethod
    def _rollback_placement(
        placed_evidence: list[Path],
        placed_manifest: Path | None,
    ) -> None:
        for path in reversed(placed_evidence):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                path.parent.rmdir()
        if placed_manifest is not None:
            with contextlib.suppress(OSError):
                placed_manifest.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                placed_manifest.parent.rmdir()

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
