from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from flameox.catalog import Catalog
from flameox.evidence import GenerationPublisher, schema_for, table_names
from flameox.models import ContractModel
from flameox.storage import GenerationManifest, Workspace
from flameox.storage.quotas import StorageQuota

_COMPACTION_BATCH_ROWS = 65_536


class CompactionResult(ContractModel):
    input_corpus_commit_id: str
    output_corpus_commit_id: str
    superseded_generation_count: int
    reachable_file_count_before: int
    reachable_file_count_after: int
    row_counts: dict[str, int]


class CompactionService:
    """Replace reachable small generations with one immutable generation."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def compact(self) -> CompactionResult:
        head = self.workspace.corpus.read_head()
        manifests = tuple(
            GenerationManifest.model_validate_json((self.workspace.paths.root / path).read_text())
            for path in head.generation_manifests
        )
        file_count_before = sum(len(manifest.files) for manifest in manifests)
        if len(manifests) < 2:
            return CompactionResult(
                input_corpus_commit_id=head.commit_id,
                output_corpus_commit_id=head.commit_id,
                superseded_generation_count=0,
                reachable_file_count_before=file_count_before,
                reachable_file_count_after=file_count_before,
                row_counts={},
            )
        row_counts: dict[str, int] = {}
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            for table_name in table_names():
                count_row = snapshot.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()
                if count_row is None or int(count_row[0]) == 0:
                    continue
                row_counts[table_name] = int(count_row[0])
            StorageQuota(self.workspace).require_generation_row_count(sum(row_counts.values()))

            if not row_counts:
                return CompactionResult(
                    input_corpus_commit_id=head.commit_id,
                    output_corpus_commit_id=head.commit_id,
                    superseded_generation_count=0,
                    reachable_file_count_before=file_count_before,
                    reachable_file_count_after=file_count_before,
                    row_counts={},
                )

            def prepare(
                staging_root: Path,
                _generation_id: str,
                _published_at: datetime,
            ) -> dict[str, Path]:
                staged: dict[str, Path] = {}
                for table_name in sorted(row_counts):
                    schema = schema_for(table_name)
                    projection = ", ".join(f'"{name}"' for name in schema.names)
                    reader = snapshot.execute(
                        f'SELECT {projection} FROM "{table_name}"'
                    ).to_arrow_reader(_COMPACTION_BATCH_ROWS)
                    staged_path = staging_root / f"{table_name}.parquet"
                    with pq.ParquetWriter(
                        staged_path,
                        schema,
                        compression="zstd",
                        version="2.6",
                        write_statistics=True,
                    ) as writer:
                        for batch in reader:
                            writer.write_batch(
                                batch.cast(schema),
                                row_group_size=_COMPACTION_BATCH_ROWS,
                            )
                    with staged_path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    staged[table_name] = staged_path
                return staged

            published = await GenerationPublisher(self.workspace).publish_prepared_parquet(
                prepare,
                publisher="flameox.compaction",
                publisher_version="2",
                operation_digest=head.inventory_digest,
                supersedes=tuple(manifest.generation_id for manifest in manifests),
                expected_head=head.commit_id,
            )
        return CompactionResult(
            input_corpus_commit_id=head.commit_id,
            output_corpus_commit_id=published.commit.commit_id,
            superseded_generation_count=len(manifests),
            reachable_file_count_before=file_count_before,
            reachable_file_count_after=len(published.manifest.files),
            row_counts=row_counts,
        )
