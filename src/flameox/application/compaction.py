from __future__ import annotations

from flameox.catalog import Catalog
from flameox.evidence import GenerationPublisher, schema_for, table_names
from flameox.models import ContractModel
from flameox.storage import GenerationManifest, Workspace

_COMMON_COLUMNS = {
    "schema_version",
    "evidence_generation_id",
    "published_at",
    "extractor_name",
    "extractor_version",
}


class CompactionResult(ContractModel):
    schema_version: int = 1
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

    def compact(self) -> CompactionResult:
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
        rows_by_table: dict[str, list[dict[str, object]]] = {}
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            for table_name in table_names():
                schema = schema_for(table_name)
                columns = [name for name in schema.names if name not in _COMMON_COLUMNS]
                count_row = snapshot.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()
                if count_row is None or int(count_row[0]) == 0:
                    continue
                projection = ", ".join(f'"{column}"' for column in columns)
                rows_by_table[table_name] = (
                    snapshot.execute(f'SELECT {projection} FROM "{table_name}"')
                    .to_arrow_table()
                    .to_pylist()
                )
        published = GenerationPublisher(self.workspace).publish_rows(
            rows_by_table,
            publisher="flameox.compaction",
            publisher_version="1",
            supersedes=tuple(manifest.generation_id for manifest in manifests),
            expected_head=head.commit_id,
        )
        return CompactionResult(
            input_corpus_commit_id=head.commit_id,
            output_corpus_commit_id=published.commit.commit_id,
            superseded_generation_count=len(manifests),
            reachable_file_count_before=file_count_before,
            reachable_file_count_after=len(published.manifest.files),
            row_counts={table_name: len(rows) for table_name, rows in rows_by_table.items()},
        )
