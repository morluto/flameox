from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flameox.adapters.sarif import DEFAULT_EXCLUDE_PATHS, parse_sarif
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.workers.coverage_contract import COVERAGE_WORKER, CoverageWorkerRequest
from flameox.workers.harness import IsolatedWorkerHarness


class SourceEvidenceProvider:
    """Bounded projections for native coverage.py and SARIF artifacts."""

    def __init__(self, harness: IsolatedWorkerHarness, project_root: Path) -> None:
        self.harness = harness
        self.project_root = project_root

    def analyze(
        self,
        capability_id: str,
        path: Path,
        format_name: str,
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis | None:
        if capability_id == "coverage.summary" and format_name == "coverage":
            result = self.harness.run_typed_sync(
                COVERAGE_WORKER,
                CoverageWorkerRequest(
                    artifact_path=str(path),
                    project_root=str(self.project_root),
                    max_rows=max_rows,
                ),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            observed = result.line_count + result.arc_count
            return ProviderAnalysis(
                provider_id="coverage.py",
                provider_version=result.reader_version,
                blocks=[
                    {
                        "type": "metrics",
                        "values": {
                            "file_count": result.file_count,
                            "line_count": result.line_count,
                            "arc_count": result.arc_count,
                        },
                    },
                    {"type": "table", "rows": [dict(row) for row in result.rows]},
                ],
                rows_observed=observed,
                complete=not result.truncated,
                limitations=list(result.limitations),
            )
        if capability_id != "static.performance_candidates" or format_name != "sarif":
            return None
        parsed = parse_sarif(
            path,
            source_root=self.project_root,
            include_paths=tuple(arguments.get("include_paths", ())),
            exclude_paths=tuple(arguments.get("exclude_paths", ())),
            default_exclude_paths=DEFAULT_EXCLUDE_PATHS,
            maximum_candidates=max_rows,
        )
        if not parsed.supported:
            raise ProviderFailure("UNSUPPORTED_FORMAT", "Only SARIF 2.1.0 is supported")
        coverage = parsed.coverage
        rows = [asdict(candidate) for candidate in parsed.candidates]
        return ProviderAnalysis(
            provider_id="sarif",
            provider_version="2.1.0",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "result_count": coverage.result_count,
                        "normalized_count": coverage.normalized_count,
                        "excluded_count": coverage.excluded_count,
                        "invalid_count": coverage.invalid_count,
                        "omitted_count": coverage.omitted_count,
                        "exit_status": parsed.exit_status,
                        "analyzers": [asdict(analyzer) for analyzer in parsed.analyzers],
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=coverage.normalized_count + coverage.omitted_count,
            complete=coverage.omitted_count == 0,
            limitations=list(parsed.limitations),
        )
