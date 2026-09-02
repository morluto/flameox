from __future__ import annotations

from pathlib import Path
from typing import cast

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.workers.compute_sanitizer_contract import (
    COMPUTE_SANITIZER_WORKER,
    ComputeSanitizerWorkerRequest,
)
from flameox.workers.harness import IsolatedWorkerHarness
from flameox.workers.pstats_contract import PSTATS_WORKER, PstatsMetric, PstatsWorkerRequest
from flameox.workers.v8_profiles_contract import (
    V8_PROFILE_WORKER,
    V8ProfileRequest,
)


class StructuredWorkerProviders:
    """Explicit adapters for workers whose result is already bounded structured evidence."""

    def __init__(self, harness: IsolatedWorkerHarness, project_root: Path) -> None:
        self.harness = harness
        self.project_root = project_root

    def analyze(
        self,
        capability_id: str,
        path: Path,
        input_sha256: str,
        format_name: str,
        arguments: dict[str, object],
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis | None:
        if capability_id == "cpu.hotspots" and format_name == "pstats":
            metric = str(arguments.get("metric") or "self_time_seconds")
            if metric not in {
                "self_time_seconds",
                "cumulative_time_seconds",
                "total_calls",
                "primitive_calls",
            }:
                raise ProviderFailure("INVALID_INPUT", f"Unsupported pstats metric: {metric}")
            pstats_result = self.harness.run_typed_sync(
                PSTATS_WORKER,
                PstatsWorkerRequest(
                    metric=cast(PstatsMetric, metric),
                    artifact_path=str(path),
                    max_rows=max_rows,
                ),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            return ProviderAnalysis(
                provider_id="python-pstats",
                provider_version=PSTATS_WORKER.implementation,
                blocks=[
                    {
                        "type": "metrics",
                        "values": {
                            "function_count": pstats_result.function_count,
                            "metric": pstats_result.metric,
                            "reader_python_version": pstats_result.reader_version,
                        },
                    },
                    {"type": "table", "rows": [dict(row) for row in pstats_result.rows]},
                ],
                rows_observed=pstats_result.function_count,
                complete=not pstats_result.truncated,
                limitations=list(pstats_result.limitations),
            )
        if capability_id == "cpu.hotspots" and format_name == "cpuprofile":
            result = self.harness.run_typed_sync(
                V8_PROFILE_WORKER,
                V8ProfileRequest(
                    profile_kind="cpu",
                    artifact_path=str(path),
                    artifact_id=input_sha256,
                    project_root=str(self.project_root),
                    max_nodes=max_rows,
                    max_samples=max_rows,
                    max_rows=max_rows,
                ),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            rows = [dict(row) for row in result.frame_measurements]
            return ProviderAnalysis(
                provider_id="v8-cpu-profile",
                provider_version=V8_PROFILE_WORKER.implementation,
                blocks=[
                    {
                        "type": "metrics",
                        "values": {
                            "node_count": result.node_count,
                            "sample_count": result.sample_count,
                        },
                    },
                    {"type": "table", "rows": rows},
                ],
                rows_observed=result.node_count,
                complete=len(rows) >= result.node_count,
                limitations=list(result.limitations),
            )
        if capability_id == "sanitizer.failures" and format_name == "compute-sanitizer":
            sanitizer_result = self.harness.run_typed_sync(
                COMPUTE_SANITIZER_WORKER,
                ComputeSanitizerWorkerRequest(
                    artifact_path=str(path),
                    project_root=str(self.project_root),
                    max_records=max_rows,
                    max_frames=max_rows,
                ),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            return ProviderAnalysis(
                provider_id="compute-sanitizer",
                provider_version=COMPUTE_SANITIZER_WORKER.implementation,
                blocks=[
                    {"type": "metrics", "values": dict(sanitizer_result.classifications)},
                    {
                        "type": "table",
                        "rows": [dict(row) for row in sanitizer_result.records],
                    },
                ],
                rows_observed=len(sanitizer_result.records) + int(sanitizer_result.truncated),
                complete=not sanitizer_result.truncated,
                limitations=list(sanitizer_result.limitations),
            )
        return None
