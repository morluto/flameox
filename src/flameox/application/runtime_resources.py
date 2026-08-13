from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type RuntimeResourceMetric = Literal[
    "runtime_resource.peak_rss_bytes",
    "runtime_resource.minimum_free_bytes",
    "runtime_resource.staging_growth_bytes",
]


@dataclass(frozen=True, slots=True)
class RuntimeResourceMetricDefinition:
    """One reviewed scalar admitted to runtime-resource comparison."""

    metric: RuntimeResourceMetric
    evidence_column: str
    unavailable_key: str
    unit: Literal["bytes"]
    scope: str
    aggregation: str
    collection_backend: str
    compatibility_fields: tuple[Literal["sampling_interval_ms", "peak_rss_backend"], ...]
    unavailable_behavior: str
    supports_confirmatory_paired_comparison: bool
    limitations: tuple[str, ...]


_RUNTIME_RESOURCE_METRICS = (
    RuntimeResourceMetricDefinition(
        metric="runtime_resource.peak_rss_bytes",
        evidence_column="peak_rss_bytes",
        unavailable_key="peak_rss",
        unit="bytes",
        scope="broker-owned workload process tree, including recursively discovered descendants",
        aggregation="maximum sampled sum of resident bytes across the observed process tree",
        collection_backend="psutil_recursive_polling",
        compatibility_fields=("sampling_interval_ms", "peak_rss_backend"),
        unavailable_behavior="missing or inaccessible process samples make the run ineligible",
        supports_confirmatory_paired_comparison=True,
        limitations=(
            "the sampled maximum is not a guarantee of the process tree lifetime maximum",
            "descendants that start and exit between samples may be absent",
        ),
    ),
    RuntimeResourceMetricDefinition(
        metric="runtime_resource.minimum_free_bytes",
        evidence_column="minimum_free_bytes",
        unavailable_key="minimum_free",
        unit="bytes",
        scope="filesystem containing the workspace staging root",
        aggregation="minimum sampled free-byte count while the workload is running",
        collection_backend="python_shutil_disk_usage",
        compatibility_fields=("sampling_interval_ms",),
        unavailable_behavior="a failed filesystem observation makes the run ineligible",
        supports_confirmatory_paired_comparison=True,
        limitations=(
            "the value includes concurrent filesystem activity outside the workload",
            "the sampled minimum may miss shorter-lived free-space changes",
        ),
    ),
    RuntimeResourceMetricDefinition(
        metric="runtime_resource.staging_growth_bytes",
        evidence_column="staging_growth_bytes",
        unavailable_key="staging_growth",
        unit="bytes",
        scope="workspace staging tree excluding separately declared writable-root growth",
        aggregation="nonnegative final tree size minus initial tree size and declared-root growth",
        collection_backend="bounded_tree_size_before_after",
        compatibility_fields=(),
        unavailable_behavior=(
            "an unreadable or over-budget tree observation makes the run ineligible"
        ),
        supports_confirmatory_paired_comparison=True,
        limitations=(
            "the bounded traversal does not account for files created and removed "
            "between observations",
            "zero-valued runs are outside the current log-ratio comparison domain",
        ),
    ),
)

RUNTIME_RESOURCE_METRIC_CATALOG: Mapping[str, RuntimeResourceMetricDefinition] = MappingProxyType(
    {definition.metric: definition for definition in _RUNTIME_RESOURCE_METRICS}
)
RUNTIME_RESOURCE_METRICS = frozenset(RUNTIME_RESOURCE_METRIC_CATALOG)


def runtime_resource_metric_definition(metric: str) -> RuntimeResourceMetricDefinition:
    try:
        return RUNTIME_RESOURCE_METRIC_CATALOG[metric]
    except KeyError as exc:
        supported = ", ".join(sorted(RUNTIME_RESOURCE_METRIC_CATALOG))
        raise ValueError(f"runtime-resource metric must be one of {supported}") from exc
