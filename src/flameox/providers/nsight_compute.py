from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.workers.harness import IsolatedWorkerHarness
from flameox.workers.nsight_compute_contract import (
    NSIGHT_COMPUTE_WORKER,
    NsightComputeWorkerRequest,
)

_INSTALL_ROOTS = (
    Path("/opt/nvidia/nsight-compute"),
    Path("/usr/local/NVIDIA-Nsight-Compute"),
)


def find_report_interface(executable: Path | None = None) -> Path | None:
    """Locate the vendor-shipped reader; third-party binary decoders are not accepted."""
    if executable is not None:
        for parent in executable.resolve().parents:
            candidate = parent / "extras" / "python" / "ncu_report.py"
            if candidate.is_file():
                return candidate
    candidates: set[Path] = set()
    for root in _INSTALL_ROOTS:
        direct = root / "extras" / "python" / "ncu_report.py"
        if direct.is_file():
            candidates.add(direct)
        if root.is_dir():
            candidates.update(root.glob("*/extras/python/ncu_report.py"))
    return max(candidates, key=_interface_key) if candidates else None


def _interface_key(path: Path) -> tuple[Version, str]:
    match = re.search(r"\d+(?:\.\d+)+", path.as_posix())
    try:
        version = Version(match.group()) if match else Version("0")
    except InvalidVersion:
        version = Version("0")
    return version, path.as_posix()


class NsightComputeProvider:
    """Read an explicit native Nsight Compute report with NVIDIA's typed interface."""

    def __init__(
        self,
        harness: IsolatedWorkerHarness,
        *,
        interface_path: Path | None = None,
    ) -> None:
        self.harness = harness
        self.interface_path = interface_path

    def analyze(
        self,
        path: Path,
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis:
        executable_text = shutil.which("ncu")
        interface = self.interface_path or find_report_interface(
            Path(executable_text) if executable_text else None
        )
        if interface is None:
            raise ProviderFailure(
                "UNAVAILABLE_CAPABILITY",
                "The official ncu_report Python interface was not found.",
                details={
                    "remediation": (
                        "Install Nsight Compute with its extras/python interface outside Flameox."
                    )
                },
            )
        metric_limit = max(1, (max_rows + 1) // 2)
        observation_limit = max(1, max_rows - metric_limit)
        response = self.harness.run_typed_sync(
            NSIGHT_COMPUTE_WORKER,
            NsightComputeWorkerRequest(
                artifact_path=str(path),
                interface_path=str(interface),
                max_ranges=min(1_000, observation_limit),
                max_actions=min(10_000, observation_limit),
                max_metrics=metric_limit,
                max_observations=observation_limit,
            ),
            timeout_seconds=timeout_seconds,
            maximum_rss_bytes=maximum_rss_bytes,
            maximum_writable_growth_bytes=maximum_output_bytes,
        )
        rows: list[dict[str, Any]] = [
            {"evidence_kind": "measurement", **dict(item)} for item in response.measurements
        ]
        rows.extend(
            {"evidence_kind": "observation", **dict(item)} for item in response.observations
        )
        rows = rows[:max_rows]
        observed = len(response.measurements) + len(response.observations)
        return ProviderAnalysis(
            provider_id="nsight-compute",
            provider_version=response.report_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "range_count": response.range_count,
                        "action_count": response.action_count,
                        "metric_count": len(response.measurements),
                        "observation_count": len(response.observations),
                        "metric_ids": list(response.metric_ids),
                        "section_ids": list(response.section_ids),
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=not response.truncated and observed <= len(rows),
            limitations=list(response.limitations),
        )
