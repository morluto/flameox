from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_MAX_LINE_BYTES = 64 * 1024


class ReliabilityProvider:
    """Bounded projections for explicit pytest and semantic-observation streams."""

    def analyze(self, path: Path, format_name: str, *, max_rows: int) -> ProviderAnalysis:
        if format_name == "pytest":
            return self._pytest(path, max_rows=max_rows)
        if format_name == "observations":
            return self._observations(path, max_rows=max_rows)
        raise ProviderFailure(
            "UNSUPPORTED_FORMAT", f"Unsupported reliability format: {format_name}"
        )

    def _pytest(self, path: Path, *, max_rows: int) -> ProviderAnalysis:
        rows: list[dict[str, Any]] = []
        observed = 0
        collected: set[str] = set()
        phases: dict[str, dict[str, str]] = defaultdict(dict)
        finished = False
        interrupted = False
        for index, event in self._events(path):
            observed += 1
            event_name = event.get("event")
            if event_name == "test_collected" and isinstance(event.get("nodeid"), str):
                collected.add(event["nodeid"])
            elif event_name == "test_phase":
                nodeid = event.get("nodeid")
                phase = event.get("phase")
                outcome = event.get("outcome")
                if all(isinstance(item, str) for item in (nodeid, phase, outcome)):
                    phases[str(nodeid)][str(phase)] = str(outcome)
            elif event_name == "run_finished":
                finished = True
            elif event_name in {"interrupted", "internal_error"}:
                interrupted = True
            if len(rows) < max_rows:
                rows.append(self._pytest_row(index, event))
        outcomes = self._outcomes(phases)
        completion = "interrupted" if interrupted else "complete" if finished else "incomplete"
        return ProviderAnalysis(
            provider_id="pytest",
            provider_version="event-stream-v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "completion": completion,
                        "collected": len(collected),
                        "executed": len(phases),
                        "unexecuted": len(collected.difference(phases)),
                        **outcomes,
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=[
                "Failure tracebacks remain in the native pytest artifact and are not projected."
            ],
        )

    def _observations(self, path: Path, *, max_rows: int) -> ProviderAnalysis:
        rows: list[dict[str, Any]] = []
        observed = 0
        for index, event in self._events(path):
            if set(event) != {"name", "phase", "monotonic_ns", "values"}:
                raise ProviderFailure(
                    "DECODE_FAILURE", "Observation fields differ from the SDK contract"
                )
            if (
                not isinstance(event["name"], str)
                or not isinstance(event["monotonic_ns"], int)
                or (event["phase"] is not None and not isinstance(event["phase"], str))
                or not isinstance(event["values"], dict)
            ):
                raise ProviderFailure("DECODE_FAILURE", "Observation field types are invalid")
            observed += 1
            if len(rows) < max_rows:
                rows.append(
                    {
                        "index": index,
                        "name": event["name"],
                        "phase": event["phase"],
                        "monotonic_ns": event["monotonic_ns"],
                        "values": event["values"],
                    }
                )
        return ProviderAnalysis(
            provider_id="flameox-sdk-observations",
            provider_version="1",
            blocks=[
                {"type": "metrics", "values": {"observation_count": observed}},
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=["Observation values are caller-authored semantic evidence."],
        )

    @staticmethod
    def _events(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        try:
            with path.open("rb") as stream:
                for index, raw in enumerate(stream):
                    if len(raw) > _MAX_LINE_BYTES:
                        raise ValueError("event line exceeds its byte bound")
                    if not raw.strip():
                        continue
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise ValueError("event must be an object")
                    yield index, event
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ProviderFailure("DECODE_FAILURE", "Event stream is invalid") from error

    @staticmethod
    def _pytest_row(index: int, event: dict[str, Any]) -> dict[str, Any]:
        allowed = (
            "event",
            "nodeid",
            "phase",
            "outcome",
            "duration_ns",
            "worker_id",
            "fixture",
            "scope",
        )
        return {"index": index, **{key: event[key] for key in allowed if key in event}}

    @staticmethod
    def _outcomes(phases: dict[str, dict[str, str]]) -> dict[str, int]:
        counts = {"passed": 0, "failed": 0, "skipped": 0, "errored": 0}
        for reports in phases.values():
            if reports.get("setup") == "failed" or reports.get("teardown") == "failed":
                counts["errored"] += 1
            elif reports.get("call") in counts:
                counts[str(reports["call"])] += 1
            elif "skipped" in reports.values():
                counts["skipped"] += 1
            else:
                counts["errored"] += 1
        return counts
