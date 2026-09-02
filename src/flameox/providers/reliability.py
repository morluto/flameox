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
        collected: set[str] = set()
        phase_events: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        collection_errors: list[dict[str, Any]] = []
        global_failures: list[dict[str, Any]] = []
        finished = False
        interrupted = False
        for index, event in self._events(path):
            event_name = event.get("event")
            if event_name == "test_collected" and isinstance(event.get("nodeid"), str):
                collected.add(event["nodeid"])
            elif event_name == "test_phase":
                nodeid = event.get("nodeid")
                phase = event.get("phase")
                outcome = event.get("outcome")
                if all(isinstance(item, str) for item in (nodeid, phase, outcome)):
                    phase_events[str(nodeid)][str(phase)] = self._pytest_row(index, event)
            elif event_name == "collection_error" and isinstance(event.get("nodeid"), str):
                collection_errors.append(
                    {
                        "index": index,
                        "nodeid": str(event["nodeid"]),
                        "classification": "errored",
                        "failing_phase": "collection",
                        "phase_outcomes": {"collection": str(event.get("outcome", "failed"))},
                    }
                )
            elif event_name == "run_finished":
                finished = True
            elif event_name in {"interrupted", "internal_error"}:
                interrupted = True
                global_failures.append(
                    {"index": index, "classification": str(event_name), "nodeid": None}
                )
        phases = {
            nodeid: {phase: str(event["outcome"]) for phase, event in reports.items()}
            for nodeid, reports in phase_events.items()
        }
        outcomes = self._outcomes(phases)
        outcomes["errored"] += len(collection_errors)
        diagnostic_rows = [
            self._pytest_outcome_row(nodeid, reports)
            for nodeid, reports in phase_events.items()
            if self._pytest_classification(phases[nodeid]) in {"failed", "errored"}
        ]
        diagnostic_rows.extend(global_failures)
        diagnostic_rows.extend(collection_errors)
        diagnostic_rows.extend(
            {
                "index": None,
                "nodeid": nodeid,
                "classification": "unexecuted",
                "failing_phase": None,
                "phase_outcomes": {},
            }
            for nodeid in sorted(collected.difference(phases))
        )
        priority = {
            "errored": 0,
            "failed": 1,
            "internal_error": 2,
            "interrupted": 3,
            "unexecuted": 4,
        }
        diagnostic_rows.sort(
            key=lambda row: (
                priority[str(row["classification"])],
                str(row.get("nodeid") or ""),
            )
        )
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
                {"type": "table", "rows": diagnostic_rows[:max_rows]},
            ],
            rows_observed=len(diagnostic_rows),
            complete=len(diagnostic_rows) <= max_rows,
            limitations=[
                "Rows contain failure, error, interruption, and unexecuted identities; passing "
                "and skipped identities remain summarized in metrics.",
                "Failure tracebacks remain in the native pytest artifact and are not projected.",
            ],
        )

    @staticmethod
    def _pytest_classification(reports: dict[str, str]) -> str:
        if reports.get("setup") == "failed" or reports.get("teardown") == "failed":
            return "errored"
        if reports.get("call") == "failed":
            return "failed"
        if "skipped" in reports.values():
            return "skipped"
        if reports.get("call") == "passed":
            return "passed"
        return "errored"

    @classmethod
    def _pytest_outcome_row(cls, nodeid: str, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
        outcomes = {phase: str(event["outcome"]) for phase, event in reports.items()}
        classification = cls._pytest_classification(outcomes)
        failing_phase = next(
            (phase for phase in ("setup", "call", "teardown") if outcomes.get(phase) == "failed"),
            None,
        )
        return {
            "index": min(int(event["index"]) for event in reports.values()),
            "nodeid": nodeid,
            "classification": classification,
            "failing_phase": failing_phase,
            "phase_outcomes": outcomes,
        }

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
