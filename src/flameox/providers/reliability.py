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

    def analyze(
        self, capability_id: str, path: Path, format_name: str, *, max_rows: int
    ) -> ProviderAnalysis:
        if format_name == "pytest":
            if capability_id == "pytest.fixtures":
                return self._pytest_fixtures(path, max_rows=max_rows)
            return self._pytest(path, max_rows=max_rows)
        if format_name == "observations":
            return self._observations(path, max_rows=max_rows)
        raise ProviderFailure(
            "UNSUPPORTED_FORMAT", f"Unsupported reliability format: {format_name}"
        )

    def _pytest_fixtures(self, path: Path, *, max_rows: int) -> ProviderAnalysis:
        invocations: dict[str, dict[str, Any]] = {}
        finished = False
        interrupted = False
        for index, event in self._events(path):
            event_name = event.get("event")
            if event_name == "run_finished":
                finished = True
                continue
            if event_name in {"interrupted", "internal_error"}:
                interrupted = True
                continue
            if event_name != "fixture_phase":
                continue
            required = ("fixture", "scope", "phase", "outcome", "worker_id", "invocation_id")
            if not all(isinstance(event.get(field), str) for field in required):
                raise ProviderFailure("DECODE_FAILURE", "Fixture event fields are invalid")
            duration = event.get("duration_ns")
            if duration is not None and (
                not isinstance(duration, int) or isinstance(duration, bool) or duration < 0
            ):
                raise ProviderFailure("DECODE_FAILURE", "Fixture duration is invalid")
            invocation_id = str(event["invocation_id"])
            invocation = invocations.setdefault(
                invocation_id,
                {
                    "index": index,
                    "fixture": str(event["fixture"]),
                    "scope": str(event["scope"]),
                    "worker_id": str(event["worker_id"]),
                    "nodeid": str(event.get("nodeid") or ""),
                    "setup_duration_ns": None,
                    "teardown_duration_ns": None,
                    "setup_outcome": None,
                    "teardown_outcome": None,
                },
            )
            if any(
                invocation[field] != str(event.get(field) or "")
                for field in ("fixture", "scope", "worker_id", "nodeid")
            ):
                raise ProviderFailure("DECODE_FAILURE", "Fixture invocation identity changed")
            phase = str(event["phase"])
            if phase not in {"setup", "teardown"} or invocation[f"{phase}_outcome"] is not None:
                raise ProviderFailure("DECODE_FAILURE", "Fixture phase is invalid or repeated")
            invocation[f"{phase}_duration_ns"] = duration
            invocation[f"{phase}_outcome"] = str(event["outcome"])

        aggregate_values: dict[tuple[str, str], dict[str, Any]] = {}
        worker_work: dict[str, int] = defaultdict(int)
        invocation_rows: list[dict[str, Any]] = []
        for invocation_id, invocation in invocations.items():
            setup_duration = invocation["setup_duration_ns"]
            teardown_duration = invocation["teardown_duration_ns"]
            known_work = sum(
                value for value in (setup_duration, teardown_duration) if isinstance(value, int)
            )
            worker_work[str(invocation["worker_id"])] += known_work
            complete = (
                isinstance(setup_duration, int)
                and isinstance(teardown_duration, int)
                and invocation["teardown_outcome"] == "completed"
            )
            invocation_rows.append(
                {
                    "row_kind": "fixture_invocation",
                    "invocation_id": invocation_id,
                    **invocation,
                    "known_work_ns": known_work,
                    "complete": complete,
                }
            )
            key = (str(invocation["fixture"]), str(invocation["scope"]))
            aggregate = aggregate_values.setdefault(
                key,
                {
                    "invocation_count": 0,
                    "workers": set(),
                    "setup_work_ns": 0,
                    "teardown_work_ns": 0,
                    "incomplete_invocation_count": 0,
                },
            )
            aggregate["invocation_count"] += 1
            aggregate["workers"].add(str(invocation["worker_id"]))
            aggregate["setup_work_ns"] += setup_duration if isinstance(setup_duration, int) else 0
            aggregate["teardown_work_ns"] += (
                teardown_duration if isinstance(teardown_duration, int) else 0
            )
            aggregate["incomplete_invocation_count"] += int(not complete)
        aggregate_rows = [
            {
                "row_kind": "fixture_aggregate",
                "fixture": fixture,
                "scope": scope,
                "invocation_count": values["invocation_count"],
                "worker_count": len(values["workers"]),
                "setup_work_ns": values["setup_work_ns"],
                "teardown_work_ns": values["teardown_work_ns"],
                "known_work_ns": values["setup_work_ns"] + values["teardown_work_ns"],
                "incomplete_invocation_count": values["incomplete_invocation_count"],
            }
            for (fixture, scope), values in aggregate_values.items()
        ]
        aggregate_rows.sort(key=lambda row: (-int(row["known_work_ns"]), str(row["fixture"])))
        invocation_rows.sort(
            key=lambda row: (-int(row["known_work_ns"]), str(row["invocation_id"]))
        )
        rows = [*aggregate_rows, *invocation_rows]
        completion = "interrupted" if interrupted else "complete" if finished else "incomplete"
        incomplete_count = sum(not bool(row["complete"]) for row in invocation_rows)
        return ProviderAnalysis(
            provider_id="pytest",
            provider_version="event-stream-v2",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "completion": completion,
                        "fixture_count": len(aggregate_rows),
                        "invocation_count": len(invocation_rows),
                        "worker_count": len(worker_work),
                        "incomplete_invocation_count": incomplete_count,
                        "summed_fixture_work_ns": sum(worker_work.values()),
                        "max_worker_fixture_work_ns": max(worker_work.values(), default=0),
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=[
                "Summed fixture work can overlap across xdist workers and is not wall-clock "
                "critical-path duration.",
                "max_worker_fixture_work_ns is accumulated known work on one worker, not a "
                "critical-path measurement.",
                "Interrupted runs retain incomplete fixture invocations instead of treating "
                "missing finalizers as zero duration.",
            ],
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
