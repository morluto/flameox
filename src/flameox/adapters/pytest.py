from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import ArtifactKind, RunManifest
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class PytestExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    complete: bool
    execution_status: str
    collected_count: int
    executed_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    errored_count: int
    unexecuted_count: int
    fixture_setup_count: int
    fixture_setup_ns: int
    collection_duration_ns: int | None
    workers: tuple[str, ...]
    interrupted: bool
    recovered_sidecar_events: int
    sidecar_recovery_failures: int
    first_failure_observed_ns: int | None
    first_failure_reported_ns: int | None
    measurement_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class PytestExtractor:
    name = "pytest"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> PytestExtractionResult:
        run = RunStore(self.workspace).read(run_id)
        registration = self._registration(run)
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        events, truncated_stream = self._events(artifact.payload_path, run_id)
        maximum = self.workspace.config.storage.max_rows_per_generation
        if len(events) > maximum:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Pytest extraction exceeded the {maximum}-event generation limit.",
            )

        run_started = next(
            (event for event in events if event.get("event") == "run_started"),
            None,
        )
        if run_started is None:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The pytest event stream has no run_started event.",
                run_id=run_id,
            )
        run_started_ns = _integer(run_started, "run_started_at_ns")
        rows: list[dict[str, Any]] = []
        collected: set[str] = set()
        phases: dict[str, dict[str, str]] = defaultdict(dict)
        workers: set[str] = set()
        fixture_setup_count = 0
        fixture_setup_ns = 0
        failure_observed: list[int] = []
        failure_reported: list[int] = []
        limitations: list[str] = []
        crashed = False
        crashed_workers: set[str] = set()
        sidecar_events = [
            event for event in events if event.get("event") == "worker_sidecar_recovered"
        ]
        recovered_sidecar_events = sum(
            _integer(event, "recovered_count") for event in sidecar_events
        )
        recovered_sidecars = {
            _text(event, "worker_id")
            for event in sidecar_events
            if event.get("outcome") == "complete"
        }
        failed_sidecars = {
            _text(event, "worker_id")
            for event in sidecar_events
            if event.get("outcome") != "complete"
        }
        interrupted = any(
            event.get("event") in {"interrupted", "internal_error"} for event in events
        )

        for index, event in enumerate(events):
            event_name = event.get("event")
            if event_name == "test_collected":
                collected.add(_text(event, "nodeid"))
            elif event_name == "fixture_setup":
                duration = _integer(event, "duration_ns")
                worker_id = _text(event, "worker_id")
                workers.add(worker_id)
                fixture_setup_count += 1
                fixture_setup_ns += duration
                rows.append(
                    self._row(
                        run_id,
                        registration.artifact_id,
                        "pytest.fixture_setup",
                        duration,
                        "ns",
                        index,
                        worker_id,
                        "setup",
                        {
                            "fixture": _text(event, "fixture"),
                            "scope": _text(event, "scope"),
                            "nodeid": str(event.get("nodeid", "")),
                            "outcome": _text(event, "outcome"),
                        },
                    )
                )
            elif event_name == "test_phase":
                nodeid = _text(event, "nodeid")
                phase = _text(event, "phase")
                outcome = _text(event, "outcome")
                worker_id = _text(event, "worker_id")
                workers.add(worker_id)
                phases[nodeid][phase] = outcome
                rows.append(
                    self._row(
                        run_id,
                        registration.artifact_id,
                        f"pytest.phase.{phase}",
                        _integer(event, "duration_ns"),
                        "ns",
                        index,
                        worker_id,
                        phase,
                        {"nodeid": nodeid, "outcome": outcome},
                    )
                )
                if outcome == "failed":
                    failure_observed.append(
                        max(0, _integer(event, "stopped_at_ns") - run_started_ns)
                    )
                    failure_reported.append(
                        max(
                            0,
                            _integer(event, "controller_received_at_ns") - run_started_ns,
                        )
                    )
            elif event_name in {"worker_created", "worker_ready", "worker_down"}:
                worker_id = _text(event, "worker_id")
                workers.add(worker_id)
                if event_name == "worker_down" and event.get("outcome") == "crashed":
                    crashed = True
                    crashed_workers.add(worker_id)
        outcome_counts = _test_outcomes(phases)
        executed = set(phases)
        unexecuted = collected - executed
        first_observed = min(failure_observed, default=None)
        first_reported = min(failure_reported, default=None)
        collection_started = [
            _integer(event, "observed_at_ns")
            for event in events
            if event.get("event") == "collection_started"
        ]
        collection_finished = [
            _integer(event, "observed_at_ns")
            for event in events
            if event.get("event") == "collection_finished"
        ]
        collection_duration = (
            max(0, max(collection_finished) - min(collection_started))
            if collection_started and collection_finished
            else None
        )
        for name, value in (
            ("pytest.tests.collected", len(collected)),
            ("pytest.tests.executed", len(executed)),
            ("pytest.tests.passed", outcome_counts["passed"]),
            ("pytest.tests.failed", outcome_counts["failed"]),
            ("pytest.tests.skipped", outcome_counts["skipped"]),
            ("pytest.tests.errored", outcome_counts["errored"]),
            ("pytest.tests.unexecuted", len(unexecuted)),
            ("pytest.fixture_setup.total", fixture_setup_ns),
        ):
            rows.append(
                self._row(
                    run_id,
                    registration.artifact_id,
                    name,
                    value,
                    "ns" if name.endswith(".total") else "count",
                    len(rows),
                    None,
                    "summary",
                    {},
                    aggregation="sum",
                )
            )
        for name, failure_value in (
            ("pytest.time_to_first_failure.observed", first_observed),
            ("pytest.time_to_first_failure.reported", first_reported),
        ):
            if failure_value is not None:
                rows.append(
                    self._row(
                        run_id,
                        registration.artifact_id,
                        name,
                        failure_value,
                        "ns",
                        len(rows),
                        None,
                        "failure",
                        {},
                        aggregation="first",
                    )
                )

        if collection_duration is not None:
            rows.append(
                self._row(
                    run_id,
                    registration.artifact_id,
                    "pytest.collection",
                    collection_duration,
                    "ns",
                    len(rows),
                    None,
                    "collection",
                    {},
                    aggregation="single",
                )
            )
        complete, completion_limitations = _completion(
            events,
            interrupted=interrupted,
            crashed=crashed,
            failed_sidecars=failed_sidecars,
            truncated_stream=truncated_stream,
        )
        limitations.extend(completion_limitations)
        if crashed:
            limitations.append(_crash_limitation(crashed_workers, recovered_sidecars))
        if any(event.get("scheduler") not in {None, "no"} for event in events):
            limitations.append(
                "Stable xdist hooks do not expose exact per-test controller queue latency."
            )
        published = self.publisher.publish_rows(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
        )
        return PytestExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            complete=complete,
            execution_status=run.execution_status.value,
            collected_count=len(collected),
            executed_count=len(executed),
            passed_count=outcome_counts["passed"],
            failed_count=outcome_counts["failed"],
            skipped_count=outcome_counts["skipped"],
            errored_count=outcome_counts["errored"],
            unexecuted_count=len(unexecuted),
            fixture_setup_count=fixture_setup_count,
            fixture_setup_ns=fixture_setup_ns,
            collection_duration_ns=collection_duration,
            workers=tuple(sorted(workers)),
            interrupted=interrupted,
            recovered_sidecar_events=recovered_sidecar_events,
            sidecar_recovery_failures=len(failed_sidecars),
            first_failure_observed_ns=first_observed,
            first_failure_reported_ns=first_reported,
            measurement_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    def _registration(self, run: RunManifest) -> Any:
        matches = [item for item in run.artifacts if item.kind is ArtifactKind.TEST_EXECUTION]
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one pytest event artifact.",
                run_id=run.run_id,
            )
        return matches[0]

    def _events(self, path: Any, run_id: str) -> tuple[list[dict[str, Any]], bool]:
        events: list[dict[str, Any]] = []
        truncated = False
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n") and stream.read(1) == "":
                            truncated = True
                            break
                        raise
                    if (
                        not isinstance(event, dict)
                        or event.get("schema") != "flameox.pytest-event.v1"
                    ):
                        raise ValueError("unsupported event")
                    events.append(event)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported pytest event stream.",
                run_id=run_id,
            ) from exc
        return events, truncated

    def _row(
        self,
        run_id: str,
        artifact_id: str,
        name: str,
        value: int,
        unit: str,
        value_index: int,
        worker_id: str | None,
        phase: str,
        dimensions: dict[str, str],
        *,
        aggregation: str = "sample",
    ) -> dict[str, Any]:
        identity = {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": name,
            "value_index": value_index,
            "worker_id": worker_id,
            "dimensions": dimensions,
        }
        return {
            "measurement_id": digest_model(identity),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": name,
            "value_int": value,
            "value_float": None,
            "unit": unit,
            "aggregation": aggregation,
            "scope": "test_suite",
            "trial_id": None,
            "worker_id": worker_id,
            "worker_run_index": None,
            "value_index": value_index,
            "loop_count": None,
            "is_warmup": False,
            "block_id": None,
            "variant_id": None,
            "order_in_block": None,
            "phase": phase,
            "dimensions": dimensions,
            "evidence_level": "observed",
        }


def _test_outcomes(phases: dict[str, dict[str, str]]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errored": 0}
    for reports in phases.values():
        if reports.get("setup") == "failed" or reports.get("teardown") == "failed":
            counts["errored"] += 1
        elif reports.get("call") == "failed":
            counts["failed"] += 1
        elif "skipped" in reports.values():
            counts["skipped"] += 1
        elif reports.get("call") == "passed":
            counts["passed"] += 1
        else:
            counts["errored"] += 1
    return counts


def _completion(
    events: list[dict[str, Any]],
    *,
    interrupted: bool,
    crashed: bool,
    failed_sidecars: set[str],
    truncated_stream: bool,
) -> tuple[bool, list[str]]:
    limitations: list[str] = []
    has_terminal_event = any(event.get("event") == "session_finished" for event in events)
    if not has_terminal_event:
        limitations.append("The pytest session did not emit a terminal event; evidence is partial.")
    if interrupted:
        limitations.append("The pytest session was interrupted or ended with an internal error.")
    if truncated_stream:
        limitations.append(
            "The pytest event stream ended with a truncated record; the valid prefix was recovered."
        )
    if failed_sidecars:
        limitations.append(
            "One or more worker sidecars were unavailable, partial, or failed recovery."
        )
    complete = (
        has_terminal_event
        and not interrupted
        and not crashed
        and not failed_sidecars
        and not truncated_stream
    )
    return complete, limitations


def _crash_limitation(
    crashed_workers: set[str],
    recovered_sidecars: set[str],
) -> str:
    unrecovered_crashes = crashed_workers - recovered_sidecars
    if unrecovered_crashes:
        return (
            "At least one crashed xdist worker had no complete recoverable fixture sidecar: "
            + ", ".join(sorted(unrecovered_crashes))
            + "."
        )
    return (
        "Fixture and test-start events emitted before an xdist worker crash were "
        "recovered from bounded sidecars."
    )


def _integer(value: dict[str, Any], field: str) -> int:
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Pytest event field {field!r} is invalid.",
        )
    return result


def _text(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Pytest event field {field!r} is invalid.",
        )
    return result
