from __future__ import annotations

import json
from collections import defaultdict
from enum import StrEnum
from typing import Any

from flameox.domain.errors import DomainError, ErrorCode, missing_artifact_input
from flameox.domain.identity import digest_model
from flameox.domain.models import ArtifactKind, ExecutionStatus, RunManifest
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class PytestCompletionState(StrEnum):
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    INCOMPLETE = "incomplete"


class PytestExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    completion: PytestCompletionState
    execution_status: ExecutionStatus
    collected_count: int
    executed_count: int
    passed_count: int
    failed_count: int
    environment_blocked_count: int
    skipped_count: int
    errored_count: int
    unexecuted_count: int
    fixture_setup_count: int
    fixture_setup_ns: int
    collection_duration_ns: int | None
    workers: tuple[str, ...]
    first_failure_observed_ns: int | None
    first_failure_reported_ns: int | None
    measurement_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class PytestExtractor:
    name = "pytest"
    version = "2"

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
        outcome_counts = _test_outcomes(phases)
        executed = set(phases)
        unexecuted = collected - executed
        compiler_diagnostic, compiler_nodes = self._external_compiler_evidence(
            run,
            collected,
        )
        environment_blocked_nodes, environment_blocked_count, classified_counts = (
            self._classify_external_compiler_outcomes(
                phases,
                unexecuted,
                compiler_nodes,
            )
        )
        self._apply_external_compiler_evidence(
            rows,
            limitations,
            environment_blocked_nodes,
            compiler_diagnostic,
            run_id=run_id,
            artifact_id=registration.artifact_id,
        )
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
            ("pytest.tests.failed", classified_counts["failed"]),
            ("pytest.tests.environment_blocked", environment_blocked_count),
            ("pytest.tests.skipped", classified_counts["skipped"]),
            ("pytest.tests.errored", classified_counts["errored"]),
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
        completion, completion_limitations = _completion_state(
            events,
            interrupted=interrupted,
            crashed=crashed,
            truncated_stream=truncated_stream,
        )
        limitations.extend(completion_limitations)
        if crashed:
            limitations.append(
                "At least one xdist worker crashed; native report evidence is partial."
            )
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
            completion=completion,
            execution_status=run.execution_status,
            collected_count=len(collected),
            executed_count=len(executed),
            passed_count=outcome_counts["passed"],
            failed_count=classified_counts["failed"],
            environment_blocked_count=environment_blocked_count,
            skipped_count=classified_counts["skipped"],
            errored_count=classified_counts["errored"],
            unexecuted_count=len(unexecuted),
            fixture_setup_count=fixture_setup_count,
            fixture_setup_ns=fixture_setup_ns,
            collection_duration_ns=collection_duration,
            workers=tuple(sorted(workers)),
            first_failure_observed_ns=first_observed,
            first_failure_reported_ns=first_reported,
            measurement_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    def _external_compiler_evidence(
        self,
        run: RunManifest,
        collected: set[str],
    ) -> tuple[str | None, set[str]]:
        outputs: list[tuple[str, str]] = []
        for registration in run.artifacts:
            if registration.kind is not ArtifactKind.PROCESS_OUTPUT:
                continue
            if registration.role not in {"stdout", "stderr"}:
                continue
            try:
                payload = ArtifactStore(self.workspace).get(registration.artifact_id)
                with payload.payload_path.open("rb") as stream:
                    bounded = stream.read(64 * 1024)
                outputs.append((registration.role, bounded.decode(errors="replace")))
            except (DomainError, OSError, UnicodeError):
                continue
        markers = (
            "cuda_runtime.h: no such file or directory",
            "fatal error: cuda_runtime.h",
            "nvcc fatal",
            "cannot find -lcudart",
            "hip_runtime.h: no such file or directory",
            "clang: error: no such file or directory",
            "gcc: fatal error",
        )
        affected_nodes: set[str] = set()
        for _, output in outputs:
            lowered = output.casefold()
            if any(marker in lowered for marker in markers):
                affected_nodes.update(nodeid for nodeid in collected if nodeid in output)
        diagnostic = self._compiler_diagnostic_window(outputs, markers)
        return diagnostic, affected_nodes

    @staticmethod
    def _compiler_diagnostic_window(
        outputs: list[tuple[str, str]],
        markers: tuple[str, ...],
    ) -> str | None:
        for role, output in outputs:
            lowered = output.casefold()
            marker_positions = [
                (lowered.find(marker), marker) for marker in markers if lowered.find(marker) >= 0
            ]
            if not marker_positions:
                continue
            position, _ = min(marker_positions)
            start = max(0, position - 250)
            end = min(len(output), position + 250)
            detail = f"{role}: {output[start:end]}"
            return " ".join(detail.split())[:500] or (
                "External compiler returned no diagnostic detail."
            )
        return None

    @staticmethod
    def _apply_external_compiler_evidence(
        rows: list[dict[str, Any]],
        limitations: list[str],
        environment_blocked_nodes: set[str],
        compiler_diagnostic: str | None,
        *,
        run_id: str,
        artifact_id: str,
    ) -> None:
        if environment_blocked_nodes:
            for row in rows:
                dimensions = row.get("dimensions")
                nodeid = dimensions.get("nodeid") if isinstance(dimensions, dict) else None
                if nodeid in environment_blocked_nodes:
                    assert isinstance(dimensions, dict)
                    dimensions["classification"] = "environment_blocked"
                    dimensions["original_outcome"] = dimensions.get("outcome", "unknown")
                    row["measurement_id"] = digest_model(
                        {
                            "run_id": run_id,
                            "artifact_id": artifact_id,
                            "name": row["name"],
                            "value_index": row["value_index"],
                            "worker_id": row["worker_id"],
                            "dimensions": dimensions,
                        }
                    )
        if compiler_diagnostic is not None:
            limitations.extend(
                (
                    "An external compiler prerequisite failure was detected; affected pytest "
                    "outcomes are environment-blocked evidence, not a confirmed application "
                    "defect.",
                    f"External compiler diagnostic: {compiler_diagnostic}",
                )
            )

    @staticmethod
    def _classify_external_compiler_outcomes(
        phases: dict[str, dict[str, str]],
        unexecuted: set[str],
        compiler_nodes: set[str],
    ) -> tuple[set[str], int, dict[str, int]]:
        outcome_counts = _test_outcomes(phases)
        environment_blocked_nodes = {
            nodeid
            for nodeid, reports in phases.items()
            if nodeid in compiler_nodes and _test_outcome(reports) in {"failed", "errored"}
        }
        environment_blocked_nodes.update(compiler_nodes & unexecuted)
        environment_blocked_count = len(environment_blocked_nodes)
        blocked_outcomes = {
            outcome: sum(
                1
                for nodeid in environment_blocked_nodes
                if _test_outcome(phases[nodeid]) == outcome
            )
            for outcome in ("failed", "errored")
        }
        classified_counts = {
            **outcome_counts,
            "failed": outcome_counts["failed"] - blocked_outcomes["failed"],
            "errored": outcome_counts["errored"] - blocked_outcomes["errored"],
        }
        return environment_blocked_nodes, environment_blocked_count, classified_counts

    def _registration(self, run: RunManifest) -> Any:
        matches = [item for item in run.artifacts if item.kind is ArtifactKind.TEST_EXECUTION]
        if not matches:
            raise missing_artifact_input(
                run_id=run.run_id,
                requirement="pytest event-stream",
                artifact_kinds=(ArtifactKind.TEST_EXECUTION.value,),
                capture_adapters=("pytest",),
            )
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one pytest event artifact.",
                run_id=run.run_id,
            )
        return matches[0]

    def _events(self, path: Any, run_id: str) -> tuple[list[dict[str, Any]], bool]:
        payloads: list[dict[str, Any]] = []
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
                    if not isinstance(event, dict):
                        raise ValueError("unsupported event")
                    payloads.append(event)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported pytest event stream.",
                run_id=run_id,
            ) from exc
        if not payloads:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a supported pytest event stream.",
                run_id=run_id,
            )
        if payloads[0].get("$report_type") == "SessionStart":
            try:
                return _reportlog_events(payloads), truncated
            except (TypeError, ValueError) as exc:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "The artifact is not a supported pytest event stream.",
                    run_id=run_id,
                ) from exc
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            "The artifact is not a supported pytest event stream.",
            run_id=run_id,
        )

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


def _reportlog_events(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize maintained pytest-reportlog records into Flameox evidence semantics."""
    events: list[dict[str, Any]] = []
    run_started_ns: int | None = None
    for report in reports:
        report_type = report.get("$report_type")
        if report_type == "CollectReport":
            extension = report.get("flameox")
            if isinstance(extension, dict):
                nodeids = extension.get("collected_nodeids", ())
                if isinstance(nodeids, list):
                    events.extend(
                        {
                            "event": "test_collected",
                            "observed_at_ns": 0,
                            "nodeid": nodeid,
                        }
                        for nodeid in nodeids
                        if isinstance(nodeid, str)
                    )
            continue
        if report_type == "TestReport":
            nodeid = report.get("nodeid")
            phase = report.get("when")
            outcome = report.get("outcome")
            if not all(isinstance(value, str) for value in (nodeid, phase, outcome)):
                raise ValueError("invalid pytest TestReport")
            properties = _report_properties(report)
            longrepr = report.get("longrepr")
            worker_crash = (
                phase == "???"
                and isinstance(longrepr, str)
                and " crashed while running " in longrepr
            )
            candidate_start = _last_property(properties.get("flameox.run_started_ns"))
            if isinstance(candidate_start, int):
                run_started_ns = (
                    candidate_start
                    if run_started_ns is None
                    else min(run_started_ns, candidate_start)
                )
            worker_id = report.get("worker_id", "master")
            if not isinstance(worker_id, str):
                worker_id = "master"
            started_at_ns = round(float(report.get("start", 0.0)) * 1_000_000_000)
            stopped_at_ns = round(float(report.get("stop", 0.0)) * 1_000_000_000)
            received = _last_property(
                properties.get("flameox.controller_received_ns", stopped_at_ns)
            )
            events.append(
                {
                    "event": "test_collected",
                    "observed_at_ns": started_at_ns,
                    "nodeid": nodeid,
                }
            )
            if worker_crash:
                events.append(
                    {
                        "event": "worker_down",
                        "observed_at_ns": int(received),
                        "worker_id": worker_id,
                        "outcome": "crashed",
                    }
                )
            events.append(
                {
                    "event": "test_phase",
                    "observed_at_ns": int(received),
                    "nodeid": nodeid,
                    "worker_id": worker_id,
                    "phase": phase,
                    "outcome": outcome,
                    "duration_ns": round(float(report.get("duration", 0.0)) * 1_000_000_000),
                    "started_at_ns": started_at_ns,
                    "stopped_at_ns": stopped_at_ns,
                    "controller_received_at_ns": int(received),
                    "wasxfail": bool(report.get("wasxfail", False)),
                }
            )
            fixtures = properties.get("flameox.fixture_setup", ())
            fixtures = fixtures if isinstance(fixtures, list) else [fixtures]
            for fixture in fixtures:
                if not isinstance(fixture, dict):
                    continue
                events.append(
                    {
                        "event": "fixture_setup",
                        "observed_at_ns": int(fixture.get("started_at_ns", started_at_ns)),
                        "duration_ns": int(fixture.get("duration_ns", 0)),
                        "fixture": str(fixture.get("fixture", "unknown")),
                        "scope": str(fixture.get("scope", "unknown")),
                        "nodeid": nodeid,
                        "worker_id": worker_id,
                        "outcome": str(fixture.get("outcome", "unknown")),
                    }
                )
            continue
        if report_type == "SessionFinish":
            if int(report.get("exitstatus", 1)) == 2:
                events.append(
                    {
                        "event": "interrupted",
                        "observed_at_ns": max(
                            (int(event.get("observed_at_ns", 0)) for event in events),
                            default=0,
                        ),
                    }
                )
            events.append(
                {
                    "event": "session_finished",
                    "observed_at_ns": max(
                        (int(event.get("observed_at_ns", 0)) for event in events), default=0
                    ),
                    "exit_status": int(report.get("exitstatus", 1)),
                }
            )
            continue
        if report_type in {"SessionStart", "WarningMessage", "CollectReport"}:
            continue
        # pytest-reportlog explicitly permits consumers to ignore new report types.
    if run_started_ns is None:
        starts = [
            int(event["started_at_ns"]) for event in events if event.get("event") == "test_phase"
        ]
        run_started_ns = min(starts, default=0)
    events.insert(
        0,
        {
            "event": "run_started",
            "observed_at_ns": run_started_ns,
            "run_started_at_ns": run_started_ns,
            "scheduler": (
                "xdist" if any(event.get("worker_id") != "master" for event in events) else "no"
            ),
        },
    )
    return events


def _report_properties(report: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    raw = report.get("user_properties", ())
    if not isinstance(raw, list):
        return properties
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2 or not isinstance(item[0], str):
            continue
        name, value = item
        current = properties.get(name)
        if current is None:
            properties[name] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            properties[name] = [current, value]
    return properties


def _last_property(value: Any) -> Any:
    return value[-1] if isinstance(value, list) and value else value


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


def _test_outcome(reports: dict[str, str]) -> str:
    if reports.get("setup") == "failed" or reports.get("teardown") == "failed":
        return "errored"
    if reports.get("call") == "failed":
        return "failed"
    if "skipped" in reports.values():
        return "skipped"
    if reports.get("call") == "passed":
        return "passed"
    return "errored"


def _completion_state(
    events: list[dict[str, Any]],
    *,
    interrupted: bool,
    crashed: bool,
    truncated_stream: bool,
) -> tuple[PytestCompletionState, list[str]]:
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
    complete = has_terminal_event and not interrupted and not crashed and not truncated_stream
    state = (
        PytestCompletionState.COMPLETE
        if complete
        else (
            PytestCompletionState.INTERRUPTED if interrupted else PytestCompletionState.INCOMPLETE
        )
    )
    return state, limitations


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
