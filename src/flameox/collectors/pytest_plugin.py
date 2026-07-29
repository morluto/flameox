from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import pytest

_MAX_SIDECAR_BYTES = 16 * 1024 * 1024
_SIDECAR_MARKER_RESERVE = 512
_SIDECAR_INPUT = "flameox_evidence_sidecar"
_SIDECAR_EVENTS = {"fixture_setup", "test_started", "sidecar_truncated"}
_output: Path | None = None
_worker_id = "master"
_wrote_collection = False
_sidecar_truncated = False


def _append_payload(payload: dict[str, Any]) -> None:
    global _sidecar_truncated
    if _output is None:
        return
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    if _worker_id != "master":
        if _sidecar_truncated:
            return
        current_size = _output.stat().st_size if _output.exists() else 0
        if current_size + len(encoded.encode()) + _SIDECAR_MARKER_RESERVE > _MAX_SIDECAR_BYTES:
            _sidecar_truncated = True
            encoded = (
                json.dumps(
                    {
                        "schema": "flameox.pytest-event.v1",
                        "event": "sidecar_truncated",
                        "observed_at_ns": time.time_ns(),
                        "worker_id": _worker_id,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    with _output.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()


def _write(event: str, **fields: Any) -> None:
    _append_payload(
        {
            "schema": "flameox.pytest-event.v1",
            "event": event,
            "observed_at_ns": time.time_ns(),
            **fields,
        }
    )


def _sidecar_path(primary: Path, worker_id: str) -> Path:
    worker_digest = hashlib.sha256(worker_id.encode()).hexdigest()[:16]
    return primary.with_name(f"{primary.name}.{worker_digest}.worker")


def pytest_configure(config: pytest.Config) -> None:
    global _output, _worker_id
    configured = os.environ.get("FLAMEOX_PYTEST_EVIDENCE_PATH")
    if configured is None:
        return
    worker_input = getattr(config, "workerinput", None)
    _worker_id = str(worker_input.get("workerid", "worker")) if worker_input else "master"
    if worker_input:
        sidecar = worker_input.get(_SIDECAR_INPUT)
        if isinstance(sidecar, str):
            _output = Path(sidecar)
        return
    _output = Path(configured)
    _output.parent.mkdir(parents=True, exist_ok=True)
    _output.write_text("")
    _write(
        "run_started",
        run_started_at_ns=int(os.environ["FLAMEOX_PYTEST_RUN_STARTED_NS"]),
        pytest_version=pytest.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        scheduler=str(getattr(config.option, "dist", "no")),
        requested_workers=str(getattr(config.option, "numprocesses", 0) or 0),
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    if _worker_id == "master":
        _write("session_started")


@pytest.hookimpl(wrapper=True)
def pytest_collection(session: pytest.Session) -> Any:
    if _worker_id == "master":
        _write("collection_started")
    return (yield)


def pytest_collection_finish(session: pytest.Session) -> None:
    global _wrote_collection
    if _worker_id != "master" or getattr(session.config.option, "numprocesses", None):
        return
    for item in session.items:
        _write("test_collected", nodeid=item.nodeid)
    _wrote_collection = True
    _write("collection_finished", test_count=len(session.items), worker_id="master")


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: pytest.FixtureRequest) -> Any:
    start_ns = time.time_ns()
    start = time.perf_counter_ns()
    outcome = "passed"
    try:
        result = yield
    except BaseException:
        outcome = "failed"
        raise
    finally:
        _append_payload(
            {
                "schema": "flameox.pytest-event.v1",
                "event": "fixture_setup",
                "observed_at_ns": start_ns,
                "duration_ns": time.perf_counter_ns() - start,
                "fixture": str(fixturedef.argname),
                "scope": str(fixturedef.scope),
                "nodeid": getattr(request.node, "nodeid", ""),
                "worker_id": _worker_id,
                "outcome": outcome,
            }
        )
    return result


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    if _worker_id != "master":
        _append_payload(
            {
                "schema": "flameox.pytest-event.v1",
                "event": "test_started",
                "observed_at_ns": time.time_ns(),
                "nodeid": nodeid,
                "worker_id": _worker_id,
            }
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _output is None or _worker_id != "master":
        return
    worker_id = str(getattr(report, "worker_id", _worker_id))
    _write(
        "test_phase",
        nodeid=report.nodeid,
        worker_id=worker_id,
        phase=report.when,
        outcome=report.outcome,
        duration_ns=round(report.duration * 1_000_000_000),
        started_at_ns=round(report.start * 1_000_000_000),
        stopped_at_ns=round(report.stop * 1_000_000_000),
        controller_received_at_ns=time.time_ns(),
        wasxfail=bool(getattr(report, "wasxfail", False)),
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    if _worker_id == "master":
        _write("session_finished", exit_status=int(exitstatus))


def pytest_keyboard_interrupt(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
    if _worker_id == "master":
        _write("interrupted", exception_type=excinfo.typename)


def pytest_internalerror(
    excrepr: Any,
    excinfo: pytest.ExceptionInfo[BaseException],
) -> None:
    if _worker_id == "master":
        _write("internal_error", exception_type=excinfo.typename)


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    worker_id = str(node.gateway.id)
    if _output is not None:
        sidecar = _sidecar_path(_output, worker_id)
        sidecar.write_text("")
        node.workerinput[_SIDECAR_INPUT] = str(sidecar)
    _write("worker_created", worker_id=worker_id)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodeready(node: Any) -> None:
    _write("worker_ready", worker_id=str(node.gateway.id))


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node: Any, ids: list[str]) -> None:
    global _wrote_collection
    worker_id = str(node.gateway.id)
    if not _wrote_collection:
        for nodeid in ids:
            _write("test_collected", nodeid=nodeid)
        _wrote_collection = True
    _write("collection_finished", test_count=len(ids), worker_id=worker_id)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object | None) -> None:
    _recover_sidecar(node)
    _write(
        "worker_down",
        worker_id=str(node.gateway.id),
        outcome="crashed" if error is not None else "clean",
        error_type=type(error).__name__ if error is not None else None,
    )


def _recover_sidecar(node: Any) -> None:
    configured = node.workerinput.get(_SIDECAR_INPUT)
    worker_id = str(node.gateway.id)
    if not isinstance(configured, str):
        _write(
            "worker_sidecar_recovered",
            worker_id=worker_id,
            recovered_count=0,
            rejected_count=0,
            outcome="unavailable",
        )
        return
    sidecar = Path(configured)
    recovered = 0
    rejected = 0
    truncated = False
    outcome = "failed"
    try:
        if _output is None or sidecar != _sidecar_path(_output, worker_id):
            raise ValueError("sidecar path differs from controller allocation")
        if sidecar.is_symlink() or not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        remaining = _MAX_SIDECAR_BYTES
        with sidecar.open("rb") as stream:
            while line := stream.readline(remaining + 1):
                if len(line) > remaining:
                    raise ValueError("sidecar exceeds recovery budget")
                remaining -= len(line)
                try:
                    payload = json.loads(line.decode())
                    if (
                        not isinstance(payload, dict)
                        or payload.get("schema") != "flameox.pytest-event.v1"
                        or payload.get("event") not in _SIDECAR_EVENTS
                        or payload.get("worker_id") != worker_id
                    ):
                        raise ValueError("invalid sidecar event")
                    _append_payload(payload)
                    if payload["event"] == "sidecar_truncated":
                        truncated = True
                    else:
                        recovered += 1
                except (json.JSONDecodeError, TypeError, ValueError):
                    rejected += 1
        outcome = "complete" if rejected == 0 and not truncated else "partial"
    except (OSError, UnicodeError, ValueError):
        outcome = "failed"
    if outcome == "complete":
        try:
            sidecar.unlink(missing_ok=True)
        except OSError:
            if outcome == "complete":
                outcome = "partial"
    _write(
        "worker_sidecar_recovered",
        worker_id=worker_id,
        recovered_count=recovered,
        rejected_count=rejected,
        outcome=outcome,
        truncated=truncated,
    )
