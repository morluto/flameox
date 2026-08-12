from __future__ import annotations

import os
import time
from typing import Any

import pytest

_FIXTURES: pytest.StashKey[list[dict[str, Any]]] = pytest.StashKey()
_RUN_STARTED_NS: pytest.StashKey[int] = pytest.StashKey()


def pytest_configure(config: pytest.Config) -> None:
    """Initialize Flameox-only annotations carried by pytest-reportlog."""
    config.stash[_FIXTURES] = []
    configured = os.environ.get("FLAMEOX_PYTEST_RUN_STARTED_NS")
    config.stash[_RUN_STARTED_NS] = int(configured) if configured is not None else time.time_ns()


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: pytest.FixtureRequest) -> Any:
    """Measure fixture setup, which pytest's native reports do not itemize."""
    started_at_ns = time.time_ns()
    started = time.perf_counter_ns()
    outcome = "passed"
    try:
        result = yield
    except BaseException:
        outcome = "failed"
        raise
    finally:
        request.config.stash[_FIXTURES].append(
            {
                "duration_ns": time.perf_counter_ns() - started,
                "fixture": str(fixturedef.argname),
                "scope": str(fixturedef.scope),
                "started_at_ns": started_at_ns,
                "outcome": outcome,
            }
        )
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    report = yield
    report.user_properties.append(("flameox.run_started_ns", item.config.stash[_RUN_STARTED_NS]))
    if call.when == "setup":
        fixtures = item.config.stash[_FIXTURES]
        for fixture in fixtures:
            report.user_properties.append(("flameox.fixture_setup", fixture))
        fixtures.clear()
    return report


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record when the controller observed a report without replacing its schema."""
    report.user_properties.append(("flameox.controller_received_ns", time.time_ns()))


@pytest.hookimpl(wrapper=True)
def pytest_report_to_serializable(config: pytest.Config, report: Any) -> Any:
    """Retain collection evidence omitted by pytest's default wire representation."""
    serialized = yield
    if isinstance(serialized, dict) and isinstance(report, pytest.CollectReport):
        serialized["flameox"] = {
            "collected_nodeids": [
                nodeid
                for item in (report.result or ())
                if isinstance(item, pytest.Item)
                if isinstance((nodeid := getattr(item, "nodeid", None)), str)
            ],
        }
    return serialized
