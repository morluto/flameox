from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import flameox.sdk as sdk
from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.memray_options import memray_capture_options
from flameox.adapters.options import bind_adapter_options, run_semantics
from flameox.application.capture import _memray_region_validation_issues
from flameox.domain import DomainError, ErrorCode
from flameox.sdk import memray_region

pytestmark = pytest.mark.unit


def test_memray_capture_options_bind_one_explicit_sdk_region(tmp_path: Path) -> None:
    options = bind_adapter_options(
        "memray",
        {
            "mode": "sdk",
            "region": "steady_step",
            "warmup_count": 3,
            "native_traces": True,
            "trace_python_allocators": True,
        },
        project_root=tmp_path,
    )

    semantics = run_semantics("memray", "1.19.3", options)

    assert semantics.scope.mode is not None
    assert semantics.scope.mode.value == "sdk"
    assert semantics.scope.process_scope is not None
    assert semantics.scope.process_scope.value == "workload_process"
    assert semantics.scope.bounds == {"warmup_count": 3}
    assert semantics.scope.filters == {
        "region": "steady_step",
        "thread_scope": "all_threads",
    }
    assert semantics.configuration == {
        "native_traces": True,
        "trace_python_allocators": True,
    }


@pytest.mark.parametrize(
    "options",
    (
        {"mode": "sdk"},
        {"mode": "whole_entrypoint", "region": "steady_step"},
    ),
)
def test_memray_capture_options_reject_ambiguous_scope(options: dict[str, object]) -> None:
    with pytest.raises(DomainError) as raised:
        memray_capture_options(options)

    assert raised.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "required exactly" in raised.value.details["validation_error"]


def test_whole_entrypoint_memray_rejects_an_unbound_warmup_count() -> None:
    with pytest.raises(DomainError) as raised:
        memray_capture_options({"warmup_count": 1})

    assert raised.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "sdk" in raised.value.details["validation_error"]


@pytest.mark.parametrize(
    ("events", "expected_codes"),
    (
        ((), {"memray_region_missing"}),
        (
            ({"name": "flameox.memray.region.start", "values": {"region": "steady_step"}},),
            {"memray_region_unclosed"},
        ),
        (
            (
                {"name": "flameox.memray.region.start", "values": {"region": "steady_step"}},
                {
                    "name": "flameox.memray.region.error",
                    "values": {"region": "steady_step", "reason": "overlap"},
                },
                {"name": "flameox.memray.region.end", "values": {"region": "steady_step"}},
                {
                    "name": "flameox.memray.region.error",
                    "values": {"region": "steady_step", "reason": "repeated"},
                },
            ),
            {"memray_region_overlap", "memray_region_repeated"},
        ),
    ),
)
def test_memray_region_lifecycle_failures_are_typed(
    tmp_path: Path,
    events: tuple[dict[str, object], ...],
    expected_codes: set[str],
) -> None:
    observations = tmp_path / "observations.jsonl"
    if events:
        observations.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    issues = _memray_region_validation_issues(
        observations,
        expected_region="steady_step",
        max_bytes=16 * 1024,
    )

    assert {code for code, _message in issues} == expected_codes


def test_memray_sdk_invocation_gives_the_workload_one_authorized_output(tmp_path: Path) -> None:
    workload = (sys.executable, "workload.py", "--batches", "2")

    invocation = build_capture_invocation(
        "memray",
        workload,
        tmp_path,
        executable=None,
        options={"mode": "sdk", "region": "steady_step"},
    )

    assert invocation.argv == workload
    assert invocation.environment["FLAMEOX_MEMRAY_OUTPUT"] == str(tmp_path / "memory.bin")
    assert json.loads(invocation.environment["FLAMEOX_MEMRAY_CONFIG"]) == {
        "mode": "sdk",
        "native_traces": False,
        "region": "steady_step",
        "trace_python_allocators": False,
        "warmup_count": 0,
    }
    assert any("every thread" in limitation for limitation in invocation.limitations)
    assert any(
        "Nested, concurrent, or repeated" in limitation for limitation in invocation.limitations
    )


def test_whole_entrypoint_memray_invocation_uses_only_supported_provider_flags(
    tmp_path: Path,
) -> None:
    invocation = build_capture_invocation(
        "memray",
        (sys.executable, "workload.py"),
        tmp_path,
        executable=None,
        options={"native_traces": True, "trace_python_allocators": True},
    )

    assert invocation.argv == (
        sys.executable,
        "-m",
        "memray",
        "run",
        "--native",
        "--trace-python-allocators",
        "--output",
        str(tmp_path / "memory.bin"),
        "workload.py",
    )


@pytest.mark.optional
@pytest.mark.requires_memray
def test_memray_sdk_region_excludes_setup_allocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("memray", reason="optional provider unavailable: install memray")
    from memray._memray import compute_statistics

    capture = tmp_path / "memory.bin"
    monkeypatch.setattr(sdk, "_MEMRAY_REGION_STATE", "idle")
    monkeypatch.setenv(
        "FLAMEOX_MEMRAY_CONFIG",
        json.dumps({"mode": "sdk", "region": "steady_step"}),
    )
    monkeypatch.setenv("FLAMEOX_MEMRAY_OUTPUT", str(capture))

    setup = bytearray(4_000_000)
    with memray_region("steady_step"):
        measured = bytearray(128_000)

    stats = compute_statistics(str(capture), report_progress=False, num_largest=1)
    assert len(setup) == 4_000_000
    assert len(measured) == 128_000
    assert capture.is_file()
    assert 100_000 <= stats.total_memory_allocated < 1_000_000
    assert stats.total_num_allocations > 0
