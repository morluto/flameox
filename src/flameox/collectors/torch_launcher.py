from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, assert_never, cast

from flameox.atomic import atomic_write_json


@dataclass(frozen=True, slots=True)
class _ModuleTarget:
    name: str


@dataclass(frozen=True, slots=True)
class _ScriptTarget:
    path: Path


@dataclass(frozen=True, slots=True)
class _InlineTarget:
    code: str


type _Target = _ModuleTarget | _ScriptTarget | _InlineTarget


class _Profiler(Protocol):
    def events(self) -> object: ...

    def export_chrome_trace(self, path: str) -> None: ...


def _parse_target(options: argparse.Namespace, parser: argparse.ArgumentParser) -> _Target:
    if options.module is not None:
        return _ModuleTarget(options.module)
    if options.script is not None:
        return _ScriptTarget(Path(options.script).resolve())
    if options.inline_code is not None:
        return _InlineTarget(options.inline_code)
    parser.error("one Python target is required")


def _prepare_target(target: _Target) -> str:
    if isinstance(target, _ModuleTarget):
        sys.path.insert(0, str(Path.cwd()))
        return target.name
    if isinstance(target, _ScriptTarget):
        sys.path.insert(0, str(target.path.parent))
        return str(target.path)
    if isinstance(target, _InlineTarget):
        sys.path.insert(0, str(Path.cwd()))
        return "<flameox-inline-python>"
    assert_never(target)


def _run_target(target: _Target, target_name: str) -> None:
    if isinstance(target, _ModuleTarget):
        runpy.run_module(target.name, run_name="__main__", alter_sys=True)
    elif isinstance(target, _InlineTarget):
        namespace = {
            "__name__": "__main__",
            "__file__": target_name,
            "__package__": None,
            "__cached__": None,
        }
        exec(compile(target.code, target_name, "exec"), namespace, namespace)
    elif isinstance(target, _ScriptTarget):
        runpy.run_path(str(target.path), run_name="__main__")
    else:
        assert_never(target)


def _write_diagnostic(
    path: Path,
    *,
    phase: str,
    status: str,
    detail: str | None = None,
    values: dict[str, int | float] | None = None,
) -> None:
    payload: dict[str, object] = {
        "phase": phase,
        "status": status,
    }
    if detail is not None:
        payload["detail"] = " ".join(detail.split())[:500]
    if values:
        payload.update(values)
    with suppress(OSError):
        atomic_write_json(path, payload)


def _provider_diagnostics(profile: object) -> dict[str, int | float]:
    """Return bounded scalar facts exposed by the completed provider profile."""

    events_method = getattr(profile, "events", None)
    if not callable(events_method):
        return {}
    try:
        events = events_method()
        event_count = 0
        min_start: float | None = None
        max_end: float | None = None
        for event in events:
            event_count += 1
            time_range = getattr(event, "time_range", None)
            start = getattr(time_range, "start", None)
            end = getattr(time_range, "end", None)
            if (
                isinstance(start, (int, float))
                and not isinstance(start, bool)
                and math.isfinite(start)
                and isinstance(end, (int, float))
                and not isinstance(end, bool)
                and math.isfinite(end)
                and end >= start
            ):
                min_start = float(start) if min_start is None else min(min_start, start)
                max_end = float(end) if max_end is None else max(max_end, end)
    except Exception:
        return {}

    values: dict[str, int | float] = {"event_count": event_count}
    if min_start is not None and max_end is not None:
        values["active_duration_us"] = max_end - min_start
    return values


def _trace_size(path: Path) -> dict[str, int]:
    try:
        return {"artifact_size_bytes": path.stat().st_size}
    except OSError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one declared Python entrypoint with torch.profiler."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module")
    target.add_argument("--script")
    target.add_argument("--inline-code")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    target_case = _parse_target(options, parser)
    workload_arguments = list(options.arguments)
    if workload_arguments[:1] == ["--"]:
        workload_arguments = workload_arguments[1:]
    diagnostic_path = Path(options.output).resolve().parent / "torch-profiler-diagnostics.json"
    _write_diagnostic(diagnostic_path, phase="wrapper_startup", status="started")
    phase = "wrapper_startup"
    profile: _Profiler | None = None
    try:
        target_name = _prepare_target(target_case)
        try:
            import torch
        except ImportError as exc:
            parser.error(f"PyTorch is unavailable: {exc}")
        try:
            config = json.loads(options.config)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid profiler configuration: {exc}")
        if not isinstance(config, dict) or config.get("mode") != "whole_entrypoint":
            parser.error("Whole-entrypoint launcher requires whole_entrypoint mode")
        configured_activities = config.get("activities")
        if not isinstance(configured_activities, list):
            parser.error("Profiler activities are missing")
        activities = []
        if "cpu" in configured_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in configured_activities and not torch.cuda.is_available():
            parser.error("The capture plan requires CUDA, but CUDA is unavailable")
        if "cuda" in configured_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if "cuda_if_available" in configured_activities and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if not activities:
            parser.error("No requested torch.profiler activity is available")
        output = Path(options.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        argv0 = "-c" if isinstance(target_case, _InlineTarget) else target_name
        sys.argv = [argv0, *workload_arguments]
        phase = "profiler_initialization"
        _write_diagnostic(diagnostic_path, phase=phase, status="running")
        with torch.profiler.profile(
            activities=activities,
            record_shapes=config["record_shapes"],
            profile_memory=config["profile_memory"],
            with_stack=config["with_stack"],
            with_flops=config["with_flops"],
            with_modules=config["with_modules"],
        ) as active_profile:
            profile = cast(_Profiler, active_profile)
            phase = "workload_execution"
            _write_diagnostic(diagnostic_path, phase=phase, status="running")
            try:
                _run_target(target_case, target_name)
            except SystemExit as exc:
                if exc.code not in {None, 0}:
                    raise
            phase = "profiler_finalization"
            _write_diagnostic(diagnostic_path, phase=phase, status="running")
        phase = "trace_finalization"
        _write_diagnostic(diagnostic_path, phase=phase, status="running")
        profile.export_chrome_trace(str(output))
        diagnostics = _provider_diagnostics(profile)
        diagnostics.update(_trace_size(output))
        _write_diagnostic(
            diagnostic_path,
            phase="completed",
            status="succeeded",
            values=diagnostics,
        )
    except BaseException as exc:
        diagnostics = _provider_diagnostics(profile) if profile is not None else {}
        diagnostics.update(_trace_size(Path(options.output).resolve()))
        _write_diagnostic(
            diagnostic_path,
            phase=phase,
            status="failed",
            detail=f"{type(exc).__name__}: {exc}",
            values=diagnostics,
        )
        raise


if __name__ == "__main__":
    main()
