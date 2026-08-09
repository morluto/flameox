from __future__ import annotations

import argparse
import json
import runpy
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

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
) -> None:
    payload: dict[str, object] = {
        "schema_version": "flameox.torch-profiler-diagnostics.v1",
        "phase": phase,
        "status": status,
    }
    if detail is not None:
        payload["detail"] = " ".join(detail.split())[:500]
    with suppress(OSError):
        atomic_write_json(path, payload)


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
        ) as profile:
            phase = "workload_execution"
            _write_diagnostic(diagnostic_path, phase=phase, status="running")
            _run_target(target_case, target_name)
            phase = "profiler_finalization"
            _write_diagnostic(diagnostic_path, phase=phase, status="running")
        phase = "trace_finalization"
        _write_diagnostic(diagnostic_path, phase=phase, status="running")
        profile.export_chrome_trace(str(output))
        _write_diagnostic(diagnostic_path, phase="completed", status="succeeded")
    except BaseException as exc:
        _write_diagnostic(
            diagnostic_path,
            phase=phase,
            status="failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
