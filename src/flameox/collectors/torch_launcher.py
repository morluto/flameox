from __future__ import annotations

import argparse
import json
import runpy
import sys
from contextlib import suppress
from pathlib import Path

from flameox.atomic import atomic_write_json


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
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    diagnostic_path = Path(options.output).resolve().parent / "torch-profiler-diagnostics.json"
    _write_diagnostic(diagnostic_path, phase="wrapper_startup", status="started")
    phase = "wrapper_startup"
    try:
        if options.module is not None:
            sys.path.insert(0, str(Path.cwd()))
            script_path = None
        else:
            script_path = Path(options.script).resolve()
            sys.path.insert(0, str(script_path.parent))
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
        target_name = options.module or options.script
        assert target_name is not None
        sys.argv = [target_name, *options.arguments]
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
            if options.module is not None:
                runpy.run_module(options.module, run_name="__main__", alter_sys=True)
            else:
                assert script_path is not None
                runpy.run_path(str(script_path), run_name="__main__")
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
