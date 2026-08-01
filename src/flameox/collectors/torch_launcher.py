from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


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
    with torch.profiler.profile(
        activities=activities,
        record_shapes=config["record_shapes"],
        profile_memory=config["profile_memory"],
        with_stack=config["with_stack"],
        with_flops=config["with_flops"],
        with_modules=config["with_modules"],
    ) as profile:
        if options.module is not None:
            runpy.run_module(options.module, run_name="__main__", alter_sys=True)
        else:
            assert script_path is not None
            runpy.run_path(str(script_path), run_name="__main__")
    profile.export_chrome_trace(str(output))


if __name__ == "__main__":
    main()
