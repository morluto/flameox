from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one declared Python entrypoint with torch.profiler."
    )
    parser.add_argument("--output", required=True)
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
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_name = options.module or options.script
    assert target_name is not None
    sys.argv = [target_name, *options.arguments]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profile:
        if options.module is not None:
            runpy.run_module(options.module, run_name="__main__", alter_sys=True)
        else:
            assert script_path is not None
            runpy.run_path(str(script_path), run_name="__main__")
    profile.export_chrome_trace(str(output))


if __name__ == "__main__":
    main()
