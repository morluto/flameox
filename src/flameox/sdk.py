from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flameox_phase",
    default=None,
)
_WRITE_LOCK = threading.Lock()
_MAX_EVENT_BYTES = 16 * 1024
_TORCH_PROFILER_CONFIG = "FLAMEOX_TORCH_PROFILER_CONFIG"
_TORCH_PROFILER_OUTPUT_ROOT = "FLAMEOX_TORCH_PROFILER_OUTPUT_ROOT"


def observe(name: str, **values: Any) -> None:
    """Emit one bounded semantic observation when capture has enabled the SDK."""
    if not name or len(name) > 200:
        raise ValueError("observation names must contain 1 to 200 characters")
    path = os.environ.get("FLAMEOX_OBSERVATIONS_PATH")
    if path is None:
        return
    payload = {
        "schema_version": 1,
        "kind": "annotation",
        "name": name,
        "phase": _PHASE.get(),
        "monotonic_ns": time.monotonic_ns(),
        "values": _bounded_value(values),
    }
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    if len(encoded) > _MAX_EVENT_BYTES:
        raise ValueError("observation exceeds the 16 KiB event limit")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, destination.open("ab") as stream:
        stream.write(encoded)
        stream.flush()


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Annotate a bounded logical phase around user code."""
    if not name or len(name) > 200:
        raise ValueError("phase names must contain 1 to 200 characters")
    token = _PHASE.set(name)
    observe("flameox.phase.start", phase_name=name)
    try:
        yield
    finally:
        observe("flameox.phase.end", phase_name=name)
        _PHASE.reset(token)


class TorchProfilerSession:
    """Narrow scheduled-profiler handle exposed to an approved workload."""

    def __init__(self, profile: Any, torch: Any) -> None:
        self._profile = profile
        self._torch = torch

    def step(self) -> None:
        """Advance one declared workload step in the configured schedule."""
        self._profile.step()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Emit both a semantic phase and a trace-visible record-function range."""
        if not name or len(name) > 200:
            raise ValueError("phase names must contain 1 to 200 characters")
        with ExitStack() as stack:
            stack.enter_context(phase(name))
            stack.enter_context(self._torch.profiler.record_function(f"flameox.phase:{name}"))
            yield


@contextmanager
def torch_profiler() -> Iterator[TorchProfilerSession]:
    """Run the scheduled profiler configuration bound into a Flameox capture plan."""
    raw_config = os.environ.get(_TORCH_PROFILER_CONFIG)
    output_root_value = os.environ.get(_TORCH_PROFILER_OUTPUT_ROOT)
    if raw_config is None or output_root_value is None:
        raise RuntimeError("torch_profiler() requires a Flameox torch.profiler SDK capture plan")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid Flameox torch.profiler configuration") from exc
    if not isinstance(config, dict) or config.get("mode") != "sdk":
        raise RuntimeError("Flameox torch.profiler SDK configuration is invalid")
    schedule = config.get("schedule")
    activities = config.get("activities")
    expected_cycles = config.get("expected_cycles")
    if (
        not isinstance(schedule, dict)
        or not isinstance(activities, list)
        or not isinstance(expected_cycles, int)
        or isinstance(expected_cycles, bool)
        or expected_cycles < 1
    ):
        raise RuntimeError("Flameox torch.profiler SDK configuration is incomplete")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable in the workload environment") from exc

    selected_activities = []
    if "cpu" in activities:
        selected_activities.append(torch.profiler.ProfilerActivity.CPU)
    if "cuda" in activities and not torch.cuda.is_available():
        raise RuntimeError("the Flameox capture plan requires CUDA, but CUDA is unavailable")
    if "cuda" in activities:
        selected_activities.append(torch.profiler.ProfilerActivity.CUDA)
    if "cuda_if_available" in activities and torch.cuda.is_available():
        selected_activities.append(torch.profiler.ProfilerActivity.CUDA)
    if not selected_activities:
        raise RuntimeError("no requested torch.profiler activity is available")
    output_root = Path(output_root_value)
    output_root.mkdir(parents=True, exist_ok=True)
    cycle = 0

    def export_trace(profile: Any) -> None:
        nonlocal cycle
        if cycle >= expected_cycles:
            raise RuntimeError("torch.profiler emitted more trace cycles than planned")
        output = output_root / f"torch-trace-cycle-{cycle:04d}.json"
        if output.exists():
            raise RuntimeError(f"torch.profiler cycle output already exists: {output.name}")
        profile.export_chrome_trace(str(output))
        cycle += 1

    profile = torch.profiler.profile(
        activities=selected_activities,
        schedule=torch.profiler.schedule(
            wait=schedule["wait"],
            warmup=schedule["warmup"],
            active=schedule["active"],
            repeat=schedule["repeat"],
            skip_first=schedule["skip_first"],
        ),
        on_trace_ready=export_trace,
        record_shapes=config["record_shapes"],
        profile_memory=config["profile_memory"],
        with_stack=config["with_stack"],
        with_flops=config["with_flops"],
        with_modules=config["with_modules"],
    )
    failed = False
    try:
        with profile:
            yield TorchProfilerSession(profile, torch)
    except BaseException:
        failed = True
        raise
    finally:
        if not failed and cycle != expected_cycles:
            raise RuntimeError(
                f"torch.profiler emitted {cycle} of {expected_cycles} planned trace cycles"
            )
        if not failed and cycle == expected_cycles:
            manifest = output_root / "torch-profiler-cycles.json"
            temporary = output_root / ".torch-profiler-cycles.tmp"
            if manifest.exists() or temporary.exists():
                raise RuntimeError("torch.profiler cycle manifest already exists")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": "flameox.torch-profiler-cycles.v1",
                        "expected_cycles": expected_cycles,
                        "files": [
                            f"torch-trace-cycle-{index:04d}.json"
                            for index in range(expected_cycles)
                        ],
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            os.replace(temporary, manifest)


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("observation nesting exceeds eight levels")
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("observations cannot contain non-finite numbers")
        return value
    if isinstance(value, list | tuple):
        if len(value) > 256:
            raise ValueError("observation lists cannot exceed 256 items")
        return [_bounded_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 256 or any(not isinstance(key, str) for key in value):
            raise ValueError("observation objects require at most 256 string keys")
        return {key: _bounded_value(item, depth=depth + 1) for key, item in value.items()}
    raise TypeError(f"unsupported observation value: {type(value).__name__}")
