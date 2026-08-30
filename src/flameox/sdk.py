from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Any

from flameox.adapters.benchmark_samples import (
    BenchmarkDevice,
    BenchmarkSamplesV1,
    BenchmarkSeries,
)
from flameox.adapters.memray_options import MemrayCaptureOptions, memray_capture_options
from flameox.adapters.torch_benchmark import TorchBenchmarkOptions, parse_torch_benchmark_options
from flameox.atomic import atomic_write_json
from flameox.domain import DomainError

_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flameox_phase",
    default=None,
)
_WRITE_LOCK = threading.Lock()
_MAX_EVENT_BYTES = 16 * 1024
_TORCH_PROFILER_CONFIG = "FLAMEOX_TORCH_PROFILER_CONFIG"
_TORCH_PROFILER_OUTPUT_ROOT = "FLAMEOX_TORCH_PROFILER_OUTPUT_ROOT"
_TORCH_BENCHMARK_CONFIG = "FLAMEOX_TORCH_BENCHMARK_CONFIG"
_TORCH_BENCHMARK_OUTPUT = "FLAMEOX_BENCHMARK_OUTPUT"
_MEMRAY_CONFIG = "FLAMEOX_MEMRAY_CONFIG"
_MEMRAY_OUTPUT = "FLAMEOX_MEMRAY_OUTPUT"
_MEMRAY_REGION_LOCK = threading.Lock()
_MEMRAY_REGION_STATE = "idle"


def observe(name: str, **values: Any) -> None:
    """Emit one bounded semantic observation when capture has enabled the SDK."""
    if not name or len(name) > 200:
        raise ValueError("observation names must contain 1 to 200 characters")
    path = os.environ.get("FLAMEOX_OBSERVATIONS_PATH")
    if path is None:
        return
    payload = {
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
    """Annotate a bounded logical phase around user code.

    Observation I/O remains fatal when the body succeeds. If both the body and
    the closing observation fail, the body failure remains primary and receives
    a bounded note naming the secondary instrumentation failure.
    """
    if not name or len(name) > 200:
        raise ValueError("phase names must contain 1 to 200 characters")
    token = _PHASE.set(name)
    body_error: BaseException | None = None
    try:
        observe("flameox.phase.start", phase_name=name)
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                observe("flameox.phase.end", phase_name=name)
            except BaseException as observation_error:
                if body_error is None:
                    raise
                body_error.add_note(
                    "Flameox phase-end observation also failed "
                    f"({type(observation_error).__name__}); the workload failure remains primary."
                )
    finally:
        _PHASE.reset(token)


def torch_benchmark(
    name: str,
    operation: Callable[[], object],
    *,
    dimensions: Mapping[str, str] | None = None,
) -> None:
    """Measure one prepared callable with PyTorch's Timer and publish bounded samples.

    Setup, lazy initialization, and correctness checks belong to the workload before this call.
    The optional CUDA-event series is deliberately a separately named device metric.
    """

    if not callable(operation):
        raise TypeError("torch_benchmark operation must be callable")
    config = _torch_benchmark_config()
    output = _torch_benchmark_output()
    try:
        import torch
        from torch.utils.benchmark import Timer
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable in the workload environment") from exc

    cuda_available = torch.cuda.is_available()
    measurement = Timer(
        stmt="operation()",
        globals={"operation": operation},
        num_threads=config.num_threads,
    ).blocked_autorange(min_run_time=config.min_run_time_seconds)
    host_samples = _timer_samples_ns(measurement, config.max_samples)
    host = BenchmarkSeries(
        name=name,
        unit="ns",
        measurement_clock="host_monotonic",
        synchronization="device_synchronize" if cuda_available else "not_required",
        scope="operator",
        loop_count=_timer_loop_count(measurement),
        dimensions=dict(dimensions or {}),
        samples=host_samples,
    )
    benchmarks = [host]
    if config.cuda_event_timing:
        if not cuda_available:
            raise RuntimeError("CUDA-event timing was requested, but CUDA is unavailable")
        device, event_samples = _cuda_event_samples(
            torch,
            operation,
            sample_count=config.max_samples,
        )
        benchmarks.append(
            BenchmarkSeries(
                name=f"{name}.cuda_event",
                unit="ns",
                measurement_clock="cuda_event",
                synchronization="device_synchronize",
                scope="device",
                loop_count=1,
                device=device,
                dimensions=dict(dimensions or {}),
                samples=event_samples,
            )
        )
    _append_torch_benchmark_document(
        output,
        producer_version=str(torch.__version__),
        benchmarks=tuple(benchmarks),
    )


def _torch_benchmark_config() -> TorchBenchmarkOptions:
    raw = os.environ.get(_TORCH_BENCHMARK_CONFIG)
    if raw is None:
        raise RuntimeError("torch_benchmark() requires a Flameox torch.benchmark capture plan")
    try:
        return parse_torch_benchmark_options(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Flameox torch.benchmark configuration is invalid") from exc


def _torch_benchmark_output() -> Path:
    value = os.environ.get(_TORCH_BENCHMARK_OUTPUT)
    if value is None:
        raise RuntimeError("torch_benchmark() requires a Flameox benchmark output path")
    return Path(value)


def _timer_loop_count(measurement: Any) -> int:
    count = getattr(measurement, "number_per_run", None)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise RuntimeError("torch.utils.benchmark.Timer returned no valid loop count")
    return count


def _timer_samples_ns(measurement: Any, max_samples: int) -> tuple[int, ...]:
    values = getattr(measurement, "times", None)
    if not isinstance(values, list | tuple) or not values:
        raise RuntimeError("torch.utils.benchmark.Timer returned no samples")
    samples: list[int] = []
    for value in values[:max_samples]:
        if not isinstance(value, float) or value < 0 or value != value:
            raise RuntimeError("torch.utils.benchmark.Timer returned an invalid sample")
        samples.append(round(value * 1_000_000_000))
    return tuple(samples)


def _cuda_event_samples(
    torch: Any,
    operation: Callable[[], object],
    *,
    sample_count: int,
) -> tuple[BenchmarkDevice, tuple[int, ...]]:
    index = torch.cuda.current_device()
    stream = torch.cuda.current_stream()
    stream_id = getattr(stream, "cuda_stream", None)
    device = BenchmarkDevice(
        type="cuda",
        index=index,
        stream=str(stream_id) if stream_id is not None else None,
    )
    samples: list[int] = []
    for _ in range(sample_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        if not isinstance(elapsed_ms, float) or elapsed_ms < 0 or elapsed_ms != elapsed_ms:
            raise RuntimeError("CUDA event timing returned an invalid sample")
        samples.append(round(elapsed_ms * 1_000_000))
    return device, tuple(samples)


def _append_torch_benchmark_document(
    output: Path,
    *,
    producer_version: str,
    benchmarks: tuple[BenchmarkSeries, ...],
) -> None:
    with _WRITE_LOCK:
        if output.exists():
            try:
                existing = BenchmarkSamplesV1.model_validate_json(
                    output.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(
                    "Flameox benchmark output is not a valid benchmark document"
                ) from exc
            if (
                existing.producer != "torch.utils.benchmark"
                or existing.producer_version != producer_version
            ):
                raise RuntimeError("Flameox benchmark output belongs to another producer")
            document = BenchmarkSamplesV1(
                schema_version="flameox.benchmark-samples.v1",
                producer=existing.producer,
                producer_version=existing.producer_version,
                benchmarks=(*existing.benchmarks, *benchmarks),
            )
        else:
            document = BenchmarkSamplesV1(
                schema_version="flameox.benchmark-samples.v1",
                producer="torch.utils.benchmark",
                producer_version=producer_version,
                benchmarks=benchmarks,
            )
        atomic_write_json(output, document.model_dump(mode="json"))


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
    cycle_count = schedule.get("repeat") if isinstance(schedule, dict) else None
    if (
        not isinstance(schedule, dict)
        or not isinstance(activities, list)
        or not isinstance(cycle_count, int)
        or isinstance(cycle_count, bool)
        or cycle_count < 1
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
        if cycle >= cycle_count:
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
        if not failed and cycle != cycle_count:
            raise RuntimeError(
                f"torch.profiler emitted {cycle} of {cycle_count} planned trace cycles"
            )


@contextmanager
def memray_region(name: str) -> Iterator[None]:
    """Track exactly one plan-bound memory region for this process.

    Memray covers every thread while a tracker is active and permits only one
    tracker per process. A second Flameox region therefore fails before it can
    produce ambiguous evidence.
    """

    selected, output = _memray_region_config(name)
    try:
        import memray
    except ImportError as exc:
        raise RuntimeError("Memray is unavailable in the workload environment") from exc
    global _MEMRAY_REGION_STATE
    with _MEMRAY_REGION_LOCK:
        if _MEMRAY_REGION_STATE != "idle":
            reason = "overlap" if _MEMRAY_REGION_STATE == "active" else "repeated"
            with suppress(BaseException):
                observe("flameox.memray.region.error", region=name, reason=reason)
            raise RuntimeError(
                "Memray SDK capture permits exactly one region per process; end the active "
                "region or create a fresh capture plan with one declared region."
            )
        _MEMRAY_REGION_STATE = "active"
    completed = False
    failure: BaseException | None = None
    cuda_initialized_before = _torch_cuda_initialized()
    try:
        observe(
            "flameox.memray.region.start",
            region=name,
            warmup_count=selected.warmup_count,
            torch_cuda_initialized=cuda_initialized_before,
        )
        with memray.Tracker(
            output,
            native_traces=selected.native_traces,
            trace_python_allocators=selected.trace_python_allocators,
        ):
            yield
        completed = True
    except BaseException as error:
        failure = error
        raise
    finally:
        with _MEMRAY_REGION_LOCK:
            _MEMRAY_REGION_STATE = "closed"
        try:
            observe(
                "flameox.memray.region.end",
                region=name,
                completed=completed,
                torch_cuda_initialized=_torch_cuda_initialized(),
            )
        except BaseException as observation_error:
            if failure is None:
                raise
            failure.add_note(
                "Flameox Memray region-end observation also failed "
                f"({type(observation_error).__name__}); the workload failure remains primary."
            )


def _memray_region_config(name: str) -> tuple[MemrayCaptureOptions, str]:
    raw_config = os.environ.get(_MEMRAY_CONFIG)
    output = os.environ.get(_MEMRAY_OUTPUT)
    if raw_config is None or output is None:
        raise RuntimeError("memray_region() requires a Flameox Memray SDK capture plan")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid Flameox Memray configuration") from exc
    if not isinstance(config, dict):
        raise RuntimeError("Flameox Memray SDK configuration is invalid")
    try:
        selected = memray_capture_options(config)
    except DomainError as exc:
        raise RuntimeError("Flameox Memray SDK configuration is invalid") from exc
    if selected.mode != "sdk" or selected.region != name:
        raise RuntimeError("memray_region() must use the exact region declared by the capture plan")
    return selected, output


def _torch_cuda_initialized() -> bool | None:
    """Read already-loaded Torch state without creating allocations in the region."""

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    if not callable(is_initialized):
        return None
    try:
        return bool(is_initialized())
    except Exception:
        return None


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
