from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MAX_CANDIDATES = 32
_MAX_TIMINGS_PER_CANDIDATE = 3
_MAX_CONFIG_KWARGS = 32
_MAX_CONFIG_NAME_LENGTH = 100


def _digest(value: object) -> str:
    encoded = json.dumps(
        _digest_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _digest_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else type(value).__qualname__
    if isinstance(value, (list, tuple)):
        return [_digest_value(item) for item in value[:_MAX_CONFIG_KWARGS]]
    if isinstance(value, Mapping):
        return {
            key: _digest_value(value[key])
            for key in sorted(
                key for key in value if isinstance(key, str) and len(key) <= _MAX_CONFIG_NAME_LENGTH
            )[:_MAX_CONFIG_KWARGS]
        }
    return type(value).__qualname__


def _safe_value(value: object) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _digest(value)
    return None


def _config(value: object) -> dict[str, object]:
    kwargs = getattr(value, "kwargs", {})
    normalized_kwargs: dict[str, object] = {}
    if isinstance(kwargs, Mapping):
        ordered_kwargs = sorted(kwargs.items(), key=lambda pair: str(pair[0]))
        for name, item in ordered_kwargs[:_MAX_CONFIG_KWARGS]:
            text = str(name)
            if not text or len(text) > _MAX_CONFIG_NAME_LENGTH:
                continue
            normalized_kwargs[text] = _safe_value(item)
    return {
        "kwargs": normalized_kwargs,
        "num_warps": _safe_value(getattr(value, "num_warps", None)),
        "num_stages": _safe_value(getattr(value, "num_stages", None)),
        "num_ctas": _safe_value(getattr(value, "num_ctas", None)),
        "maxnreg": _safe_value(getattr(value, "maxnreg", None)),
        "ir_override": _safe_value(getattr(value, "ir_override", None)),
    }


def _timings(value: object) -> tuple[list[float | None], bool]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    truncated = len(values) > _MAX_TIMINGS_PER_CANDIDATE
    result: list[float | None] = []
    for item in values[:_MAX_TIMINGS_PER_CANDIDATE]:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            numeric = float(item)
            result.append(numeric if math.isfinite(numeric) else None)
        else:
            result.append(None)
    return result, truncated


def _candidate(config: object, timings: object) -> tuple[dict[str, object], bool]:
    values, truncated = _timings(timings)
    return {"config": _config(config), "timings_ms": values}, truncated


def _event(
    *,
    fn: object,
    key: object,
    best_config: object,
    configs_timings: object,
    duration: object,
    cache_hit: object,
) -> dict[str, object]:
    items = list(configs_timings.items()) if isinstance(configs_timings, Mapping) else []
    candidates: list[dict[str, object]] = []
    timings_truncated = False
    for config, timings in items:
        candidate, candidate_timings_truncated = _candidate(config, timings)
        candidates.append(candidate)
        timings_truncated = timings_truncated or candidate_timings_truncated
    candidates.sort(key=lambda candidate: _digest(candidate["config"]))
    winner = _config(best_config)
    winner_id = _digest(winner)
    selected = candidates[:_MAX_CANDIDATES]
    if all(_digest(candidate["config"]) != winner_id for candidate in selected) and candidates:
        selected[-1] = next(
            candidate for candidate in candidates if _digest(candidate["config"]) == winner_id
        )
        selected.sort(key=lambda candidate: _digest(candidate["config"]))
    duration_ms = None
    if (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
    ):
        duration_ms = float(duration) * 1_000
    module = getattr(fn, "__module__", "")
    name = getattr(fn, "__name__", type(fn).__name__)
    function_name = f"{module}.{name}".strip(".")[:200]
    return {
        "function_name": function_name,
        "key_digest": _digest(key),
        "cache_hit": bool(cache_hit),
        "duration_ms": duration_ms,
        "winner": winner,
        "candidate_count": len(candidates),
        "candidates_truncated": len(candidates) > len(selected),
        "timings_truncated": timings_truncated,
        "candidates": selected,
    }


def _write_event(path: Path, event: dict[str, object]) -> None:
    payload = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()


def _write_unavailable(path: Path, reason: str) -> None:
    _write_event(path, {"listener_unavailable": reason})


def _compiler_target(value: object) -> dict[str, object]:
    backend = getattr(value, "backend", None)
    arch = getattr(value, "arch", None)
    warp_size = getattr(value, "warp_size", None)
    if isinstance(value, Mapping):
        backend = value.get("backend", backend)
        arch = value.get("arch", value.get("architecture", arch))
        warp_size = value.get("warp_size", warp_size)
    if not isinstance(backend, str) or not backend or len(backend) > 100:
        raise ValueError("compiler target backend is invalid")
    if isinstance(arch, int) and not isinstance(arch, bool):
        architecture = f"sm_{arch}"
    elif isinstance(arch, str) and arch:
        architecture = arch if arch.startswith("sm_") else f"sm_{arch}"
    else:
        raise ValueError("compiler target architecture is invalid")
    if not isinstance(warp_size, int) or isinstance(warp_size, bool) or warp_size <= 0:
        raise ValueError("compiler target warp size is invalid")
    return {"backend": backend, "architecture": architecture, "warp_size": warp_size}


def _compiler_event(*, metadata: object, cache_hit: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError("compiler metadata is invalid")
    version = metadata.get("triton_version")
    if not isinstance(version, str) or not version or len(version) > 200:
        raise ValueError("compiler metadata version is invalid")
    return {
        "cache_hit": bool(cache_hit),
        "target": _compiler_target(metadata.get("target")),
        "triton_version": version,
    }


def _run_target(target: list[str]) -> None:
    if not target:
        raise SystemExit("triton compiler capture requires a Python script or module target")
    if target[0] == "-m":
        if len(target) < 2:
            raise SystemExit("triton compiler capture requires a module name after -m")
        sys.argv = [target[1], *target[2:]]
        runpy.run_module(target[1], run_name="__main__", alter_sys=True)
        return
    if target[0].startswith("-"):
        raise SystemExit(
            "triton compiler capture supports only a Python script or -m module target"
        )
    sys.argv = target
    runpy.run_path(target[0], run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--events", required=True)
    parser.add_argument("--compiler-events", required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    target = arguments.target
    if target[:1] == ["--"]:
        target = target[1:]
    events_path = Path(arguments.events)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.touch(exist_ok=True)
    compiler_events_path = Path(arguments.compiler_events)
    compiler_events_path.parent.mkdir(parents=True, exist_ok=True)
    compiler_events_path.touch(exist_ok=True)

    try:
        triton = importlib.import_module("triton")
    except ModuleNotFoundError:
        _write_unavailable(
            events_path,
            "Triton was unavailable in the root Python interpreter; no autotune selection "
            "evidence was collected.",
        )
        _write_unavailable(
            compiler_events_path,
            "Triton was unavailable in the root Python interpreter; compiler target evidence "
            "was not collected.",
        )
        _run_target(target)
        return

    try:
        autotuning = triton.knobs.autotuning
        previous_autotune_listener = autotuning.listener
    except AttributeError:
        _write_unavailable(
            events_path,
            "Installed Triton does not expose knobs.autotuning.listener; compiler artifacts "
            "were captured without autotune selection evidence.",
        )
        autotuning = None
        previous_autotune_listener = None

    try:
        compilation = triton.knobs.compilation
        previous_compiler_listener = compilation.listener
    except AttributeError:
        _write_unavailable(
            compiler_events_path,
            "Installed Triton does not expose knobs.compilation.listener; compiler target "
            "evidence was not collected.",
        )
        compilation = None
        previous_compiler_listener = None

    def listener(**kwargs: Any) -> None:
        with contextlib.suppress(OSError, TypeError, ValueError):
            _write_event(events_path, _event(**kwargs))
        if previous_autotune_listener is not None:
            previous_autotune_listener(**kwargs)

    def compiler_listener(**kwargs: Any) -> None:
        with contextlib.suppress(OSError, TypeError, ValueError):
            _write_event(compiler_events_path, _compiler_event(**kwargs))
        if previous_compiler_listener is not None:
            previous_compiler_listener(**kwargs)

    if autotuning is not None:
        autotuning.listener = listener
    if compilation is not None:
        compilation.listener = compiler_listener
    try:
        _run_target(target)
    finally:
        if autotuning is not None:
            autotuning.listener = previous_autotune_listener
        if compilation is not None:
            compilation.listener = previous_compiler_listener


if __name__ == "__main__":
    main()
