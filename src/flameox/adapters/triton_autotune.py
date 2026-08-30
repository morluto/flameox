from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import Field, JsonValue

from flameox.domain import canonical_json, digest_model
from flameox.models import ContractModel

_MAX_EVENTS = 128
_MAX_EVENT_BYTES = 64 * 1024
_MAX_CANDIDATES = 32
_MAX_TIMINGS_PER_CANDIDATE = 3


class TritonAutotuneCandidate(ContractModel):
    config_id: str
    config: dict[str, JsonValue]
    timings_ms: tuple[float | None, ...] = Field(max_length=_MAX_TIMINGS_PER_CANDIDATE)


class TritonAutotuneSelection(ContractModel):
    selection_id: str
    run_id: str
    function_name: str
    key_digest: str
    cache_hit: bool
    duration_ms: float | None
    winner_config_id: str
    candidate_count: int = Field(ge=1)
    candidates_truncated: bool
    candidates: tuple[TritonAutotuneCandidate, ...] = Field(max_length=_MAX_CANDIDATES)
    limitations: tuple[str, ...] = ()

    def row(self) -> dict[str, object]:
        return {
            "selection_id": self.selection_id,
            "run_id": self.run_id,
            "function_name": self.function_name,
            "key_digest": self.key_digest,
            "cache_hit": self.cache_hit,
            "duration_ms": self.duration_ms,
            "winner_config_id": self.winner_config_id,
            "candidate_count": self.candidate_count,
            "candidates_truncated": self.candidates_truncated,
            "candidates_json": canonical_json(
                [candidate.model_dump(mode="json") for candidate in self.candidates]
            ),
            "limitations": list(self.limitations),
        }


def load_triton_autotune_selections(
    path: Path,
    *,
    run_id: str,
) -> tuple[tuple[TritonAutotuneSelection, ...], tuple[str, ...]]:
    """Read bounded listener output without treating it as a provider-native artifact."""
    if not path.is_file():
        return (), ("Triton autotune listener output was unavailable after capture.",)
    selections: list[TritonAutotuneSelection] = []
    limitations: list[str] = []
    skipped_invalid = 0
    skipped_oversized = 0
    skipped_duplicate = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if len(selections) >= _MAX_EVENTS:
                    limitations.append(
                        f"Triton autotune evidence is limited to {_MAX_EVENTS} selections per run."
                    )
                    break
                if len(line.encode("utf-8")) > _MAX_EVENT_BYTES:
                    skipped_oversized += 1
                    continue
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    unavailable = _listener_unavailable(value)
                    if unavailable is not None:
                        limitations.append(unavailable)
                        continue
                    selection = _selection(value, run_id=run_id)
                except (TypeError, ValueError, json.JSONDecodeError):
                    skipped_invalid += 1
                    continue
                if any(item.selection_id == selection.selection_id for item in selections):
                    skipped_duplicate += 1
                    continue
                selections.append(selection)
    except OSError:
        return (), ("Triton autotune listener output could not be read after capture.",)
    if skipped_oversized:
        limitations.append(
            f"{skipped_oversized} oversized Triton autotune listener event(s) were omitted."
        )
    if skipped_invalid:
        limitations.append(
            f"{skipped_invalid} invalid Triton autotune listener event(s) were omitted."
        )
    if skipped_duplicate:
        limitations.append(
            f"{skipped_duplicate} duplicate Triton autotune listener event(s) were deduplicated."
        )
    if not selections and not limitations:
        limitations.append(
            "No multi-configuration Triton autotune decision was observed in the root Python "
            "process."
        )
    return tuple(selections), tuple(limitations)


def _listener_unavailable(value: object) -> str | None:
    if not isinstance(value, Mapping) or set(value) != {"listener_unavailable"}:
        return None
    reason = value["listener_unavailable"]
    if not isinstance(reason, str) or not reason or len(reason) > 300:
        raise ValueError("listener availability marker is invalid")
    return reason


def _selection(value: object, *, run_id: str) -> TritonAutotuneSelection:
    event = _mapping(value, "listener event")
    function_name = _text(event, "function_name", maximum=200)
    key_digest = _digest(event, "key_digest")
    cache_hit = _bool(event, "cache_hit")
    duration_ms = _finite_number_or_none(event.get("duration_ms"), "duration_ms")
    winner_config = _config(event.get("winner"))
    winner_config_id = digest_model(winner_config)
    candidates_value = event.get("candidates")
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ValueError("listener event has no candidates")
    if len(candidates_value) > _MAX_CANDIDATES:
        raise ValueError("listener event has too many candidates")
    candidates = tuple(_candidate(item) for item in candidates_value)
    candidate_ids = [item.config_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("listener event repeats a candidate configuration")
    if winner_config_id not in candidate_ids:
        raise ValueError("listener event does not retain the selected candidate")
    candidate_count = _integer(event, "candidate_count", minimum=len(candidates))
    candidates_truncated = _bool(event, "candidates_truncated")
    if candidates_truncated != (candidate_count > len(candidates)):
        raise ValueError("listener candidate truncation is inconsistent")
    event_limitations: list[str] = []
    if _bool(event, "timings_truncated"):
        event_limitations.append(
            "One or more provider timing vectors exceeded the retained timing bound."
        )
    if cache_hit and duration_ms is not None:
        raise ValueError("cache-hit listener event reports a tuning duration")
    if not cache_hit and duration_ms is None:
        event_limitations.append("Triton did not report a finite autotuning duration.")
    payload = {
        "run_id": run_id,
        "function_name": function_name,
        "key_digest": key_digest,
        "cache_hit": cache_hit,
        "duration_ms": duration_ms,
        "winner_config_id": winner_config_id,
        "candidate_count": candidate_count,
        "candidates_truncated": candidates_truncated,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    return TritonAutotuneSelection(
        selection_id=digest_model(payload),
        run_id=run_id,
        function_name=function_name,
        key_digest=key_digest,
        cache_hit=cache_hit,
        duration_ms=duration_ms,
        winner_config_id=winner_config_id,
        candidate_count=candidate_count,
        candidates_truncated=candidates_truncated,
        candidates=candidates,
        limitations=tuple(event_limitations),
    )


def _candidate(value: object) -> TritonAutotuneCandidate:
    candidate = _mapping(value, "candidate")
    config = _config(candidate.get("config"))
    timings = candidate.get("timings_ms")
    if not isinstance(timings, list) or not timings or len(timings) > _MAX_TIMINGS_PER_CANDIDATE:
        raise ValueError("candidate timings are invalid")
    return TritonAutotuneCandidate(
        config_id=digest_model(config),
        config=config,
        timings_ms=tuple(_finite_number_or_none(item, "timing") for item in timings),
    )


def _config(value: object) -> dict[str, JsonValue]:
    config = _mapping(value, "configuration")
    required = {"kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg", "ir_override"}
    if set(config) != required:
        raise ValueError("configuration fields are invalid")
    kwargs = _mapping(config["kwargs"], "configuration kwargs")
    if len(kwargs) > 32:
        raise ValueError("configuration has too many keyword values")
    normalized_kwargs: dict[str, JsonValue] = {}
    for name, item in kwargs.items():
        if not isinstance(name, str) or not name or len(name) > 100:
            raise ValueError("configuration keyword name is invalid")
        normalized_kwargs[name] = _config_value(item)
    normalized: dict[str, JsonValue] = {"kwargs": normalized_kwargs}
    for name in ("num_warps", "num_stages", "num_ctas", "maxnreg", "ir_override"):
        normalized[name] = _config_value(config[name])
    return normalized


def _config_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return cast(JsonValue, value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("configuration value is invalid")


def _mapping(value: object, subject: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{subject} is not an object")
    return cast(Mapping[str, object], value)


def _text(value: Mapping[str, object], key: str, *, maximum: int) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item or len(item) > maximum:
        raise ValueError(f"{key} is invalid")
    return item


def _digest(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.startswith("sha256:") or len(item) != 71:
        raise ValueError(f"{key} is invalid")
    return item


def _bool(value: Mapping[str, object], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"{key} is invalid")
    return item


def _integer(value: Mapping[str, object], key: str, *, minimum: int) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < minimum or item > 2**31 - 1:
        raise ValueError(f"{key} is invalid")
    return item


def _finite_number_or_none(value: object, subject: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{subject} is invalid")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{subject} is invalid")
    return numeric
