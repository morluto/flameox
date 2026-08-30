"""NVBench JSON + jsonbin sidecar import and extraction.

NVBench (https://github.com/NVIDIA/nvbench, pinned to main commit c184889)
emits a JSON document via ``--json <path>`` and optional binary sidecars via
``--jsonbin <path>``.  The sidecars contain ``sample_times`` and
``sample_freqs`` as little-endian float32 values.  The JSON references each
sidecar through a summary entry whose ``hint`` starts with ``file/`` and
whose ``data`` array carries the relative ``filename`` (e.g.
``out.json-bin/0.bin``) and float32 ``size`` (value count, serialized as a
decimal string because json_printer.cu writes all int64 values as strings).

This adapter preserves the native JSON and sidecar files through the bounded
bundle import primitive, then extracts sample measurements as float values
without lossy integer conversion.  NVBench reports times in seconds
(``cuda_timer.get_duration()`` in ``nvbench/detail/measure_cold.cu``) and
frequencies in hertz; the measurements table stores them in ``value_float``
with ``unit="seconds"`` or ``unit="hertz"`` respectively.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    digest_model,
    missing_artifact_input,
)
from flameox.domain.models import ArtifactRegistration
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

_NVBENCH_PRODUCER = "nvbench"
_SIDECAR_ROLE = "nvbench_sidecar"
_SAMPLE_TIMES_HINT = "file/sample_times"
_SAMPLE_FREQS_HINT = "file/sample_freqs"
_MAX_STATES = 100_000
_MAX_SUMMARIES_PER_STATE = 1_000
_MAX_SAMPLES_PER_SERIES = 1_000_000
_SUPPORTED_JSON_MAJOR = 1

BoundedString = Annotated[str, StringConstraints(min_length=1, max_length=500)]


def _parse_decimal_size(raw: str) -> int:
    """Parse a decimal string size from an NVBench int64 summary datum.

    Rejects empty strings, negative values, non-decimal characters, and
    values exceeding ``_MAX_SAMPLES_PER_SERIES``.
    """
    stripped = raw.strip()
    if not stripped or not stripped.isascii() or not stripped.isdigit():
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar size {raw!r} is not a non-negative decimal integer.",
        )
    value = int(stripped)
    if value > _MAX_SAMPLES_PER_SERIES:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar declares {value} samples, exceeding the limit.",
        )
    return value


class _NvbenchModel(ContractModel):
    """NVBench JSON fields are forward-compatible; allow unknown keys."""

    model_config = ConfigDict(extra="allow", frozen=True, validate_default=True)


class NvbenchSummaryDatum(_NvbenchModel):
    """A single named value within an NVBench summary's ``data`` array."""

    name: BoundedString
    type: Literal["int64", "float64", "string"]
    value: int | float | str


_VALID_FILE_HINTS = frozenset({_SAMPLE_TIMES_HINT, _SAMPLE_FREQS_HINT})


class NvbenchSummary(_NvbenchModel):
    """An NVBench summary entry within a benchmark state."""

    tag: BoundedString
    name: str | None = None
    description: str | None = None
    hint: str | None = None
    hide: str | None = None
    data: tuple[NvbenchSummaryDatum, ...] = Field(default_factory=tuple, max_length=100)

    @property
    def is_file_sidecar(self) -> bool:
        """True if this summary declares a binary sidecar via a known file hint."""
        return self.hint in _VALID_FILE_HINTS

    @property
    def sidecar_filename(self) -> str | None:
        """Return the relative sidecar filename if this summary references one.

        Only the two documented file hints (``file/sample_times`` and
        ``file/sample_freqs``) are recognised.  Any other ``file/`` hint
        is an unknown encoding and must be rejected by the caller.
        """
        if not self.is_file_sidecar:
            return None
        for datum in self.data:
            if (
                datum.name == "filename"
                and datum.type == "string"
                and isinstance(datum.value, str)
                and 1 <= len(datum.value) <= 500
            ):
                return datum.value
        return None

    @property
    def sidecar_size(self) -> int | None:
        """Return the declared float32 value count if this summary references a sidecar.

        NVBench's json_printer.cu serializes all int64 named values as
        decimal strings (JSON encodes numbers as double-precision floats,
        which would truncate int64s).  The ``size`` datum therefore arrives
        as a string like ``"27393"`` even though its ``type`` is
        ``"int64"``.  We parse the decimal string, rejecting negatives,
        non-decimal characters, and overflow beyond ``_MAX_SAMPLES_PER_SERIES``.
        """
        if not self.is_file_sidecar:
            return None
        for datum in self.data:
            if datum.name != "size":
                continue
            if datum.type == "int64" and isinstance(datum.value, str):
                return _parse_decimal_size(datum.value)
            return None
        return None


class NvbenchState(_NvbenchModel):
    """An NVBench benchmark execution state."""

    name: str | None = None
    device: int | None = None
    type_config_index: int | None = None
    is_skipped: bool = False
    skip_reason: str | None = None
    summaries: tuple[NvbenchSummary, ...] = Field(
        default_factory=tuple,
        max_length=_MAX_SUMMARIES_PER_STATE,
    )


class NvbenchBenchmark(_NvbenchModel):
    """An NVBench benchmark entry."""

    name: BoundedString
    index: int | None = None
    states: tuple[NvbenchState, ...] = Field(default_factory=tuple, max_length=_MAX_STATES)


class NvbenchJsonVersion(_NvbenchModel):
    """The NVBench JSON schema version."""

    major: int
    minor: int
    patch: int
    string: str | None = None


class NvbenchNvbenchVersion(_NvbenchModel):
    """The NVBench tool version."""

    major: int
    minor: int
    patch: int
    string: str | None = None


class NvbenchMeta(_NvbenchModel):
    """The NVBench JSON metadata section."""

    argv: tuple[str, ...] = Field(default_factory=tuple, max_length=10_000)
    version: dict[str, Any] = Field(default_factory=dict)


class NvbenchJsonDocument(_NvbenchModel):
    """A bounded subset of the NVBench JSON output document.

    Only the fields required for sample extraction and provenance are modeled.
    Unknown fields are ignored so that forward-compatible NVBench versions do
    not break extraction.
    """

    meta: NvbenchMeta = Field(default_factory=NvbenchMeta)
    benchmarks: tuple[NvbenchBenchmark, ...] = Field(
        default_factory=tuple,
        max_length=10_000,
    )

    @property
    def json_version(self) -> NvbenchJsonVersion | None:
        raw = self.meta.version.get("json")
        if not isinstance(raw, dict):
            return None
        try:
            return NvbenchJsonVersion.model_validate(raw)
        except ValidationError:
            return None

    @property
    def nvbench_version(self) -> NvbenchNvbenchVersion | None:
        raw = self.meta.version.get("nvbench")
        if not isinstance(raw, dict):
            return None
        try:
            return NvbenchNvbenchVersion.model_validate(raw)
        except ValidationError:
            return None


def require_supported_nvbench_schema(
    document: NvbenchJsonDocument,
    *,
    run_id: str | None = None,
) -> NvbenchJsonVersion:
    version = document.json_version
    if version is None or version.major != _SUPPORTED_JSON_MAJOR:
        declared = "missing" if version is None else str(version.major)
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Unsupported NVBench JSON schema major {declared}; only major "
            f"{_SUPPORTED_JSON_MAJOR} is verified.",
            run_id=run_id,
        )
    return version


class NvbenchExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    producer_version: str | None
    benchmark_count: int
    measurement_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class NvbenchExtractor:
    """Extract NVBench sample measurements from a preserved JSON + sidecar bundle.

    The run must contain exactly one primary NVBench JSON artifact (role
    ``"primary"``) and zero or more sidecar artifacts (role
    ``"nvbench_sidecar"``).  Sidecars are matched to JSON summary references
    by their display name (the relative filename stored in the JSON).
    """

    name = "nvbench.json"
    version = "1"
    compatibility_family = "nvbench.json.2026.v1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> NvbenchExtractionResult:
        run = self.runs.read(run_id)
        nvbench_artifacts = tuple(
            item
            for item in run.artifacts
            if item.kind is ArtifactKind.BENCHMARK_SAMPLES
            and item.producer in {_NVBENCH_PRODUCER, "flameox.import"}
        )
        primaries = tuple(item for item in nvbench_artifacts if item.role == "primary")
        if not primaries:
            raise missing_artifact_input(
                run_id=run_id,
                requirement="primary NVBench JSON",
                artifact_kinds=(ArtifactKind.BENCHMARK_SAMPLES.value,),
                capture_adapters=("nvbench",),
            )
        if len(primaries) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one primary NVBench JSON artifact.",
                run_id=run_id,
            )
        primary = primaries[0]
        sidecars = {
            item.display_name: item for item in nvbench_artifacts if item.role == _SIDECAR_ROLE
        }
        stored = self.artifacts.get(primary.artifact_id)
        document = self._load_json(stored.payload_path)
        producer_version = self._producer_version(document, primary)
        json_version = require_supported_nvbench_schema(document, run_id=run_id)
        summaries = tuple(self._iter_unique_sidecar_summaries(document, run_id=run_id))
        declared_rows = sum(size for _, _, _, _, size in summaries)
        max_rows = self.workspace.config.storage.max_rows_per_generation
        if declared_rows > max_rows:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "NVBench extraction exceeds the workspace generation row limit.",
                details={"rows": declared_rows, "max_rows": max_rows},
            )

        limitations: list[str] = []
        consumed_sidecar_ids: set[str] = set()
        if producer_version is None:
            limitations.append("The NVBench producer version was not declared.")

        rows: list[dict[str, object]] = []
        for benchmark_name, state_index, state, summary, _ in summaries:
            series_rows, sidecar_artifact_id = self._extract_summary(
                run_id=run_id,
                benchmark_name=benchmark_name,
                state_index=state_index,
                state=state,
                summary=summary,
                sidecars=sidecars,
            )
            rows.extend(series_rows)
            if sidecar_artifact_id is not None:
                consumed_sidecar_ids.add(sidecar_artifact_id)

        input_artifact_ids = (
            primary.artifact_id,
            *sorted(consumed_sidecar_ids - {primary.artifact_id}),
        )
        published = self.publisher.publish_rows_idempotent(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=input_artifact_ids,
            operation_identity={
                "compatibility_family": self.compatibility_family,
                "producer_version": producer_version,
                "json_version": (
                    f"{json_version.major}.{json_version.minor}.{json_version.patch}"
                    if json_version
                    else None
                ),
                "measurement_count": len(rows),
            },
        )
        return NvbenchExtractionResult(
            run_id=run_id,
            artifact_id=primary.artifact_id,
            producer_version=producer_version,
            benchmark_count=len(document.benchmarks),
            measurement_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(dict.fromkeys(limitations)),
        )

    def _extract_summary(
        self,
        *,
        run_id: str,
        benchmark_name: str,
        state_index: int,
        state: NvbenchState,
        summary: NvbenchSummary,
        sidecars: dict[str, ArtifactRegistration],
    ) -> tuple[list[dict[str, object]], str | None]:
        size = self._declared_sidecar_size(summary, run_id=run_id)
        if size is None:
            return [], None
        filename = summary.sidecar_filename
        if filename is None:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"NVBench summary with hint {summary.hint!r} is missing "
                "the required filename datum.",
                run_id=run_id,
            )
        registration = sidecars.get(filename)
        if registration is None:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"NVBench sidecar {filename!r} was not preserved in the run.",
                run_id=run_id,
            )
        stored = self.artifacts.get(registration.artifact_id)
        samples = self._decode_float32_sidecar(stored.payload_path, size)
        if any(not math.isfinite(value) for value in samples):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"NVBench sidecar {filename!r} contains a non-finite sample.",
                run_id=run_id,
            )
        unit = self._unit_for_hint(summary.hint)
        measurement_name = f"nvbench.{benchmark_name}.sample_times"
        if summary.hint == _SAMPLE_FREQS_HINT:
            measurement_name = f"nvbench.{benchmark_name}.sample_freqs"
        dimensions: dict[str, str] = {
            "nvbench_benchmark": benchmark_name,
            "nvbench_state_index": str(state_index),
        }
        if state.device is not None:
            dimensions["device.index"] = str(state.device)
        if state.name:
            dimensions["nvbench_state"] = state.name
        if summary.tag:
            dimensions["nvbench_summary_tag"] = summary.tag
        rows: list[dict[str, object]] = []
        for value_index, value in enumerate(samples):
            identity = {
                "run_id": run_id,
                "artifact_id": registration.artifact_id,
                "benchmark_name": benchmark_name,
                "state_index": state_index,
                "summary_tag": summary.tag,
                "value_index": value_index,
            }
            rows.append(
                {
                    "measurement_id": digest_model(identity),
                    "run_id": run_id,
                    "artifact_id": registration.artifact_id,
                    "name": measurement_name,
                    "value_int": None,
                    "value_float": float(value),
                    "unit": unit,
                    "aggregation": "sample",
                    "scope": "device",
                    "trial_id": None,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": value_index,
                    "loop_count": None,
                    "is_warmup": False,
                    "block_id": None,
                    "variant_id": None,
                    "order_in_block": None,
                    "phase": None,
                    "dimensions": dimensions,
                    "evidence_level": "observed",
                }
            )
        return rows, registration.artifact_id

    @classmethod
    def _iter_unique_sidecar_summaries(
        cls,
        document: NvbenchJsonDocument,
        *,
        run_id: str,
    ) -> Iterator[tuple[str, int, NvbenchState, NvbenchSummary, int]]:
        seen: dict[str, tuple[str, int]] = {}
        for benchmark in document.benchmarks:
            for state_index, state in enumerate(benchmark.states):
                if state.is_skipped:
                    continue
                for summary in state.summaries:
                    size = cls._declared_sidecar_size(summary, run_id=run_id)
                    if size is None:
                        continue
                    filename = summary.sidecar_filename
                    if filename is None:
                        raise DomainError(
                            ErrorCode.ARTIFACT_PARSE_FAILED,
                            f"NVBench summary with hint {summary.hint!r} is missing "
                            "the required filename datum.",
                            run_id=run_id,
                        )
                    identity = (summary.hint or "", size)
                    previous = seen.get(filename)
                    if previous is not None:
                        if previous != identity:
                            raise DomainError(
                                ErrorCode.ARTIFACT_PARSE_FAILED,
                                f"NVBench document reuses sidecar {filename!r} with "
                                "conflicting sizes or sample hints.",
                                run_id=run_id,
                            )
                        continue
                    seen[filename] = identity
                    yield benchmark.name, state_index, state, summary, size

    @staticmethod
    def _declared_sidecar_size(summary: NvbenchSummary, *, run_id: str) -> int | None:
        if summary.hint is None or not summary.hint.startswith("file/"):
            return None
        if not summary.is_file_sidecar:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"Unknown NVBench file hint {summary.hint!r}; "
                "only file/sample_times and file/sample_freqs are supported.",
                run_id=run_id,
            )
        size = summary.sidecar_size
        if size is None:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"NVBench summary with hint {summary.hint!r} is missing "
                "a valid decimal-string size datum.",
                run_id=run_id,
            )
        return size

    @staticmethod
    def _decode_float32_sidecar(path: Path, count: int) -> list[float]:
        expected_bytes = count * 4
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "NVBench sidecar payload could not be read.",
            ) from exc
        if len(raw) != expected_bytes:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"NVBench sidecar size mismatch: expected {expected_bytes} bytes, got {len(raw)}.",
            )
        return list(struct.unpack(f"<{count}f", raw))

    @staticmethod
    def _unit_for_hint(hint: str | None) -> str:
        """Return the SI unit for a sidecar hint.

        Units are pinned to NVIDIA/nvbench main (commit c184889):

        - ``file/sample_times``: ``cuda_timer.get_duration()`` returns
          seconds (``nvbench/detail/measure_cold.cu`` feeds
          ``m_cuda_times`` to ``do_process_bulk_data_float64`` with
          ``hint="sample_times"``).  The ``stdrel_criterion`` compares
          ``total_measured_time`` against ``min-time`` (default 0.5 s),
          confirming seconds.
        - ``file/sample_freqs``: sample frequencies are raw hertz
          (1/seconds), written via the same bulk-data path with
          ``hint="sample_freqs"``.
        """
        if hint == _SAMPLE_TIMES_HINT:
            return "seconds"
        if hint == _SAMPLE_FREQS_HINT:
            return "hertz"
        return "unknown"

    @staticmethod
    def _producer_version(
        document: NvbenchJsonDocument,
        registration: ArtifactRegistration,
    ) -> str | None:
        nvbench_version = document.nvbench_version
        if nvbench_version is not None and nvbench_version.string:
            return nvbench_version.string
        if nvbench_version is not None:
            return f"{nvbench_version.major}.{nvbench_version.minor}.{nvbench_version.patch}"
        return registration.producer_version

    @staticmethod
    def _load_json(path: Path) -> NvbenchJsonDocument:
        try:
            with path.open(encoding="utf-8") as stream:
                return NvbenchJsonDocument.model_validate(json.load(stream))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a valid NVBench JSON document.",
            ) from exc
