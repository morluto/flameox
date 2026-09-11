"""Bounded process-lifespan analysis runtime."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import statistics
import sys
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import anyio
import ijson
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import TypeAdapter

from flameox import __version__
from flameox.adapters.json_preview import iter_json_rows
from flameox.adapters.text_preview import iter_text_fragments
from flameox.canonical import canonical_bytes
from flameox.command_binding import ExecutableResolver
from flameox.evidence_models import ConsoleDiagnostics, OutputStreams
from flameox.executable_models import ResolvedExecutable
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    OutputSink,
    ProcessExecutionError,
    ResourcePolicy,
    SubprocessBroker,
)
from flameox.paths import default_data_directory
from flameox.process_models import ProcessResult, process_exit_code
from flameox.providers.aiperf import AIPerfProvider
from flameox.providers.availability import (
    SYSTEM_PROVIDER_GUIDANCE,
    WORKLOAD_PYTHON_REQUIREMENTS,
)
from flameox.providers.benchmarks import BenchmarkProvider
from flameox.providers.capture import (
    CaptureInvocation,
    build_capture_invocation,
    materialize_capture_support,
)
from flameox.providers.contracts import (
    ProviderAnalysis,
    ProviderFailure,
    canonical_provider_projection,
)
from flameox.providers.cpu import CpuProfileProvider
from flameox.providers.inference_exports import InferenceExportProvider
from flameox.providers.kernel_evidence import KernelEvidenceProvider
from flameox.providers.memray import MemrayProvider
from flameox.providers.nsight_compute import NsightComputeProvider
from flameox.providers.nsight_systems import NsightSystemsParquetProvider
from flameox.providers.nvbench import NvbenchProvider
from flameox.providers.otlp import OtlpProvider
from flameox.providers.perfetto import PerfettoProvider
from flameox.providers.preparation import ProviderDependencies
from flameox.providers.reliability import ReliabilityProvider
from flameox.providers.source_evidence import SourceEvidenceProvider
from flameox.providers.structured_workers import StructuredWorkerProviders
from flameox.providers.xctrace import XctraceProvider
from flameox.repository import (
    EvidenceRepository,
    EvidenceSelection,
    RepositoryError,
)
from flameox.runtime_contracts import (
    CAPABILITY_BY_ID,
    CAPTURE_PROVIDER_CONTRACTS,
    MAX_INPUTS,
    MAX_ROWS,
    SEMANTIC_ORACLE_STDERR_ENV,
    SEMANTIC_ORACLE_STDOUT_ENV,
    AnalysisResult,
    Capability,
    CaptureArguments,
    CaptureTarget,
    CpuHotspotArguments,
    EvidenceSource,
    ExperimentCase,
    ExperimentDesign,
    PathSource,
    PreviewArguments,
    PySpyCaptureArguments,
    RequestLimits,
    RuntimeFailure,
    Source,
    WorkloadBudget,
    compatible_capture_providers,
)
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.source_files import (
    NativeSource,
    copy_verified_file,
    directory_files,
    hash_path,
    sha256_file,
)
from flameox.workers.harness import IsolatedWorkerHarness, WorkerRuntimeConfig

MAX_SESSION_ANALYSES = 64
MAX_SESSION_PROJECTIONS = 16
MAX_SESSION_PROJECTION_BYTES = 16 * 1024 * 1024
MAX_SESSION_RESCUES = 64
MAX_SESSION_SCRATCH_BYTES = 1024**3
MAX_SESSION_SCRATCH_FILES = 8192


@dataclass(slots=True)
class CachedAnalysis:
    result: dict[str, Any]
    sources: list[NativeSource]
    manifest_body: dict[str, Any]
    preserved: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ValidatedCaptureRequest:
    capability: Capability
    capture_arguments: CaptureArguments
    limits: RequestLimits
    cases: list[ExperimentCase]
    blocks: int


@dataclass(frozen=True, slots=True)
class BoundCapture:
    sequence_number: int
    block: int
    case: ExperimentCase
    argv: list[str]
    environment: dict[str, str]
    directory: Path
    invocation: CaptureInvocation
    collector_binding: ResolvedExecutable
    workload_binding: ResolvedExecutable


class AnalysisRuntime:
    """Own registry, broker, scratch artifacts, and the session analysis cache."""

    def __init__(
        self,
        *,
        evidence_directory: Path | None = None,
        limits: RequestLimits | None = None,
    ) -> None:
        self.limits = limits or RequestLimits()
        self.session_id = f"{os.getpid()}-{secrets.token_hex(8)}"
        self._temporary = tempfile.TemporaryDirectory(prefix="flameox-session-")
        self.scratch = Path(self._temporary.name)
        self.broker = SubprocessBroker()
        self.workers = IsolatedWorkerHarness(
            WorkerRuntimeConfig(
                working_directory=self.scratch,
                staging_root=self.scratch,
                filesystem_path=self.scratch,
                maximum_rss_bytes=self.limits.max_memory_bytes,
                max_response_bytes=min(self.limits.max_output_bytes, 4 * 1024 * 1024),
            ),
            broker=self.broker,
        )
        self.perfetto = PerfettoProvider(self.workers)
        self.aiperf = AIPerfProvider(self.workers)
        self.benchmarks = BenchmarkProvider(self.workers)
        self.cpu_profiles = CpuProfileProvider()
        self.inference_exports = InferenceExportProvider()
        self.kernel_evidence = KernelEvidenceProvider()
        self.memray = MemrayProvider(self.workers)
        self.nvbench = NvbenchProvider()
        self.nsight_compute = NsightComputeProvider(self.workers)
        self.nsight_systems = NsightSystemsParquetProvider()
        self.otlp = OtlpProvider(self.workers)
        self.reliability = ReliabilityProvider()
        self.source_evidence = SourceEvidenceProvider(self.workers)
        self.structured_workers = StructuredWorkerProviders(self.workers)
        self.xctrace = XctraceProvider()
        self.repository = EvidenceRepository(
            evidence_directory or default_data_directory(), self.session_id
        )
        self._repository_configuration = (
            "explicit_directory"
            if evidence_directory is not None
            else "environment_override"
            if os.environ.get("FLAMEOX_DATA_DIR")
            else "platform_default"
        )
        self.analyses: OrderedDict[str, CachedAnalysis] = OrderedDict()
        self.projections: OrderedDict[str, tuple[ProviderAnalysis, int]] = OrderedDict()
        self.rescues: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self.scratch_artifacts: OrderedDict[tuple[str, str], Path] = OrderedDict()
        self._protected_sources: set[Path] = set()
        self._capture_reservations: dict[Path, tuple[int, int]] = {}
        self._request_lock = anyio.Lock()
        self.dependencies = ProviderDependencies(
            self.broker, self.scratch, state_lock=self._request_lock
        )

    async def run_in_request[T](self, operation: Callable[[], T]) -> T:
        """Join blocking state work before releasing its ownership to another request."""
        result: Future[T] = Future()

        async def execute() -> None:
            try:
                async with self._request_lock:
                    result.set_result(await anyio.to_thread.run_sync(operation))
            except BaseException as error:
                # Preserve domain exceptions without adding an ExceptionGroup envelope.
                result.set_exception(error)

        # The child owns the thread wait. Cancelling the caller cancels this group
        # and joins that wait, rather than abandoning shared state on Task.cancel().
        async with anyio.create_task_group() as group:
            group.start_soon(execute)
        return result.result()

    def close(self) -> None:
        self._temporary.cleanup()

    def _cache_analysis(self, analysis_id: str, cached: CachedAnalysis) -> None:
        self.analyses[analysis_id] = cached
        self.analyses.move_to_end(analysis_id)
        while len(self.analyses) > MAX_SESSION_ANALYSES:
            self._evict_oldest_analysis()

    def _cached_projection(self, key: str) -> ProviderAnalysis | None:
        projections = getattr(self, "projections", None)
        if projections is None:
            projections = self.projections = OrderedDict()
        cached = projections.get(key)
        if cached is None:
            return None
        projections.move_to_end(key)
        return self._copy_provider_analysis(cached[0])

    def _cache_projection(self, key: str, analysis: ProviderAnalysis) -> None:
        projections = getattr(self, "projections", None)
        if projections is None:
            projections = self.projections = OrderedDict()
        copied = self._copy_provider_analysis(analysis)
        size = len(
            canonical_bytes(
                {
                    "provider_id": copied.provider_id,
                    "provider_version": copied.provider_version,
                    "blocks": copied.blocks,
                    "rows_observed": copied.rows_observed,
                    "complete": copied.complete,
                    "limitations": copied.limitations,
                }
            )
        )
        if size > MAX_SESSION_PROJECTION_BYTES:
            return
        projections[key] = (copied, size)
        projections.move_to_end(key)
        while (
            len(projections) > MAX_SESSION_PROJECTIONS
            or sum(item[1] for item in projections.values()) > MAX_SESSION_PROJECTION_BYTES
        ):
            projections.popitem(last=False)

    @staticmethod
    def _copy_provider_analysis(analysis: ProviderAnalysis) -> ProviderAnalysis:
        return ProviderAnalysis(
            provider_id=analysis.provider_id,
            provider_version=analysis.provider_version,
            blocks=cast(list[dict[str, Any]], json.loads(json.dumps(analysis.blocks))),
            rows_observed=analysis.rows_observed,
            complete=analysis.complete,
            limitations=list(analysis.limitations),
        )

    def _evict_oldest_analysis(self, *, protected_roots: Sequence[Path] = ()) -> bool:
        selected = next(
            (
                analysis_id
                for analysis_id, cached in self.analyses.items()
                if not any(
                    source.path.is_relative_to(root) or root.is_relative_to(source.path)
                    for source in cached.sources
                    for root in protected_roots
                )
            ),
            None,
        )
        if selected is None:
            return False
        cached = self.analyses.pop(selected)
        self._release_analysis_scratch(cached)
        return True

    def _release_analysis_scratch(self, cached: CachedAnalysis) -> None:
        capture_roots: set[Path] = set()
        for source in cached.sources:
            try:
                relative = source.path.relative_to(self.scratch)
            except ValueError:
                continue
            if relative.parts and relative.parts[0].startswith("capture-"):
                capture_roots.add(self.scratch / relative.parts[0])
            elif len(relative.parts) >= 2 and relative.parts[0] == "evidence-sources":
                capture_roots.add(self.scratch / relative.parts[0] / relative.parts[1])
        for root in capture_roots:
            retained = any(
                source.path.is_relative_to(root) or root.is_relative_to(source.path)
                for analysis in self.analyses.values()
                if analysis is not cached
                for source in analysis.sources
            )
            if not retained and not any(
                path.is_relative_to(root) or root.is_relative_to(path)
                for path in self._protected_sources | self._capture_reservations.keys()
            ):
                shutil.rmtree(root, ignore_errors=True)
                for key, path in list(self.scratch_artifacts.items()):
                    if path.is_relative_to(root):
                        del self.scratch_artifacts[key]

    @staticmethod
    def _remove_scratch_artifact(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def _cache_scratch_artifact(self, key: tuple[str, str], path: Path) -> None:
        self.scratch_artifacts[key] = path
        self.scratch_artifacts.move_to_end(key)
        try:
            self._prune_scratch(protected_root=path)
        except RuntimeFailure:
            self.scratch_artifacts.pop(key, None)
            self._remove_scratch_artifact(path)
            raise

    def _prune_scratch(
        self,
        *,
        protected_root: Path | None = None,
        reserved_bytes: int = 0,
        reserved_files: int = 0,
    ) -> None:
        used_bytes, used_files = self._scratch_commitment()
        protected = (
            self._protected_sources
            | self._capture_reservations.keys()
            | ({protected_root} if protected_root is not None else set())
        )
        while (
            used_bytes + reserved_bytes > MAX_SESSION_SCRATCH_BYTES
            or used_files + reserved_files > MAX_SESSION_SCRATCH_FILES
        ):
            if self._evict_oldest_analysis(protected_roots=tuple(protected)):
                used_bytes, used_files = self._scratch_commitment()
                continue
            removable = next(
                (
                    key
                    for key, path in self.scratch_artifacts.items()
                    if not any(
                        path.is_relative_to(root) or root.is_relative_to(path) for root in protected
                    )
                ),
                None,
            )
            if removable is None:
                break
            self._remove_scratch_artifact(self.scratch_artifacts.pop(removable))
            used_bytes, used_files = self._scratch_commitment()
        if (
            used_bytes + reserved_bytes > MAX_SESSION_SCRATCH_BYTES
            or used_files + reserved_files > MAX_SESSION_SCRATCH_FILES
        ):
            raise RuntimeFailure("LIMIT_EXCEEDED", "Request exceeded the session scratch ceiling")

    def analyze(
        self,
        capability_id: str,
        sources: Sequence[Source],
        arguments: Mapping[str, Any],
        *,
        limits: RequestLimits | None = None,
        continuation: str | None = None,
    ) -> dict[str, Any]:
        protected = self._protected_sources.copy()
        retained_artifacts = set(self.scratch_artifacts)
        try:
            return self._analyze(
                capability_id, sources, arguments, limits=limits, continuation=continuation
            )
        except BaseException:
            for key in self.scratch_artifacts.keys() - retained_artifacts:
                path = self.scratch_artifacts[key]
                if not any(
                    source.path.is_relative_to(path)
                    for cached in self.analyses.values()
                    for source in cached.sources
                ):
                    self._remove_scratch_artifact(self.scratch_artifacts.pop(key))
            raise
        finally:
            self._protected_sources = protected

    def _analyze(
        self,
        capability_id: str,
        sources: Sequence[Source],
        arguments: Mapping[str, Any],
        *,
        limits: RequestLimits | None = None,
        continuation: str | None = None,
    ) -> dict[str, Any]:
        capability = CAPABILITY_BY_ID.get(capability_id)
        if capability is None:
            raise RuntimeFailure(
                "UNKNOWN_CAPABILITY",
                f"Unknown capability: {capability_id}",
                details={
                    "requested_capability": capability_id,
                    "available_capabilities": sorted(CAPABILITY_BY_ID),
                    "recovery": "Run `flameox mcp inspect --summary` to select a capability.",
                },
            )
        capability.validate_source_count(len(sources))
        selected_limits = limits.lowered_against(self.limits) if limits else self.limits
        validated = TypeAdapter(capability.model).validate_python(arguments)
        resolved = self._resolve_sources(sources, selected_limits)
        text_fragment_chars = (
            validated.text_fragment_chars if isinstance(validated, PreviewArguments) else None
        )
        if text_fragment_chars is not None and any(
            source.format != "text" or source.path.is_dir() for source in resolved
        ):
            raise RuntimeFailure(
                "INVALID_INPUT", "text_fragment_chars requires text file sources only."
            )
        if capability.id != "artifact.preview" and (
            bad := next(
                (item.format for item in resolved if item.format not in capability.formats), None
            )
        ):
            analysis_tool = f"analyze_{capability_id.replace('.', '_')}"
            raise RuntimeFailure(
                "UNSUPPORTED_FORMAT",
                f"{capability_id} does not accept artifact format {bad!r}",
                details={
                    "capability_id": capability_id,
                    "received_format": bad,
                    "accepted_formats": list(capability.formats),
                    "recovery": (
                        f"Select one accepted format or inspect `{analysis_tool}` "
                        f"with `flameox mcp inspect --tool {analysis_tool}`."
                    ),
                },
            )
        if (
            isinstance(validated, CpuHotspotArguments)
            and validated.metric is not None
            and any(item.format != "pstats" for item in resolved)
        ):
            raise RuntimeFailure(
                "INVALID_INPUT",
                "cpu.hotspots metric selection is supported only for pstats artifacts",
                details={
                    "capability_id": capability.id,
                    "option": "metric",
                    "accepted_formats": ["pstats"],
                },
            )
        identity = {
            "capability_id": capability_id,
            "inputs": [
                {
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "format": item.format,
                    "producer": item.producer,
                    "role": item.role,
                }
                for item in resolved
            ],
            "arguments": validated.model_dump(mode="json"),
            "limits": selected_limits.model_dump(mode="json"),
            "projection_implementation": self._projection_runtime_identity(capability_id, resolved),
        }
        default_offset = validated.offset if isinstance(validated, PreviewArguments) else 0
        offset = self._decode_continuation(continuation, identity, default_offset)
        analysis_request = {**identity, "offset": offset}
        analysis_id = hashlib.sha256(
            self.session_id.encode() + canonical_bytes(analysis_request)
        ).hexdigest()
        if cached := self.analyses.get(analysis_id):
            self.analyses.move_to_end(analysis_id)
            return self._copy_result(cached.result)
        provider_analysis = self._provider_projection(
            identity,
            capability_id=capability_id,
            sources=resolved,
            arguments=validated.model_dump(mode="json"),
            limits=selected_limits,
        )
        if provider_analysis is None and capability.id != "artifact.preview":
            raise RuntimeFailure(
                "UNSUPPORTED_FORMAT",
                f"No typed {capability_id} provider accepts the supplied inputs",
            )
        if provider_analysis is None:
            try:
                rows, observed, complete = self._read_rows(
                    resolved,
                    offset,
                    selected_limits.max_rows,
                    text_fragment_chars=text_fragment_chars,
                )
            except (ijson.JSONError, json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeFailure(
                    "DECODE_FAILURE", "Artifact preview could not decode the input."
                ) from error
            if (continuation is not None or offset > 0) and offset >= observed:
                raise RuntimeFailure("INVALID_INPUT", "Offset is beyond the available evidence")
            continuation_available = not complete
            truncation_reason = "row_limit"
            blocks: list[dict[str, Any]] = [
                {
                    "type": "metrics",
                    "values": {"input_count": len(resolved), "row_count": len(rows)},
                },
                {"type": "table", "rows": rows},
            ]
            provider_identity = {"id": "flameox", "version": __version__}
            limitations = [capability.limitation]
            if text_fragment_chars is not None:
                limitations.append(
                    "Text fragments use UTF-8 replacement decoding and LF-delimited lines; "
                    "offsets count fragments, not bytes. "
                    "Native artifacts retain the original bytes."
                )
        else:
            provider_rows = provider_analysis.blocks[-1]["rows"]
            if not isinstance(provider_rows, list):
                raise RuntimeFailure("DECODE_FAILURE", "Provider table block is invalid")
            if continuation is not None and offset >= len(provider_rows):
                raise RuntimeFailure(
                    "INVALID_INPUT", "Continuation offset is beyond the available evidence"
                )
            rows = provider_rows[offset : offset + selected_limits.max_rows]
            observed = provider_analysis.rows_observed
            next_offset = offset + len(rows)
            continuation_available = next_offset < len(provider_rows)
            complete = provider_analysis.complete and not continuation_available
            truncation_reason = "row_limit" if continuation_available else "provider_limit"
            blocks = [*provider_analysis.blocks[:-1], {"type": "table", "rows": rows}]
            provider_identity = {
                "id": provider_analysis.provider_id,
                "version": provider_analysis.provider_version,
            }
            limitations = provider_analysis.limitations
        for source in resolved:
            current_digest, current_size, _ = hash_path(
                source.path,
                max_bytes=selected_limits.max_input_bytes,
                max_files=selected_limits.max_input_files,
            )
            if (current_digest, current_size) != (source.sha256, source.size_bytes):
                raise RuntimeFailure(
                    "MISSING_OR_CHANGED_INPUT", f"Input changed during analysis: {source.path}"
                )
        result: dict[str, Any] = {
            "analysis_id": analysis_id,
            "capability_id": capability_id,
            "provider": provider_identity,
            "inputs": [item.public() for item in resolved],
            "blocks": blocks,
            "coverage": {
                "rows_returned": len(rows),
                "rows_observed": observed,
                "complete": complete,
            },
            "truncation": None,
            "limitations": limitations,
            "continuation": None,
        }
        if not complete:
            if continuation_available:
                result["continuation"] = self._encode_continuation(identity, offset + len(rows))
            result["truncation"] = {
                "reason": truncation_reason,
                "next_offset": offset + len(rows),
            }
        self._shrink_result(result, selected_limits.max_result_bytes, identity, offset)
        body = {
            "evidence_kind": "analysis",
            "capability_id": capability_id,
            "provider": result["provider"],
            "inputs": [
                {
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "format": item.format,
                    "role": item.role,
                }
                for item in resolved
            ],
            "capture_request": None,
            "analysis_request": analysis_request,
            "episode": {"created_at": datetime.now(UTC).isoformat()},
            "coverage": result["coverage"],
            "limitations": result["limitations"],
        }
        validated_result = AnalysisResult.model_validate(result).model_dump(
            mode="json", exclude_none=False
        )
        self._cache_analysis(
            analysis_id,
            CachedAnalysis(self._copy_result(validated_result), resolved, body),
        )
        return self._copy_result(validated_result)

    def _provider_projection(
        self,
        identity: Mapping[str, Any],
        *,
        capability_id: str,
        sources: list[NativeSource],
        arguments: Mapping[str, Any],
        limits: RequestLimits,
    ) -> ProviderAnalysis | None:
        projection_key = hashlib.sha256(canonical_bytes(identity)).hexdigest()
        cached = self._cached_projection(projection_key)
        if cached is not None:
            return cached
        projected = canonical_provider_projection(
            self._provider_analysis(
                capability_id,
                sources,
                arguments,
                # Every page slices the same bounded provider population.
                max_rows=MAX_ROWS + 1,
                limits=limits,
            )
        )
        if projected is not None:
            self._cache_projection(projection_key, projected)
        return projected

    def _projection_runtime_identity(
        self, capability_id: str, sources: Sequence[NativeSource]
    ) -> dict[str, str]:
        identity = {"flameox": __version__}
        required_tools: list[tuple[str, NativeSource]] = []
        for source in sources:
            if source.format == "perf-data" and capability_id in {
                "cpu.hotspots",
                "cpu.callers",
            }:
                required_tools.append(("perf", source))
            elif source.format == "nsys-rep" and capability_id in {
                "trace.summary",
                "trace.operations",
                "trace.lifecycle",
                "gpu.launches",
            }:
                required_tools.append(("nsys", source))
            elif source.format == "xctrace" and capability_id == "trace.summary":
                required_tools.append(("xcrun", source))
        for executable, source in required_tools:
            binding = self._require_host_tool(
                executable,
                cwd=source.path.parent,
                environment=dict(os.environ),
            )
            identity[executable] = binding.identity.sha256
        return identity

    def _validate_capture_request(
        self,
        target: CaptureTarget,
        capability_id: str,
        *,
        mode: Literal["single", "experiment"],
        experiment: ExperimentDesign | None,
        limits: RequestLimits | None,
    ) -> ValidatedCaptureRequest:
        selected_limits = limits.lowered_against(self.limits) if limits else self.limits
        capability = CAPABILITY_BY_ID.get(capability_id)
        if capability is None:
            raise RuntimeFailure(
                "UNKNOWN_CAPABILITY",
                f"Unknown capability: {capability_id}",
                details={
                    "requested_capability": capability_id,
                    "available_capabilities": sorted(CAPABILITY_BY_ID),
                    "recovery": "Run `flameox mcp inspect --summary` to select a capability.",
                },
            )
        capture_arguments = self._capture_arguments(
            target.provider_id, target.capture_arguments, capability_id=capability_id
        )
        TypeAdapter(capability.model).validate_python(target.analysis_arguments)
        if (mode == "experiment") != (experiment is not None):
            raise RuntimeFailure(
                "INVALID_INPUT", "experiment mode and design must be supplied together"
            )
        output_formats = set(self._capture_output_formats(target.provider_id))
        compatible_provider_ids = [
            contract.id for contract in compatible_capture_providers(capability)
        ]
        if target.provider_id not in compatible_provider_ids:
            raise RuntimeFailure(
                "UNSUPPORTED_FORMAT",
                (
                    f"Capture provider {target.provider_id!r} cannot feed capability "
                    f"{capability_id!r}."
                ),
                details={
                    "provider_id": target.provider_id,
                    "output_formats": sorted(output_formats),
                    "capability_id": capability_id,
                    "accepted_formats": list(capability.formats),
                    "compatible_capture_providers": compatible_provider_ids,
                },
            )
        cases = experiment.cases if experiment else [ExperimentCase(name="single")]
        source_count = (
            len(cases)
            * (experiment.blocks if experiment else 1)
            * len(CAPTURE_PROVIDER_CONTRACTS[target.provider_id].artifacts)
        )
        if source_count > MAX_INPUTS:
            raise RuntimeFailure(
                "LIMIT_EXCEEDED",
                (
                    f"Capture would produce {source_count} analysis sources; "
                    f"the limit is {MAX_INPUTS}."
                ),
            )
        capability.validate_source_count(source_count)
        return ValidatedCaptureRequest(
            capability=capability,
            capture_arguments=capture_arguments,
            limits=selected_limits,
            cases=cases,
            blocks=experiment.blocks if experiment else 1,
        )

    @staticmethod
    def _capture_output_formats(provider_id: str) -> list[str]:
        contract = CAPTURE_PROVIDER_CONTRACTS[provider_id]
        return [artifact.format for artifact in contract.artifacts]

    async def capture_and_analyze(  # noqa: C901 - capture lifecycle keeps failure artifacts together
        self,
        target: CaptureTarget,
        capability_id: str,
        *,
        mode: Literal["single", "experiment"] = "single",
        experiment: ExperimentDesign | None = None,
        limits: RequestLimits | None = None,
        progress: Any | None = None,
        preserve: bool = False,
    ) -> dict[str, Any]:
        validated_capture = self._validate_capture_request(
            target, capability_id, mode=mode, experiment=experiment, limits=limits
        )
        selected_limits = validated_capture.limits
        capture_arguments = validated_capture.capture_arguments
        cases = validated_capture.cases
        blocks = validated_capture.blocks
        sequence = self._capture_sequence(cases, blocks, experiment)
        total = len(sequence)
        has_oracle = experiment is not None and experiment.semantic_oracle is not None
        full_output = (
            target.console_output == "full" or target.provider_id == "direct" or has_oracle
        )
        full_oracle_output = target.console_output == "full"
        diagnostic_bytes = 0
        cwd = self._resolve_capture_cwd(target.cwd)
        async with self._capture_scope() as request_scratch:

            def admit() -> list[BoundCapture]:
                nonlocal diagnostic_bytes
                pending_executions: list[dict[str, Any]] = []
                bound_captures: list[BoundCapture] = []
                probed_workloads: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
                for sequence_number, block, case in sequence:
                    argv = case.argv or target.argv
                    environment = {**target.environment, **case.environment}
                    if len(environment) > 32:
                        raise RuntimeFailure(
                            "INVALID_INPUT",
                            "merged capture environment must contain at most 32 entries",
                        )
                    directory = request_scratch / f"case-{sequence_number:04d}"
                    workload_binding = self._require_host_tool(
                        argv[0], cwd=cwd, environment={**os.environ, **environment}
                    )
                    pinned_argv = [str(workload_binding.invocation_path), *argv[1:]]
                    invocation = build_capture_invocation(
                        target.provider_id,
                        pinned_argv,
                        environment,
                        capture_arguments,
                        directory,
                        self._require_managed_executable,
                    )
                    workload_key = (argv[0], tuple(sorted(environment.items())))
                    if workload_key not in probed_workloads:
                        self._require_workload_python_provider(
                            target.provider_id, argv, environment, cwd=cwd
                        )
                        probed_workloads.add(workload_key)
                    collector_binding = self._require_host_tool(
                        invocation.argv[0],
                        cwd=cwd,
                        environment={**os.environ, **invocation.environment},
                        provider_id=target.provider_id,
                    )
                    if (
                        target.provider_id == "nsight-compute"
                        and self.nsight_compute.resolve_interface(collector_binding.invocation_path)
                        is None
                    ):
                        raise RuntimeFailure(
                            "UNAVAILABLE_CAPABILITY",
                            "The official Nsight Compute ncu_report.py interface is missing.",
                            details={
                                "provider_id": target.provider_id,
                                "external_setup_guidance": SYSTEM_PROVIDER_GUIDANCE[
                                    target.provider_id
                                ],
                            },
                        )
                    self.dependencies.verify_capture_binding(target.provider_id, collector_binding)
                    if experiment is not None and experiment.semantic_oracle is not None:
                        self._require_host_tool(
                            experiment.semantic_oracle[0],
                            cwd=cwd,
                            environment={**os.environ, **environment},
                        )
                    bound_captures.append(
                        BoundCapture(
                            sequence_number,
                            block,
                            case,
                            argv,
                            environment,
                            directory,
                            invocation,
                            collector_binding,
                            workload_binding,
                        )
                    )
                    pending_executions.append(
                        self._pending_capture_execution(case, block, argv, invocation.argv, cwd)
                        | {
                            "executable_sha256": collector_binding.identity.sha256.removeprefix(
                                "sha256:"
                            ),
                            "collector_executable_sha256": (
                                collector_binding.identity.sha256.removeprefix("sha256:")
                            ),
                            "workload_executable_sha256": (
                                workload_binding.identity.sha256.removeprefix("sha256:")
                            ),
                            "returncode_scope": invocation.returncode_scope,
                            "workload_returncode": None,
                        }
                    )
                self._check_capture_provenance_capacity(
                    target=target,
                    mode=mode,
                    experiment=experiment,
                    executions=pending_executions,
                    limit=selected_limits.max_provenance_bytes,
                )
                request_bytes = len(
                    canonical_bytes(
                        self._capture_request_body(target, mode, experiment, pending_executions)
                    )
                )
                # JSON escaping can expand each retained byte to six bytes. Share
                # half the remaining budget between both streams of all executions;
                # the other half remains available for completed execution metadata.
                diagnostic_bytes = min(
                    4096,
                    (selected_limits.max_provenance_bytes - request_bytes)
                    // (24 * total * (1 + has_oracle)),
                )
                self._reserve_capture_capacity(
                    total,
                    request_scratch=request_scratch,
                    provider_id=target.provider_id,
                    full_output=full_output,
                    full_oracle_output=has_oracle and full_oracle_output,
                    limits=selected_limits,
                )
                request_scratch.mkdir()
                for item in bound_captures:
                    item.directory.mkdir()
                    materialize_capture_support(target.provider_id, item.directory)
                return bound_captures

            bound_captures = await self.run_in_request(admit)
            captured: list[NativeSource] = []
            analysis_sources: list[NativeSource] = []
            executions: list[dict[str, Any]] = []
            for item in bound_captures:
                sequence_number, block, case = item.sequence_number, item.block, item.case
                if progress:
                    await progress(sequence_number - 1, total, f"capture {case.name} block {block}")
                argv, environment, directory = item.argv, item.environment, item.directory
                invocation = item.invocation
                binding = item.collector_binding
                self._revalidate_executable(item.workload_binding)
                request = ExecutionRequest(
                    argv=invocation.argv,
                    executable_binding=binding,
                    cwd=cwd,
                    environment_allowlist=("PATH",),
                    environment_overrides=invocation.environment,
                    allowed_working_roots=(cwd,),
                    timeout_seconds=target.budget.timeout_seconds,
                    max_output_bytes=selected_limits.max_output_bytes,
                    diagnostic_bytes=None if full_output else diagnostic_bytes,
                    output_directory=request_scratch / f"console-{sequence_number:04d}"
                    if full_output
                    else None,
                    output_root=request_scratch if full_output else None,
                    resource_policy=await self.run_in_request(
                        partial(
                            self._capture_resource_policy,
                            selected_limits,
                            budget=target.budget,
                            writable_root=directory,
                        )
                    ),
                )
                failure_code: str | None = None
                try:
                    outcome = await self.broker.run(request)
                    receipt: ExecutionOutcome | ProcessExecutionError = outcome
                    process = outcome.process
                    containment = outcome.containment.value
                except ProcessExecutionError as error:
                    receipt = error
                    process = error.process
                    containment = "broker"
                    failure_code = error.code.value
                output_sources, console_metadata = await self.run_in_request(
                    partial(
                        self._capture_console_output,
                        receipt,
                        provider_id=target.provider_id,
                        role_prefix=f"capture-{sequence_number:04d}/",
                    )
                )
                captured.extend(output_sources)
                if target.provider_id == "direct":
                    analysis_sources.extend(output_sources)
                missing_artifact_roles: list[str] = []
                artifact_rejections: list[dict[str, str]] = []
                resolved_artifacts: list[NativeSource] = []
                for path, format_name, role in invocation.artifacts:
                    try:
                        native = await self.run_in_request(
                            partial(
                                self._resolve_capture_artifact,
                                path,
                                format_name=format_name,
                                role=role,
                                provider_id=target.provider_id,
                            )
                        )
                    except RuntimeFailure as error:
                        artifact_rejections.append(
                            {
                                "role": role,
                                "code": error.code,
                                "message": (
                                    "Capture artifact exceeds session storage bounds; "
                                    "the case's native output bundle was discarded."
                                    if error.code == "LIMIT_EXCEEDED"
                                    else "Capture artifact failed validation; "
                                    "the case's native output bundle was discarded."
                                ),
                            }
                        )
                        break
                    except OSError:
                        artifact_rejections.append(
                            {
                                "role": role,
                                "code": "ARTIFACT_IO_FAILURE",
                                "message": (
                                    "Capture artifact could not be inspected; "
                                    "the case's native output bundle was discarded."
                                ),
                            }
                        )
                        break
                    if native is None:
                        missing_artifact_roles.append(role)
                        continue
                    resolved_artifacts.append(native)
                if artifact_rejections:
                    await self.run_in_request(
                        partial(
                            self._discard_rejected_capture_directory,
                            directory,
                            request_scratch,
                        )
                    )
                else:
                    for native in resolved_artifacts:
                        captured_native = NativeSource(
                            native.path,
                            native.sha256,
                            native.size_bytes,
                            native.format,
                            native.producer,
                            f"capture-{sequence_number:04d}/{native.role}",
                        )
                        captured.append(captured_native)
                        analysis_sources.append(captured_native)
                termination = process.termination
                exit_code = getattr(termination, "exit_code", None)
                status = "succeeded" if exit_code == 0 and failure_code is None else "failed"
                if status == "succeeded" and missing_artifact_roles:
                    status = "failed"
                    failure_code = "CAPTURE_ARTIFACT_MISSING"
                if artifact_rejections:
                    status = "failed"
                    failure_code = failure_code or "CAPTURE_ARTIFACT_REJECTED"
                oracle: dict[str, Any] | None = None
                if (
                    status == "succeeded"
                    and experiment is not None
                    and experiment.semantic_oracle is not None
                ):
                    oracle_argv = experiment.semantic_oracle
                    oracle_environment = {
                        **environment,
                        SEMANTIC_ORACLE_STDOUT_ENV: str(output_sources[0].path),
                        SEMANTIC_ORACLE_STDERR_ENV: str(output_sources[1].path),
                    }
                    oracle_binding = self._require_host_tool(
                        oracle_argv[0],
                        cwd=cwd,
                        environment={**os.environ, **oracle_environment},
                    )
                    oracle_request = ExecutionRequest(
                        argv=tuple(oracle_argv),
                        executable_binding=oracle_binding,
                        cwd=cwd,
                        environment_allowlist=("PATH",),
                        environment_overrides=oracle_environment,
                        allowed_working_roots=(cwd,),
                        timeout_seconds=target.budget.timeout_seconds,
                        max_output_bytes=selected_limits.max_output_bytes,
                        diagnostic_bytes=None if full_oracle_output else diagnostic_bytes,
                        output_directory=request_scratch / f"oracle-console-{sequence_number:04d}"
                        if full_oracle_output
                        else None,
                        output_root=request_scratch if full_oracle_output else None,
                        resource_policy=await self.run_in_request(
                            partial(
                                self._capture_resource_policy,
                                selected_limits,
                                budget=target.budget,
                                writable_root=directory,
                            )
                        ),
                    )
                    try:
                        oracle_outcome = await self.broker.run(oracle_request)
                        oracle_receipt: ExecutionOutcome | ProcessExecutionError = oracle_outcome
                        oracle_process = oracle_outcome.process
                        oracle_failure_code: str | None = None
                    except ProcessExecutionError as error:
                        oracle_receipt = error
                        oracle_process = error.process
                        oracle_failure_code = error.code.value
                    oracle_exit_code = getattr(oracle_process.termination, "exit_code", None)
                    oracle_sources, oracle_metadata = await self.run_in_request(
                        partial(
                            self._capture_console_output,
                            oracle_receipt,
                            provider_id=target.provider_id,
                            role_prefix=f"capture-{sequence_number:04d}/oracle_",
                        )
                    )
                    captured.extend(oracle_sources)
                    oracle = {
                        "argv": oracle_argv,
                        "returncode": oracle_exit_code,
                        "status": "passed"
                        if oracle_exit_code == 0 and oracle_failure_code is None
                        else "failed",
                        "failure_code": oracle_failure_code,
                        "limit": self._terminated_limit(
                            oracle_process, selected_limits, target.budget
                        ),
                        **oracle_metadata,
                    }
                await self.run_in_request(
                    partial(self._prune_scratch, protected_root=request_scratch)
                )
                if oracle is not None and oracle["status"] == "failed":
                    status = "failed"
                    failure_code = "SEMANTIC_ORACLE_FAILED"
                executions.append(
                    {
                        "case": case.name,
                        "block": block,
                        "argv": argv,
                        "capture_argv": list(invocation.argv),
                        "cwd": str(cwd),
                        "returncode": exit_code,
                        "executable_sha256": binding.identity.sha256.removeprefix("sha256:"),
                        "collector_executable_sha256": (
                            binding.identity.sha256.removeprefix("sha256:")
                        ),
                        "workload_executable_sha256": (
                            item.workload_binding.identity.sha256.removeprefix("sha256:")
                        ),
                        "returncode_scope": invocation.returncode_scope,
                        "workload_returncode": exit_code
                        if invocation.returncode_scope == "workload"
                        else None,
                        "status": status,
                        "failure_code": failure_code,
                        **console_metadata,
                        "missing_artifact_roles": missing_artifact_roles,
                        "artifact_rejections": artifact_rejections,
                        "semantic_oracle": oracle,
                        "wall_time_ns": process.wall_time_ns,
                        "containment": containment,
                        "limit": self._terminated_limit(process, selected_limits, target.budget),
                    }
                )
                self._check_capture_provenance_capacity(
                    target=target,
                    mode=mode,
                    experiment=experiment,
                    executions=executions,
                    limit=selected_limits.max_provenance_bytes,
                )
                if progress:
                    await progress(sequence_number, total, f"captured {case.name} block {block}")

            def finish_analysis() -> dict[str, Any]:
                effective_capability_id, selected_sources = self._capture_analysis_sources(
                    capability_id, analysis_sources, captured
                )
                try:
                    if not selected_sources:
                        raise RuntimeFailure(
                            "EXECUTION_FAILURE",
                            "Capture produced no native artifacts for the requested analysis.",
                            details={"provider_id": target.provider_id},
                        )
                    result = self.analyze(
                        effective_capability_id,
                        [
                            PathSource(
                                path=str(item.path), format=item.format, producer=item.producer
                            )
                            for item in selected_sources
                        ],
                        target.analysis_arguments,
                        limits=selected_limits,
                    )
                except RuntimeFailure as error:
                    result = self._capture_failure_result(
                        target=target,
                        capability_id=capability_id,
                        mode=mode,
                        experiment=experiment,
                        executions=executions,
                        captured=captured,
                        analysis_sources=selected_sources,
                        limits=selected_limits,
                        failure=error,
                    )
                    if preserve:
                        result["preserved"] = self.preserve_evidence(str(result["analysis_id"]))
                    return result
                cached = self.analyses[str(result["analysis_id"])]
                if target.provider_id == "py-spy":
                    py_spy_arguments = cast(PySpyCaptureArguments, capture_arguments)
                    process_scope = (
                        "the target and newly created Python subprocesses"
                        if py_spy_arguments.subprocesses
                        else "the target process only"
                    )
                    result["limitations"].append(
                        f"py-spy sampled {process_scope}; sampled stacks are not complete "
                        "process-tree execution evidence."
                    )
                cached.sources = captured
                cached.manifest_body["capture_request"] = {
                    "target": target.model_dump(mode="json"),
                    "mode": mode,
                    "experiment": experiment.model_dump(mode="json") if experiment else None,
                    "executions": executions,
                }
                result = self._finalize_capture_result(
                    result,
                    cached,
                    capability_id=capability_id,
                    mode=mode,
                    experiment=experiment,
                    executions=executions,
                    max_result_bytes=selected_limits.max_result_bytes,
                )
                if preserve:
                    result["preserved"] = self.preserve_evidence(str(result["analysis_id"]))
                self._prune_scratch(protected_root=request_scratch)
                return result

            return await self.run_in_request(finish_analysis)

    @staticmethod
    def _capture_console_output(
        receipt: ExecutionOutcome | ProcessExecutionError, *, provider_id: str, role_prefix: str
    ) -> tuple[list[NativeSource], dict[str, Any]]:
        diagnostics = receipt.diagnostic_output
        if diagnostics is not None:
            metadata = ConsoleDiagnostics.model_validate(
                diagnostics.as_details()
                | {
                    "stdout": (receipt.stdout or b"").decode("utf-8", errors="replace"),
                    "stderr": (receipt.stderr or b"").decode("utf-8", errors="replace"),
                }
            )
            return [], {"console_diagnostics": metadata.model_dump(mode="json")}
        sources, streams = AnalysisRuntime._capture_output_sources(
            receipt.output_sink, provider_id=provider_id, role_prefix=role_prefix
        )
        return sources, {"output_streams": streams.model_dump(mode="json")}

    @staticmethod
    def _capture_output_sources(
        sink: OutputSink | None, *, provider_id: str, role_prefix: str
    ) -> tuple[list[NativeSource], OutputStreams]:
        if sink is None:
            raise RuntimeFailure("EXECUTION_FAILURE", "Capture did not return disk-backed output.")
        sources = []
        for role, path, expected_size in (
            ("stdout", sink.stdout_path, sink.stdout_bytes),
            ("stderr", sink.stderr_path, sink.stderr_bytes),
        ):
            digest, size = sha256_file(path)
            if size != expected_size:
                raise RuntimeFailure(
                    "MISSING_OR_CHANGED_INPUT", "Captured output changed after broker settlement."
                )
            sources.append(
                NativeSource(path, digest, size, "text", provider_id, role_prefix + role)
            )
        return sources, OutputStreams(
            stdout_bytes=sink.stdout_bytes,
            stderr_bytes=sink.stderr_bytes,
            stdout_complete=sink.stdout_complete,
            stderr_complete=sink.stderr_complete,
            io_error=sink.io_error,
        )

    def _capture_failure_result(
        self,
        *,
        target: CaptureTarget,
        capability_id: str,
        mode: str,
        experiment: ExperimentDesign | None,
        executions: list[dict[str, Any]],
        captured: list[NativeSource],
        analysis_sources: list[NativeSource],
        limits: RequestLimits,
        failure: RuntimeFailure,
    ) -> dict[str, Any]:
        capture_request = self._capture_request_body(target, mode, experiment, executions)
        failure_body = {
            "code": failure.code,
            "message": failure.message,
            "details": failure.details,
        }
        analysis_request = {
            "capability_id": capability_id,
            "inputs": [
                {
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "format": item.format,
                    "producer": item.producer,
                    "role": item.role,
                }
                for item in analysis_sources
            ],
            "arguments": target.analysis_arguments,
            "limits": limits.model_dump(mode="json"),
            "failure": failure_body,
        }
        analysis_id = hashlib.sha256(
            self.session_id.encode()
            + canonical_bytes(capture_request)
            + canonical_bytes(analysis_request)
            + canonical_bytes([item.sha256 for item in captured])
        ).hexdigest()
        result: dict[str, Any] = {
            "analysis_id": analysis_id,
            "capability_id": capability_id,
            "provider": {"id": "flameox-capture", "version": __version__},
            "inputs": [item.public() for item in captured],
            "blocks": [
                {
                    "type": "metrics",
                    "values": {
                        "captured_artifact_count": len(captured),
                        "analysis_succeeded": False,
                    },
                },
                {"type": "table", "rows": []},
            ],
            "coverage": {"rows_returned": 0, "rows_observed": 0, "complete": False},
            "truncation": None,
            "limitations": [
                "Requested analysis failed; inspect the capture outcome and retained diagnostics. "
                "No analysis claims are available."
            ],
            "continuation": None,
            "capture": {
                "mode": mode,
                "requested_capability_id": capability_id,
                "executions": executions,
                "outcome": self._capture_outcome(executions),
            },
            "analysis_failure": failure_body,
        }
        self._bound_capture_result(result, limits.max_result_bytes, analysis_request, 0)
        validated = AnalysisResult.model_validate(result).model_dump(
            mode="json", exclude_none=False
        )
        manifest_body = {
            "evidence_kind": "capture",
            "capability_id": capability_id,
            "provider": validated["provider"],
            "inputs": [
                {
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "format": item.format,
                    "role": item.role,
                }
                for item in captured
            ],
            "capture_request": capture_request,
            "analysis_request": analysis_request,
            "episode": {"created_at": datetime.now(UTC).isoformat()},
            "coverage": validated["coverage"],
            "limitations": validated["limitations"],
        }
        self._cache_analysis(analysis_id, CachedAnalysis(validated, captured, manifest_body))
        return self._copy_result(validated)

    def _finalize_capture_result(
        self,
        result: dict[str, Any],
        cached: CachedAnalysis,
        *,
        capability_id: str,
        mode: str,
        experiment: ExperimentDesign | None,
        executions: list[dict[str, Any]],
        max_result_bytes: int,
    ) -> dict[str, Any]:
        result["capture"] = {
            "mode": mode,
            "requested_capability_id": capability_id,
            "executions": executions,
            "outcome": self._capture_outcome(executions),
        }
        if experiment is not None:
            experiment_blocks, experiment_limitations = self._experiment_blocks(
                experiment, executions
            )
            result["blocks"].extend(experiment_blocks)
            result["limitations"].extend(experiment_limitations)
        analysis_request = cached.manifest_body["analysis_request"]
        continuation_identity = {
            key: value for key, value in analysis_request.items() if key != "offset"
        }
        self._bound_capture_result(
            result,
            max_result_bytes,
            continuation_identity,
            int(analysis_request["offset"]),
        )
        validated_result = AnalysisResult.model_validate(result).model_dump(
            mode="json", exclude_none=False
        )
        cached.result = self._copy_result(validated_result)
        cached.manifest_body["coverage"] = self._copy_result(validated_result["coverage"])
        cached.manifest_body["limitations"] = list(validated_result["limitations"])
        return self._copy_result(validated_result)

    @staticmethod
    def _capture_request_body(
        target: CaptureTarget,
        mode: str,
        experiment: ExperimentDesign | None,
        executions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "target": target.model_dump(mode="json"),
            "mode": mode,
            "experiment": experiment.model_dump(mode="json") if experiment else None,
            "executions": list(executions),
        }

    @staticmethod
    def _capture_sequence(
        cases: Sequence[ExperimentCase],
        blocks: int,
        experiment: ExperimentDesign | None,
    ) -> list[tuple[int, int, ExperimentCase]]:
        sequence: list[tuple[int, int, ExperimentCase]] = []
        for block in range(1, blocks + 1):
            block_cases = list(cases)
            if experiment is not None:
                random.Random(experiment.seed + block - 1).shuffle(block_cases)
            for case in block_cases:
                sequence.append((len(sequence) + 1, block, case))
        return sequence

    @staticmethod
    def _pending_capture_execution(
        case: ExperimentCase,
        block: int,
        argv: Sequence[str],
        capture_argv: Sequence[str],
        cwd: Path,
    ) -> dict[str, Any]:
        return {
            "case": case.name,
            "block": block,
            "argv": list(argv),
            "capture_argv": list(capture_argv),
            "cwd": str(cwd),
            "returncode": None,
            "status": "pending",
            "failure_code": None,
            "missing_artifact_roles": [],
            "semantic_oracle": None,
            "wall_time_ns": None,
            "containment": "broker",
            "limit": None,
        }

    def _check_capture_provenance_capacity(
        self,
        *,
        target: CaptureTarget,
        mode: str,
        experiment: ExperimentDesign | None,
        executions: Sequence[Mapping[str, Any]],
        limit: int,
    ) -> None:
        if (
            len(canonical_bytes(self._capture_request_body(target, mode, experiment, executions)))
            > limit
        ):
            raise RuntimeFailure(
                "LIMIT_EXCEEDED",
                "Capture provenance exceeds max_provenance_bytes",
            )

    @staticmethod
    def _require_host_tool(
        executable: str,
        *,
        cwd: Path,
        environment: Mapping[str, str],
        provider_id: str | None = None,
    ) -> ResolvedExecutable:
        try:
            return ExecutableResolver().require_host_tool(
                executable, cwd=cwd, environment=dict(environment)
            )
        except DomainError as error:
            code = (
                "UNAVAILABLE_CAPABILITY"
                if error.code is ErrorCode.UNAVAILABLE_CAPABILITY
                else "EXECUTION_FAILURE"
            )
            details = {}
            if provider_id in SYSTEM_PROVIDER_GUIDANCE:
                details = {
                    "provider_id": provider_id,
                    "external_setup_guidance": SYSTEM_PROVIDER_GUIDANCE[provider_id],
                }
            raise RuntimeFailure(code, error.message, details=details) from error

    @staticmethod
    def _revalidate_executable(binding: ResolvedExecutable) -> None:
        try:
            ExecutableResolver().revalidate(binding)
        except DomainError as error:
            raise RuntimeFailure(error.code.value, error.message, details=error.details) from error

    @staticmethod
    def _managed_executable(name: str) -> str | None:
        candidate = Path(sys.executable).with_name(name)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None

    def _require_managed_executable(self, provider_id: str, name: str) -> str:
        executable = (
            self.dependencies.py_spy_executable() if provider_id == "py-spy" else None
        ) or self._managed_executable(name)
        if executable is None:
            raise RuntimeFailure(
                "UNAVAILABLE_CAPABILITY",
                (
                    f"Managed provider executable is unavailable: {name}. Call "
                    f"prepare_providers with provider_ids=[{provider_id!r}], then follow its "
                    "activation guidance and retry."
                ),
                details={
                    "provider_id": provider_id,
                    "preparation_tool": "prepare_providers",
                    "provider_ids": [provider_id],
                },
            )
        return executable

    def _require_workload_python_provider(
        self,
        provider_id: str,
        target_argv: list[str],
        environment: dict[str, str],
        *,
        cwd: Path,
    ) -> None:
        requirement = WORKLOAD_PYTHON_REQUIREMENTS.get(provider_id)
        if requirement is None:
            return
        module, distribution, supported_versions = requirement
        if len(target_argv) < 2:
            return
        try:
            binding = self._require_host_tool(
                target_argv[0],
                cwd=cwd,
                environment={**os.environ, **environment},
            )
            probe = self.broker.run_sync(
                ExecutionRequest(
                    argv=(
                        target_argv[0],
                        "-I",
                        "-c",
                        (
                            "import importlib.metadata as m,importlib.util as u;"
                            f"assert u.find_spec({module!r}) is not None;"
                            f"print(m.version({distribution!r}))"
                        ),
                    ),
                    executable_binding=binding,
                    cwd=cwd,
                    environment_allowlist=("PATH",),
                    environment_overrides=environment,
                    allowed_working_roots=(cwd,),
                    timeout_seconds=10,
                    max_output_bytes=4_096,
                )
            )
            version = probe.stdout.decode("utf-8", errors="replace").strip()
            available = (
                process_exit_code(probe.process.termination) == 0
                and bool(version)
                and Version(version) in SpecifierSet(supported_versions)
            )
        except (DomainError, InvalidVersion, OSError, UnicodeError):
            available = False
        if available:
            return
        raise RuntimeFailure(
            "UNAVAILABLE_CAPABILITY",
            (
                f"{distribution} {supported_versions} is not available in the workload "
                f"interpreter {target_argv[0]!r}. Install it into that interpreter without "
                "changing the declared workload environment, then retry."
            ),
        )

    def _resolve_capture_artifact(
        self,
        path: Path,
        *,
        format_name: str,
        role: str,
        provider_id: str,
    ) -> NativeSource | None:
        if path.is_symlink():
            raise RuntimeFailure("INVALID_INPUT", "Capture artifact must not be a symlink")
        if not path.exists():
            return None
        if not path.is_file() and not path.is_dir():
            raise RuntimeFailure("INVALID_INPUT", "Capture artifact must be a regular file")
        digest, size, file_count = hash_path(
            path,
            max_bytes=MAX_SESSION_SCRATCH_BYTES,
            max_files=MAX_SESSION_SCRATCH_FILES,
        )
        if file_count == 0:
            return None
        return NativeSource(path, digest, size, format_name, provider_id, role)

    @staticmethod
    def _discard_rejected_capture_directory(directory: Path, request_scratch: Path) -> None:
        """Remove only a rejected, runtime-owned capture directory."""
        if directory.parent != request_scratch:
            return
        if directory.is_symlink():
            directory.unlink()
            return
        if not directory.is_dir():
            return
        shutil.rmtree(directory)

    @staticmethod
    def _capture_analysis_sources(
        capability_id: str,
        analysis_sources: list[NativeSource],
        captured: list[NativeSource],
    ) -> tuple[str, list[NativeSource]]:
        if analysis_sources:
            return capability_id, analysis_sources
        return "artifact.preview", [
            source for source in captured if source.role.endswith(("/stdout", "/stderr"))
        ]

    @staticmethod
    def _capture_arguments(
        provider_id: str, arguments: Mapping[str, Any], *, capability_id: str
    ) -> CaptureArguments:
        contract = CAPTURE_PROVIDER_CONTRACTS.get(provider_id)
        if contract is None:
            capture_tool = (
                "capture_process_output"
                if capability_id == "artifact.preview"
                else f"capture_{capability_id.replace('.', '_')}"
            )
            raise RuntimeFailure(
                "UNKNOWN_CAPABILITY",
                f"Unknown capture provider: {provider_id}",
                details={
                    "requested_provider": provider_id,
                    "available_capture_providers": sorted(CAPTURE_PROVIDER_CONTRACTS),
                    "recovery": (
                        "Inspect the compatible provider schema with "
                        f"`flameox mcp inspect --tool {capture_tool}`."
                    ),
                },
            )
        return cast(CaptureArguments, contract.argument_model.model_validate(arguments))

    def _reserve_capture_capacity(
        self,
        total: int,
        *,
        request_scratch: Path,
        provider_id: str,
        full_output: bool,
        full_oracle_output: bool,
        limits: RequestLimits,
    ) -> None:
        provider_files = 2 if provider_id == "coverage" else int(provider_id != "direct")
        files_per_capture = 2 * int(full_output) + provider_files + 2 * int(full_oracle_output)
        bounded_outputs = int(full_output) + int(provider_id != "direct") + int(full_oracle_output)
        reserved_bytes = total * bounded_outputs * limits.max_output_bytes
        reserved_files = total * files_per_capture
        self._prune_scratch(reserved_bytes=reserved_bytes, reserved_files=reserved_files)
        self._capture_reservations[request_scratch] = (reserved_bytes, reserved_files)

    def preserve_evidence(self, analysis_id: str) -> dict[str, Any]:
        cached = self.analyses.get(analysis_id)
        if cached is None:
            raise RuntimeFailure(
                "EXPIRED_SESSION_ANALYSIS", "The session analysis is missing or expired"
            )
        if cached.preserved is not None:
            try:
                self.repository.read(str(cached.preserved["evidence_id"]))
            except RepositoryError as exc:
                if exc.code != "MISSING_EVIDENCE":
                    raise self._repository_failure(exc) from exc
                cached.preserved = None
            except OSError as exc:
                raise RuntimeFailure(
                    "REPOSITORY_IO_FAILURE", "Preserved evidence could not be validated."
                ) from exc
            else:
                self.analyses.move_to_end(analysis_id)
                return dict(cached.preserved)
        if cached.preserved is None:
            try:
                cached.preserved = self.repository.preserve(
                    manifest_body=cached.manifest_body,
                    sources=cached.sources,
                    analysis=self._durable_analysis(cached.result),
                )
                self._release_analysis_scratch(cached)
            except RepositoryError as exc:
                raise self._repository_failure(exc) from exc
            except OSError as exc:
                raise RuntimeFailure(
                    "REPOSITORY_IO_FAILURE", "Evidence could not be preserved."
                ) from exc
        self.analyses.move_to_end(analysis_id)
        return dict(cached.preserved)

    def preflight_rescue_destination(self, destination: str) -> str:
        """Validate a new rescue store before an expensive request begins."""
        selected = self._rescue_destination(destination)
        parent_descriptor = self._open_rescue_parent(selected.parent)
        try:
            try:
                os.stat(selected.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return str(selected)
            raise RuntimeFailure(
                "INVALID_INPUT", "Rescue destination must be a new path that does not exist"
            )
        except OSError as exc:
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "Rescue destination could not be validated."
            ) from exc
        finally:
            os.close(parent_descriptor)

    def rescue_evidence(self, analysis_id: str, destination: str) -> dict[str, Any]:
        selected = self._rescue_destination(destination)
        rescue_key = (analysis_id, str(selected))
        rescues = getattr(self, "rescues", None)
        if rescues is None:
            rescues = self.rescues = OrderedDict()
        previous = rescues.get(rescue_key)
        cached = self.analyses.get(analysis_id)
        if cached is None and previous is None:
            raise RuntimeFailure(
                "EXPIRED_SESSION_ANALYSIS", "The session analysis is missing or expired"
            )
        parent_descriptor = self._open_rescue_parent(selected.parent)
        try:
            result = self._rescue_to_open_parent(
                parent_descriptor,
                selected=selected,
                cached=cached,
                previous=previous,
            )
        except RepositoryError as exc:
            raise self._repository_failure(
                exc,
                repository=EvidenceRepository(selected, f"{self.session_id}-rescue"),
                configuration_source="rescue_destination",
            ) from exc
        except OSError as exc:
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "Session evidence could not be rescued."
            ) from exc
        finally:
            os.close(parent_descriptor)
        if cached is not None:
            self.analyses.move_to_end(analysis_id)
        rescues[rescue_key] = result
        rescues.move_to_end(rescue_key)
        while len(rescues) > MAX_SESSION_RESCUES:
            rescues.popitem(last=False)
        return self._copy_result(result)

    def _rescue_destination(self, destination: str) -> Path:
        supplied = Path(destination).expanduser()
        if not supplied.is_absolute():
            raise RuntimeFailure("INVALID_INPUT", "Rescue destination must be an absolute path")
        selected = Path(os.path.abspath(supplied))
        configured = self.repository.root
        configured_physical = configured.resolve(strict=False)
        selected_physical = selected.resolve(strict=False)
        if self._paths_overlap(selected, configured) or self._paths_overlap(
            selected_physical, configured_physical
        ):
            raise RuntimeFailure(
                "INVALID_INPUT", "Rescue destination must be outside the configured repository"
            )
        return selected

    @staticmethod
    def _paths_overlap(first: Path, second: Path) -> bool:
        return first == second or first.is_relative_to(second) or second.is_relative_to(first)

    def _rescue_to_open_parent(
        self,
        parent_descriptor: int,
        *,
        selected: Path,
        cached: CachedAnalysis | None,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        anchored_parent = self._descriptor_path(parent_descriptor)
        anchored_destination = anchored_parent / selected.name
        alternate = EvidenceRepository(anchored_destination, f"{self.session_id}-rescue")
        durable = self._durable_analysis(cached.result) if cached is not None else None
        if cached is not None and durable is not None:
            expected_id = alternate.expected_evidence_id(
                manifest_body=cached.manifest_body,
                sources=cached.sources,
                analysis=durable,
            )
        elif previous is not None:
            expected_id = str(previous["evidence_id"])
        else:  # Guarded by rescue_evidence.
            raise RuntimeFailure(
                "EXPIRED_SESSION_ANALYSIS", "The session analysis is missing or expired"
            )
        try:
            destination_status = os.stat(
                selected.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            destination_status = None
        if destination_status is not None:
            try:
                manifest = alternate.read(expected_id)
            except RepositoryError as exc:
                if previous is None and not (anchored_destination / "repository.json").exists():
                    raise RuntimeFailure(
                        "INVALID_INPUT",
                        "Rescue destination must be a new path that does not exist",
                    ) from exc
                raise
            result = self._rescue_result(expected_id, len(manifest["body"]["artifacts"]), selected)
        else:
            result = self._publish_rescue_stage(
                parent_descriptor,
                anchored_parent=anchored_parent,
                anchored_destination=anchored_destination,
                selected=selected,
                cached=cached,
                durable=durable,
            )
        anchored_status = os.fstat(parent_descriptor)
        try:
            final_parent_descriptor = self._open_rescue_parent(selected.parent)
        except RuntimeFailure as exc:
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "Rescue destination changed during publication."
            ) from exc
        try:
            final_status = os.fstat(final_parent_descriptor)
        finally:
            os.close(final_parent_descriptor)
        if (anchored_status.st_dev, anchored_status.st_ino) != (
            final_status.st_dev,
            final_status.st_ino,
        ):
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "Rescue destination changed during publication."
            )
        return result

    def _publish_rescue_stage(
        self,
        parent_descriptor: int,
        *,
        anchored_parent: Path,
        anchored_destination: Path,
        selected: Path,
        cached: CachedAnalysis | None,
        durable: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if cached is None or durable is None:
            raise RuntimeFailure(
                "EXPIRED_SESSION_ANALYSIS",
                "The session analysis is missing and the rescued evidence is unavailable",
            )
        stage_name = f".flameox-rescue-{secrets.token_hex(12)}"
        os.mkdir(stage_name, mode=0o700, dir_fd=parent_descriptor)
        try:
            rescued = EvidenceRepository(
                anchored_parent / stage_name, f"{self.session_id}-rescue"
            ).preserve(
                manifest_body=cached.manifest_body,
                sources=cached.sources,
                analysis=durable,
            )
            os.rename(
                stage_name,
                selected.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            stage_name = ""
            os.fsync(parent_descriptor)
            EvidenceRepository(anchored_destination, f"{self.session_id}-rescue").read(
                str(rescued["evidence_id"])
            )
            return self._rescue_result(
                str(rescued["evidence_id"]), int(rescued["artifact_count"]), selected
            )
        finally:
            if stage_name:
                shutil.rmtree(anchored_parent / stage_name, ignore_errors=True)

    @staticmethod
    def _rescue_result(evidence_id: str, artifact_count: int, destination: Path) -> dict[str, Any]:
        return {
            "evidence_id": evidence_id,
            "uri": f"flameox://evidence/{evidence_id}",
            "artifact_count": artifact_count,
            "rescue_destination": str(destination),
            "next_action": {
                "kind": "restart_reconnect",
                "environment": {"FLAMEOX_DATA_DIR": str(destination)},
                "message": (
                    "Restart or reconnect Flameox with FLAMEOX_DATA_DIR set to the rescue "
                    "destination, then read the returned evidence_id."
                ),
            },
        }

    @staticmethod
    def _open_rescue_parent(path: Path) -> int:
        if os.name == "nt":
            raise RuntimeFailure(
                "UNAVAILABLE_CAPABILITY",
                "Secure rescue publication is unavailable on Windows hosts.",
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.anchor, flags)
        try:
            for part in path.parts[1:]:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except OSError as exc:
            os.close(descriptor)
            raise RuntimeFailure(
                "INVALID_INPUT",
                "Rescue destination parent must exist and contain no symbolic links",
            ) from exc
        return descriptor

    @staticmethod
    def _descriptor_path(descriptor: int) -> Path:
        for root in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = root / str(descriptor)
            try:
                if os.path.samestat(candidate.stat(), os.fstat(descriptor)):
                    return candidate
            except OSError:
                continue
        raise RuntimeFailure(
            "UNAVAILABLE_CAPABILITY",
            "Secure rescue publication requires descriptor-backed directory paths",
        )

    @staticmethod
    def _terminated_limit(
        process: ProcessResult, limits: RequestLimits, budget: WorkloadBudget
    ) -> dict[str, Any] | None:
        cause = process.cancellation_cause
        if cause is None:
            return None
        resources = process.resources
        configured: int | float | None = None
        observed: int | float | None = None
        unit: str | None = None
        recovery: str | None = None
        if cause.value == "timeout":
            configured = budget.timeout_seconds
            observed = process.wall_time_ns / 1_000_000_000 if process.wall_time_ns else None
            unit = "seconds"
            recovery = "Adjust target.budget.timeout_seconds if the workload duration is expected."
        elif cause.value == "output_limit":
            configured = limits.max_output_bytes
            unit = "bytes"
            recovery = "Retry with a larger max_output_bytes or reduce target output."
        elif cause.value == "memory_limit_exceeded":
            configured = budget.max_memory_bytes
            observed = process.peak_rss_bytes
            unit = "bytes"
            recovery = "Adjust target.budget.max_memory_bytes only if the workload is trusted."
        elif cause.value == "writable_limit_exceeded":
            configured = limits.max_output_bytes
            if resources is not None and resources.writable_root_growth_bytes:
                observed = sum(resources.writable_root_growth_bytes.values())
            unit = "bytes"
            recovery = "Retry with a larger max_output_bytes or reduce capture artifacts."
        elif cause.value == "storage_reserve_exceeded":
            configured = resources.minimum_free_bytes if resources is not None else None
            unit = "bytes_free"
            recovery = "Free storage before retrying; do not lower the reserve blindly."
        else:
            return None
        return {
            "kind": cause.value,
            "configured": configured,
            "observed": observed,
            "unit": unit,
            "observation_available": observed is not None,
            "recovery": recovery,
        }

    @staticmethod
    def _durable_analysis(result: Mapping[str, Any]) -> dict[str, Any]:
        """Remove process-local handles from immutable evidence data."""

        durable = dict(result)
        durable.pop("analysis_id", None)
        durable.pop("continuation", None)
        durable["inputs"] = [
            {key: value for key, value in item.items() if key != "path"}
            for item in result["inputs"]
        ]
        return durable

    @staticmethod
    def _copy_result(result: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(result)))

    def query_evidence(
        self,
        *,
        evidence_kind: str | None = None,
        capability_id: str | None = None,
        provider_id: str | None = None,
        input_sha256: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        try:
            return self.repository.query(
                evidence_kind=evidence_kind,
                capability_id=capability_id,
                provider_id=provider_id,
                input_sha256=input_sha256,
                created_after=created_after,
                created_before=created_before,
                limit=limit,
                cursor=cursor,
            )
        except RepositoryError as exc:
            raise self._repository_failure(exc) from exc
        except OSError as exc:
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "The evidence repository could not be queried."
            ) from exc

    def read_evidence(self, evidence_id: str) -> dict[str, Any]:
        try:
            return self.repository.read(evidence_id)
        except RepositoryError as exc:
            raise self._repository_failure(exc) from exc
        except OSError as exc:
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "The requested evidence could not be read."
            ) from exc

    def read_evidence_agent_projection(self, evidence_id: str) -> dict[str, Any]:
        try:
            return self.repository.read_agent_projection(evidence_id)
        except RepositoryError as exc:
            raise self._repository_failure(exc) from exc
        except OSError as exc:
            raise RuntimeFailure(
                "REPOSITORY_IO_FAILURE", "The requested evidence projection could not be read."
            ) from exc

    def _repository_failure(
        self,
        error: RepositoryError,
        *,
        repository: EvidenceRepository | None = None,
        configuration_source: str | None = None,
    ) -> RuntimeFailure:
        selected_repository = repository or self.repository
        details: dict[str, Any] = {}
        if error.code in {"REPOSITORY_CORRUPTION", "UNSUPPORTED_REPOSITORY_FORMAT"}:
            details = {
                "configuration_source": configuration_source or self._repository_configuration,
                "configuration_variable": "FLAMEOX_DATA_DIR",
                "store_identifier": hashlib.sha256(
                    str(selected_repository.root).encode()
                ).hexdigest(),
                "recovery": [
                    (
                        "Inspect or export this store with a Flameox release that supports its "
                        "format. Do not edit version fields or rewrite existing evidence."
                        if error.code == "UNSUPPORTED_REPOSITORY_FORMAT"
                        else "Restore the original repository from a known-good backup; do not "
                        "delete existing data or synthesize repository.json."
                    ),
                    "Alternatively set FLAMEOX_DATA_DIR to a distinct new directory and "
                    "restart or reconnect Flameox. Switching stores does not recover old evidence; "
                    "preserve any recoverable session evidence before ending the session.",
                ],
                **(
                    {"local_diagnostic": "flameox evidence location"}
                    if configuration_source != "rescue_destination"
                    else {}
                ),
            }
        return RuntimeFailure(error.code, error.message, details=details)

    def _resolve_sources(
        self, sources: Sequence[Source], limits: RequestLimits
    ) -> list[NativeSource]:
        if not 1 <= len(sources) <= MAX_INPUTS:
            raise RuntimeFailure("INVALID_INPUT", "sources must contain 1 to 32 entries")
        admitted: list[NativeSource | EvidenceSelection] = []
        total_size = total_files = 0
        for source_index, source in enumerate(sources):
            if isinstance(source, EvidenceSource):
                try:
                    selection = self.repository.select_source(
                        source.evidence_id,
                        selector=source.artifact_selector,
                        role=source.artifact_role,
                    )
                except RepositoryError as error:
                    failure = self._repository_failure(error)
                    failure.details["resource_uri"] = f"flameox://evidence/{source.evidence_id}"
                    raise failure from error
                total_size += selection.source.size_bytes
                total_files += len(selection.members)
                if total_size > limits.max_input_bytes or total_files > limits.max_input_files:
                    raise RuntimeFailure(
                        "LIMIT_EXCEEDED", "Evidence exceeds input byte or file limits"
                    )
                admitted.append(selection)
            else:
                path = Path(source.path)
                if not path.is_absolute():
                    raise RuntimeFailure(
                        "INVALID_INPUT", f"Source path must be absolute: {source.path}"
                    )
                try:
                    path = path.resolve(strict=True)
                except OSError as exc:
                    raise RuntimeFailure(
                        "MISSING_OR_CHANGED_INPUT", f"Source is missing: {source.path}"
                    ) from exc
                digest, size, file_count = hash_path(
                    path,
                    max_bytes=limits.max_input_bytes - total_size,
                    max_files=limits.max_input_files - total_files,
                )
                total_size += size
                total_files += file_count
                if source.expected_sha256 and digest != source.expected_sha256:
                    raise RuntimeFailure(
                        "MISSING_OR_CHANGED_INPUT", f"SHA-256 mismatch: {source.path}"
                    )
                admitted.append(
                    NativeSource(
                        path,
                        digest,
                        size,
                        source.format or self._sniff(path),
                        source.producer,
                        "input" if len(sources) == 1 else f"input-{source_index + 1:04d}",
                    )
                )
        # Admit the entire request before allocating any bundle. Pin every source so
        # acquiring a later input or conversion cannot evict an earlier input.
        needed: dict[Path, EvidenceSelection] = {}
        for item in admitted:
            if isinstance(item, EvidenceSelection):
                if item.source.is_directory:
                    path = self._evidence_destination(item)
                    self._protected_sources.add(path)
                    if not path.exists():
                        needed[path] = item
            elif item.path.is_relative_to(self.scratch):
                self._protected_sources.add(item.path)
        self._prune_scratch(
            reserved_bytes=sum(item.source.size_bytes for item in needed.values()),
            reserved_files=sum(len(item.members) for item in needed.values()),
        )
        result = []
        for item in admitted:
            if isinstance(item, EvidenceSelection):
                try:
                    self.repository.verify_source(item)
                except RepositoryError as error:
                    raise self._repository_failure(error) from error
                if item.source.is_directory:
                    result.append(self._materialize_evidence_bundle(item))
                else:
                    result.append(item.members[0][1])
            else:
                result.append(item)
        return result

    @staticmethod
    def _sniff(path: Path) -> str:
        if path.name.lower().endswith(".autotune.json"):
            return "triton-cache"
        known = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".parquet": "parquet",
            ".cpuprofile": "cpuprofile",
            ".heapprofile": "heapprofile",
            ".pftrace": "perfetto",
            ".perfetto-trace": "perfetto",
            ".trace": "xctrace",
            ".bin": "memray",
            ".nsys-rep": "nsys-rep",
            ".ncu-rep": "nsight-compute",
            ".ncu-repz": "nsight-compute",
            ".sarif": "sarif",
            ".pstats": "pstats",
        }
        if path.suffix.lower() in known:
            return known[path.suffix.lower()]
        if path.is_dir():
            return "directory"
        try:
            with path.open("rb") as stream:
                header = stream.read(16).lstrip()
        except OSError:
            return "unknown"
        if path.name == ".coverage" and header.startswith(b"SQLite format 3"):
            return "coverage"
        if header.startswith(b"PAR1"):
            return "parquet"
        if header.startswith((b"{", b"[")):
            return "json"
        return "text"

    def _evidence_destination(self, selection: EvidenceSelection) -> Path:
        key = hashlib.sha256(
            f"{selection.evidence_id}:{selection.source.role}".encode()
        ).hexdigest()
        return self.scratch / "evidence-sources" / key

    def _materialize_evidence_bundle(self, selection: EvidenceSelection) -> NativeSource:
        destination = self._evidence_destination(selection)
        key = (selection.evidence_id, f"evidence:{selection.source.role}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(
                tempfile.mkdtemp(prefix=f"{destination.name}.partial-", dir=destination.parent)
            )
            try:
                for relative, artifact in selection.members:
                    assert relative is not None
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        copy_verified_file(artifact, target)
                    except RuntimeFailure as error:
                        raise self._repository_failure(
                            RepositoryError("REPOSITORY_CORRUPTION", error.message)
                        ) from error
                stage.rename(destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        self.scratch_artifacts[key] = destination
        self.scratch_artifacts.move_to_end(key)
        source = selection.source
        try:
            digest, size, _ = hash_path(
                destination, max_bytes=source.size_bytes, max_files=len(selection.members)
            )
            if (digest, size) != (source.sha256, source.size_bytes):
                raise RuntimeFailure("MISSING_OR_CHANGED_INPUT", "Evidence materialization changed")
        except (OSError, RuntimeFailure) as error:
            raise self._repository_failure(
                RepositoryError("REPOSITORY_CORRUPTION", "Evidence materialization is invalid.")
            ) from error
        return NativeSource(
            destination,
            source.sha256,
            source.size_bytes,
            source.format,
            source.producer,
            source.role,
        )

    def _read_rows(
        self,
        sources: list[NativeSource],
        offset: int,
        limit: int,
        *,
        text_fragment_chars: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        rows: list[dict[str, Any]] = []
        observed = 0
        for source in sources:
            source_rows = (
                iter_text_fragments(source.path, text_fragment_chars)
                if text_fragment_chars is not None
                else self._iter_rows(source.path, source.format)
            )
            for row in source_rows:
                if observed >= offset and len(rows) < limit:
                    normalized = json.loads(json.dumps(row, default=str))
                    rows.append({**normalized, "input_sha256": source.sha256})
                observed += 1
                if observed >= offset + limit + 1:
                    return rows, observed, False
        return rows, observed, True

    def _provider_analysis(
        self,
        capability_id: str,
        sources: list[NativeSource],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        limits: RequestLimits,
    ) -> ProviderAnalysis | None:
        try:
            if (
                capability_id in {"cpu.hotspots", "cpu.callers"}
                and len(sources) == 1
                and (
                    cpu_profile := self.cpu_profiles.analyze(
                        capability_id,
                        *(
                            self._perf_collapsed(sources[0], limits)
                            if sources[0].format == "perf-data"
                            else (sources[0].path, sources[0].format)
                        ),
                        arguments,
                        max_rows=max_rows,
                    )
                )
            ):
                return cpu_profile
            if kernel_evidence := self.kernel_evidence.analyze(
                capability_id,
                [source.path for source in sources],
                [source.format for source in sources],
                arguments,
                max_rows=max_rows,
            ):
                return kernel_evidence
            if nvbench := self.nvbench.analyze(
                capability_id,
                [source.path for source in sources],
                [source.format for source in sources],
                arguments,
                max_rows=max_rows,
            ):
                return nvbench
            if benchmark := self.benchmarks.analyze(
                capability_id,
                [source.path for source in sources],
                [source.format for source in sources],
                arguments,
                max_rows=max_rows,
                timeout_seconds=limits.timeout_seconds,
                maximum_rss_bytes=limits.max_memory_bytes,
                maximum_output_bytes=limits.max_output_bytes,
            ):
                return benchmark
            if len(sources) == 1 and (
                source_evidence := self.source_evidence.analyze(
                    capability_id,
                    sources[0].path,
                    sources[0].format,
                    arguments,
                    max_rows=max_rows,
                    timeout_seconds=limits.timeout_seconds,
                    maximum_rss_bytes=limits.max_memory_bytes,
                    maximum_output_bytes=limits.max_output_bytes,
                )
            ):
                return source_evidence
            if (
                len(sources) == 1
                and sources[0].format in {"pytest", "observations"}
                and capability_id
                in {
                    "failures.summary",
                    "pytest.fixtures",
                    "coverage.summary",
                    "static.performance_candidates",
                }
            ):
                return self.reliability.analyze(
                    capability_id, sources[0].path, sources[0].format, max_rows=max_rows
                )
            if (
                len(sources) == 1
                and sources[0].format == "otlp"
                and capability_id
                in {"trace.summary", "trace.operations", "trace.lifecycle", "trace.window"}
            ):
                return self.otlp.analyze(
                    sources[0].path,
                    capability_id,
                    arguments,
                    max_rows=max_rows,
                    timeout_seconds=limits.timeout_seconds,
                    maximum_rss_bytes=limits.max_memory_bytes,
                    maximum_output_bytes=limits.max_output_bytes,
                )
            if (
                len(sources) == 1
                and sources[0].format == "aiperf"
                and capability_id == "inference.summary"
            ):
                return self.aiperf.analyze(
                    sources[0].path,
                    max_rows=max_rows,
                    timeout_seconds=limits.timeout_seconds,
                    maximum_rss_bytes=limits.max_memory_bytes,
                    maximum_output_bytes=limits.max_output_bytes,
                )
            if capability_id == "inference.compare" and all(
                source.format == "aiperf" for source in sources
            ):
                analyses = [
                    self.aiperf.analyze(
                        source.path,
                        max_rows=max_rows,
                        timeout_seconds=limits.timeout_seconds,
                        maximum_rss_bytes=limits.max_memory_bytes,
                        maximum_output_bytes=limits.max_output_bytes,
                    )
                    for source in sources
                ]
                return self.aiperf.compare(analyses, arguments, max_rows=max_rows)
            if inference_export := self.inference_exports.analyze(
                capability_id,
                [source.path for source in sources],
                [source.format for source in sources],
                arguments,
                max_rows=max_rows,
            ):
                return inference_export
            if (
                len(sources) == 1
                and sources[0].format == "memray"
                and capability_id in {"memory.hotspots", "memory.retained"}
            ):
                return self.memray.analyze(
                    capability_id,
                    sources[0].path,
                    sources[0].sha256,
                    max_rows=max_rows,
                    max_input_bytes=limits.max_input_bytes,
                    max_output_bytes=limits.max_output_bytes,
                    timeout_seconds=limits.timeout_seconds,
                    maximum_rss_bytes=limits.max_memory_bytes,
                )
            if (
                len(sources) == 1
                and sources[0].format == "nsight-compute"
                and capability_id in {"gpu.kernel_metrics", "kernel.compare"}
            ):
                return self.nsight_compute.analyze(
                    sources[0].path,
                    max_rows=max_rows,
                    timeout_seconds=limits.timeout_seconds,
                    maximum_rss_bytes=limits.max_memory_bytes,
                    maximum_output_bytes=limits.max_output_bytes,
                )
            if platform_trace := self._platform_trace_analysis(
                capability_id, sources, max_rows=max_rows, limits=limits
            ):
                return platform_trace
            if len(sources) == 1 and (
                structured := self.structured_workers.analyze(
                    capability_id,
                    sources[0].path,
                    sources[0].sha256,
                    sources[0].format,
                    dict(arguments),
                    max_rows=max_rows,
                    timeout_seconds=limits.timeout_seconds,
                    maximum_rss_bytes=limits.max_memory_bytes,
                    maximum_output_bytes=limits.max_output_bytes,
                )
            ):
                return structured
            if (
                len(sources) != 1
                or capability_id
                not in {"trace.summary", "trace.call_graph", "trace.pytorch", "trace.window"}
                or sources[0].format
                not in {"perfetto", "chrome-trace", "pytorch", "rocprof-pftrace"}
            ):
                return None
            return self.perfetto.analyze(
                capability_id,
                sources[0].path,
                arguments,
                max_rows=max_rows,
                timeout_seconds=limits.timeout_seconds,
                maximum_rss_bytes=limits.max_memory_bytes,
                maximum_output_bytes=limits.max_output_bytes,
            )
        except ProviderFailure as error:
            raise RuntimeFailure(error.code, error.message, details=error.details) from error
        except DomainError as error:
            code = (
                "UNAVAILABLE_CAPABILITY"
                if error.code is ErrorCode.UNAVAILABLE_CAPABILITY
                else "EXECUTION_FAILURE"
                if isinstance(error, ProcessExecutionError)
                else "LIMIT_EXCEEDED"
                if error.code is ErrorCode.LIMIT_EXCEEDED
                else "DECODE_FAILURE"
            )
            raise RuntimeFailure(code, error.message) from error

    def _platform_trace_analysis(
        self,
        capability_id: str,
        sources: list[NativeSource],
        *,
        max_rows: int,
        limits: RequestLimits,
    ) -> ProviderAnalysis | None:
        if len(sources) != 1:
            return None
        source = sources[0]
        if source.format in {"nsys-rep", "nsys-parquet"} and capability_id in {
            "trace.summary",
            "trace.operations",
            "trace.lifecycle",
            "gpu.launches",
        }:
            provider_version = "parquetdir-v1"
            path = source.path
            if source.format == "nsys-rep":
                path, provider_version = self._nsys_parquetdir(source, limits)
            return self.nsight_systems.analyze(
                path,
                capability_id=capability_id,
                max_rows=max_rows,
                provider_version=provider_version,
            )
        if source.format == "xctrace" and capability_id == "trace.summary":
            path, provider_version = self._xctrace_toc(source, limits)
            return self.xctrace.analyze(path, max_rows=max_rows, provider_version=provider_version)
        return None

    def _perf_collapsed(self, source: NativeSource, limits: RequestLimits) -> tuple[Path, str]:
        binding = ExecutableResolver().require_host_tool(
            "perf", cwd=source.path.parent, environment=dict(os.environ)
        )
        provider_version = binding.identity.sha256
        key = (source.sha256, f"perf-script:{provider_version}")
        cached = self.scratch_artifacts.get(key)
        if cached is not None and cached.is_file():
            self.scratch_artifacts.move_to_end(key)
            return cached, "perf"
        conversion_root = self.scratch / "conversions"
        conversion_root.mkdir(exist_ok=True)
        output = conversion_root / f"perf-{source.sha256[:20]}-{provider_version[:12]}.folded"
        request = ExecutionRequest(
            argv=(str(binding.invocation_path), "script", "--input", str(source.path)),
            executable_binding=binding,
            cwd=source.path.parent,
            environment_allowlist=("PATH",),
            allowed_working_roots=(source.path.parent,),
            timeout_seconds=limits.timeout_seconds,
            max_output_bytes=limits.max_output_bytes,
            resource_policy=ResourcePolicy(
                filesystem_path=self.scratch,
                staging_root=self.scratch,
                writable_roots=(conversion_root,),
                minimum_free_bytes=64 * 1024 * 1024,
                maximum_rss_bytes=limits.max_memory_bytes,
                maximum_writable_growth_bytes=limits.max_output_bytes,
            ),
        )
        outcome = self.broker.run_sync(request)
        exit_code = process_exit_code(outcome.process.termination)
        if exit_code != 0:
            retained_stderr = outcome.stderr[:4096]
            raise RuntimeFailure(
                "DECODE_FAILURE",
                "perf script failed before producing trustworthy stack evidence",
                details={
                    "decoder_exit_code": exit_code,
                    "decoder_termination": outcome.process.termination.model_dump(mode="json"),
                    "stdout_bytes": len(outcome.stdout),
                    "stderr_bytes": len(outcome.stderr),
                    "stdout_complete": True,
                    "stderr_complete": True,
                    "decoder_stderr": retained_stderr.decode("utf-8", errors="replace"),
                    "decoder_stderr_retained_bytes": len(retained_stderr),
                    "decoder_stderr_omitted_bytes": len(outcome.stderr) - len(retained_stderr),
                },
            )
        try:
            self._write_collapsed_perf_script(outcome.stdout, output)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        self._cache_scratch_artifact(key, output)
        return output, "perf"

    @staticmethod
    def _write_collapsed_perf_script(payload: bytes, output: Path) -> None:
        stacks: dict[tuple[str, ...], int] = {}
        frames: list[str] = []

        def finish() -> None:
            if frames:
                stack = tuple(reversed(frames))
                stacks[stack] = stacks.get(stack, 0) + 1
                frames.clear()

        try:
            for raw_line in payload.decode("utf-8").splitlines():
                if not raw_line.strip():
                    finish()
                    continue
                if not raw_line[0].isspace():
                    finish()
                    continue
                frame = raw_line.strip()
                address_and_symbol, separator, dso = frame.rpartition(" (")
                address, separator_after_address, symbol = address_and_symbol.partition(" ")
                if (
                    not separator
                    or not dso.endswith(")")
                    or not separator_after_address
                    or re.fullmatch(r"(?:0x)?[0-9a-fA-F]+", address) is None
                ):
                    raise RuntimeFailure(
                        "DECODE_FAILURE", "perf script returned an unsupported callchain frame"
                    )
                symbol = re.sub(r"\+0x[0-9a-fA-F]+(?:/0x[0-9a-fA-F]+)?$", "", symbol)
                if not symbol or symbol == "[unknown]":
                    symbol = "[unknown]"
                frames.append(symbol.replace(";", ":"))
        except UnicodeDecodeError as error:
            raise RuntimeFailure("DECODE_FAILURE", "perf script output is not UTF-8") from error
        finish()
        if not stacks:
            raise RuntimeFailure("DECODE_FAILURE", "perf script returned no stack samples")
        output.write_text(
            "".join(f"{';'.join(stack)} {count}\n" for stack, count in sorted(stacks.items()))
        )

    def _nsys_parquetdir(self, source: NativeSource, limits: RequestLimits) -> tuple[Path, str]:
        binding = ExecutableResolver().require_host_tool(
            "nsys", cwd=source.path.parent, environment=dict(os.environ)
        )
        provider_version = binding.identity.sha256
        key = (source.sha256, provider_version)
        cached = self.scratch_artifacts.get(key)
        if cached is not None and cached.is_dir():
            self.scratch_artifacts.move_to_end(key)
            return cached, provider_version
        conversion_root = self.scratch / "conversions"
        conversion_root.mkdir(exist_ok=True)
        output_base = conversion_root / f"nsys-{source.sha256[:20]}-{provider_version[:12]}"
        request = ExecutionRequest(
            argv=(
                str(binding.invocation_path),
                "export",
                "--type",
                "parquetdir",
                "--output",
                str(output_base),
                "--force-overwrite",
                "true",
                "--quiet",
                "true",
                str(source.path),
            ),
            executable_binding=binding,
            cwd=source.path.parent,
            environment_allowlist=("PATH",),
            allowed_working_roots=(source.path.parent,),
            timeout_seconds=limits.timeout_seconds,
            max_output_bytes=min(limits.max_output_bytes, 1024 * 1024),
            resource_policy=ResourcePolicy(
                filesystem_path=self.scratch,
                staging_root=self.scratch,
                writable_roots=(conversion_root,),
                minimum_free_bytes=64 * 1024 * 1024,
                maximum_rss_bytes=limits.max_memory_bytes,
                maximum_writable_growth_bytes=limits.max_output_bytes,
            ),
        )
        self.broker.run_sync(request)
        candidates = (output_base.with_suffix(".parquetdir"), output_base)
        exported = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if exported is None:
            raise RuntimeFailure(
                "EXECUTION_FAILURE", "Nsight Systems did not create a parquetdir export"
            )
        self._cache_scratch_artifact(key, exported)
        return exported, provider_version

    def _xctrace_toc(self, source: NativeSource, limits: RequestLimits) -> tuple[Path, str]:
        binding = ExecutableResolver().require_host_tool(
            "xcrun", cwd=source.path.parent, environment=dict(os.environ)
        )
        provider_version = binding.identity.sha256
        key = (source.sha256, f"xctrace-toc:{provider_version}")
        cached = self.scratch_artifacts.get(key)
        if cached is not None and cached.is_file():
            self.scratch_artifacts.move_to_end(key)
            return cached, provider_version
        conversion_root = self.scratch / "conversions"
        conversion_root.mkdir(exist_ok=True)
        output = conversion_root / f"xctrace-{source.sha256[:20]}-{provider_version[:12]}.xml"
        request = ExecutionRequest(
            argv=(
                str(binding.invocation_path),
                "xctrace",
                "export",
                "--input",
                str(source.path),
                "--toc",
                "--output",
                str(output),
            ),
            executable_binding=binding,
            cwd=source.path.parent,
            environment_allowlist=("PATH",),
            allowed_working_roots=(source.path.parent,),
            timeout_seconds=limits.timeout_seconds,
            max_output_bytes=min(limits.max_output_bytes, 1024 * 1024),
            resource_policy=ResourcePolicy(
                filesystem_path=self.scratch,
                staging_root=self.scratch,
                writable_roots=(conversion_root,),
                minimum_free_bytes=64 * 1024 * 1024,
                maximum_rss_bytes=limits.max_memory_bytes,
                maximum_writable_growth_bytes=limits.max_output_bytes,
            ),
        )
        self.broker.run_sync(request)
        if not output.is_file():
            raise RuntimeFailure(
                "EXECUTION_FAILURE", "xctrace did not create a table-of-contents export"
            )
        self._cache_scratch_artifact(key, output)
        return output, provider_version

    def _capture_resource_policy(
        self, limits: RequestLimits, *, budget: WorkloadBudget, writable_root: Path
    ) -> ResourcePolicy:
        _used_bytes, used_files = self._scratch_commitment(consuming_root=writable_root)
        remaining_files = MAX_SESSION_SCRATCH_FILES - used_files - 4
        if remaining_files < 1:
            raise RuntimeFailure(
                "LIMIT_EXCEEDED", "Capture would exceed the session scratch file ceiling"
            )
        return ResourcePolicy(
            filesystem_path=self.scratch,
            writable_roots=(writable_root,),
            minimum_free_bytes=64 * 1024 * 1024,
            maximum_rss_bytes=budget.max_memory_bytes,
            max_observed_files=remaining_files,
            maximum_writable_growth_bytes=limits.max_output_bytes,
        )

    def _iter_rows(self, path: Path, format_name: str) -> Iterator[dict[str, Any]]:
        if path.is_dir():
            for item in directory_files(path):
                yield {"path": item.relative_to(path).as_posix(), "size_bytes": item.stat().st_size}
        elif format_name == "parquet":
            import pyarrow.parquet as parquet

            for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
                yield from (dict(row) for row in batch.to_pylist())
        elif format_name == "csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                yield from (dict(row) for row in csv.DictReader(stream))
        elif format_name == "jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    yield value if isinstance(value, dict) else {"value": value}
        elif format_name in {
            "json",
            "sarif",
            "pyperf",
            "nvbench",
            "kernel-validation",
            "observations",
            "pytest",
            "coverage",
        }:
            yield from iter_json_rows(path)
        else:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    yield {"line": number, "text": line.rstrip("\n")}

    def _shrink_result(
        self, result: dict[str, Any], limit: int, identity: Mapping[str, Any], offset: int
    ) -> None:
        rows = result["blocks"][1]["rows"]
        had_rows = bool(rows)
        while len(canonical_bytes(result)) > limit and rows:
            rows.pop()
            result["coverage"].update(rows_returned=len(rows), complete=False)
            result["truncation"] = {"reason": "result_bytes", "next_offset": offset + len(rows)}
            result["continuation"] = self._encode_continuation(identity, offset + len(rows))
        if had_rows and not rows:
            recovery: dict[str, Any] = {}
            if identity.get("capability_id") == "artifact.preview" and all(
                item["format"] == "text" for item in identity.get("inputs", [])
            ):
                fragment_chars = identity.get("arguments", {}).get("text_fragment_chars")
                if fragment_chars == 1:
                    recovery = {
                        "recovery": (
                            "A single-character fragment and result metadata do not fit. "
                            "Use a larger max_result_bytes within server limits."
                        )
                    }
                else:
                    suggested = 128 if fragment_chars is None else max(1, fragment_chars // 2)
                    recovery = {
                        "recovery": (
                            f"Retry preview_artifact with text_fragment_chars={suggested} "
                            "and start a fresh page. Fragment offsets differ from line offsets. "
                            "If metadata still cannot fit, use a larger max_result_bytes "
                            "within server limits."
                        )
                    }
            raise RuntimeFailure(
                "LIMIT_EXCEEDED",
                "A result row exceeds max_result_bytes and cannot form an advancing page",
                details=recovery,
            )
        if len(canonical_bytes(result)) > limit:
            raise RuntimeFailure("LIMIT_EXCEEDED", "Result metadata exceeds max_result_bytes")

    def _bound_capture_result(
        self, result: dict[str, Any], limit: int, identity: Mapping[str, Any], offset: int
    ) -> None:
        capture = cast(dict[str, Any], result["capture"])
        executions = cast(list[dict[str, Any]], capture["executions"])
        if len(canonical_bytes(result)) <= limit:
            return
        compact = [
            {
                "case": item["case"],
                "block": item["block"],
                "returncode": item["returncode"],
                "executable_sha256": item.get("executable_sha256"),
                "collector_executable_sha256": item.get("collector_executable_sha256"),
                "workload_executable_sha256": item.get("workload_executable_sha256"),
                "returncode_scope": item.get("returncode_scope", "unknown"),
                "workload_returncode": item.get("workload_returncode"),
                "status": item["status"],
                "failure_code": item["failure_code"],
                "wall_time_ns": item["wall_time_ns"],
                "containment": item["containment"],
                "limit": item["limit"],
                "semantic_oracle": (
                    {
                        "status": item["semantic_oracle"]["status"],
                        "returncode": item["semantic_oracle"]["returncode"],
                        "failure_code": item["semantic_oracle"]["failure_code"],
                        "limit": item["semantic_oracle"].get("limit"),
                    }
                    if item["semantic_oracle"] is not None
                    else None
                ),
            }
            for item in executions
        ]
        capture["execution_count"] = len(compact)
        capture["executions"] = compact
        capture["executions_truncated"] = 0
        limitation = (
            "Inline capture provenance was compacted by max_result_bytes; full provenance "
            "remains available if this analysis is preserved."
        )
        result["limitations"].append(limitation)
        while compact and len(canonical_bytes(result)) > limit:
            compact.pop()
            capture["executions_truncated"] += 1
        if len(canonical_bytes(result)) > limit:
            self._shrink_result(result, limit, identity, offset)

    @staticmethod
    def _capture_outcome(executions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        succeeded = sum(item["status"] == "succeeded" for item in executions)
        failed = len(executions) - succeeded
        return {
            "status": "failed" if failed else "succeeded",
            "execution_count": len(executions),
            "succeeded_count": succeeded,
            "failed_count": failed,
        }

    @staticmethod
    def _experiment_blocks(
        experiment: ExperimentDesign, executions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        by_case_and_block = {(str(item["case"]), int(item["block"])): item for item in executions}
        baseline = experiment.cases[0].name
        rows: list[dict[str, Any]] = []
        limitations: list[str] = []
        for candidate_index, candidate in enumerate(experiment.cases[1:], 1):
            differences: list[float] = []
            eligible_blocks = 0
            for block in range(1, experiment.blocks + 1):
                baseline_execution = by_case_and_block[(baseline, block)]
                candidate_execution = by_case_and_block[(candidate.name, block)]
                if (
                    baseline_execution["status"] != "succeeded"
                    or candidate_execution["status"] != "succeeded"
                ):
                    continue
                eligible_blocks += 1
                differences.append(
                    float(candidate_execution[experiment.metric])
                    - float(baseline_execution[experiment.metric])
                )
            estimate = AnalysisRuntime._experiment_estimate(differences, experiment.estimand)
            confidence_low, confidence_high, method = AnalysisRuntime._experiment_interval(
                differences,
                experiment.estimand,
                seed=experiment.seed + candidate_index,
            )
            if estimate is None:
                decision = "inconclusive"
            elif abs(estimate) <= experiment.practical_threshold:
                decision = "within_threshold"
            elif estimate < -experiment.practical_threshold:
                decision = "practically_improved"
            else:
                decision = "practically_regressed"
            if eligible_blocks < experiment.blocks:
                limitations.append(
                    f"Experiment comparison {baseline} vs {candidate.name} excluded "
                    f"{experiment.blocks - eligible_blocks} failed or oracle-invalid blocks."
                )
            if len(differences) < 3:
                limitations.append(
                    f"Experiment comparison {baseline} vs {candidate.name} has fewer than "
                    "three eligible blocks; no confidence interval is reported."
                )
            rows.append(
                {
                    "baseline_case": baseline,
                    "candidate_case": candidate.name,
                    "metric": experiment.metric,
                    "unit": "ns",
                    "estimand": experiment.estimand,
                    "estimate": estimate,
                    "confidence_low": confidence_low,
                    "confidence_high": confidence_high,
                    "confidence_level": 0.95 if confidence_low is not None else None,
                    "method": method,
                    "practical_threshold": experiment.practical_threshold,
                    "point_estimate_classification": decision,
                    "paired_blocks": len(differences),
                    "declared_blocks": experiment.blocks,
                }
            )
        return (
            [
                {
                    "type": "metrics",
                    "values": {
                        "experiment_metric": experiment.metric,
                        "experiment_estimand": experiment.estimand,
                        "decision_basis": "descriptive_point_estimate",
                        "baseline_case": baseline,
                        "comparison_count": len(rows),
                    },
                },
                {"type": "table", "rows": rows},
            ],
            limitations,
        )

    @staticmethod
    def _experiment_estimate(values: Sequence[float], estimand: str) -> float | None:
        if not values:
            return None
        if estimand == "median_difference":
            return float(statistics.median(values))
        return float(statistics.fmean(values))

    @staticmethod
    def _experiment_interval(
        values: Sequence[float], estimand: str, *, seed: int
    ) -> tuple[float | None, float | None, str]:
        if len(values) < 3:
            return None, None, f"descriptive.{estimand}.v1"
        if all(value == values[0] for value in values):
            return values[0], values[0], f"analytic.constant.{estimand}.v1"
        generator = random.Random(seed)
        estimates = sorted(
            cast(
                float,
                AnalysisRuntime._experiment_estimate(
                    [values[generator.randrange(len(values))] for _ in values], estimand
                ),
            )
            for _ in range(1_999)
        )
        return (
            estimates[round((len(estimates) - 1) * 0.025)],
            estimates[round((len(estimates) - 1) * 0.975)],
            f"bootstrap.percentile.paired.{estimand}.v1",
        )

    def _encode_continuation(self, identity: Mapping[str, Any], offset: int) -> str:
        payload = {
            "request": self._continuation_digest(identity),
            "offset": offset,
        }
        checksum = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        return (
            base64.urlsafe_b64encode(canonical_bytes({"checksum": checksum, "payload": payload}))
            .decode()
            .rstrip("=")
        )

    def _decode_continuation(
        self, token: str | None, identity: Mapping[str, Any], default: int
    ) -> int:
        if token is None:
            return default
        try:
            value = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
            payload = value["payload"]
            expected = hashlib.sha256(canonical_bytes(payload)).hexdigest()
            if not secrets.compare_digest(value["checksum"], expected) or payload[
                "request"
            ] != self._continuation_digest(identity):
                raise ValueError
            offset = payload["offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError
            return offset
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeFailure(
                "INVALID_INPUT", "Continuation does not match this request and its inputs"
            ) from exc

    @staticmethod
    def _continuation_digest(identity: Mapping[str, Any]) -> str:
        semantic = dict(identity)
        if "inputs" in semantic:
            semantic["inputs"] = [
                {key: value for key, value in item.items() if key not in {"path", "role"}}
                for item in semantic["inputs"]
            ]
        return hashlib.sha256(canonical_bytes(semantic)).hexdigest()

    @staticmethod
    def _resolve_capture_cwd(value: str) -> Path:
        candidate = Path(value)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise RuntimeFailure("INVALID_INPUT", "cwd must be an existing directory") from exc
        if not resolved.is_dir():
            raise RuntimeFailure("INVALID_INPUT", "cwd must be a directory")
        return resolved

    def _scratch_commitment(self, *, consuming_root: Path | None = None) -> tuple[int, int]:
        files = [item for item in self.scratch.rglob("*") if item.is_file()]
        sizes = {item: item.stat().st_size for item in files}
        used_bytes, used_files = sum(sizes.values()), len(sizes)
        for root, (reserved_bytes, reserved_files) in self._capture_reservations.items():
            if consuming_root is not None and consuming_root.is_relative_to(root):
                continue
            allocated = [size for path, size in sizes.items() if path.is_relative_to(root)]
            used_bytes += max(0, reserved_bytes - sum(allocated))
            used_files += max(0, reserved_files - len(allocated))
        return used_bytes, used_files

    @asynccontextmanager
    async def _capture_scope(self) -> AsyncIterator[Path]:
        root = self.scratch / f"capture-{secrets.token_hex(12)}"
        failed = True
        try:
            yield root
            failed = False
        finally:
            with anyio.CancelScope(shield=True):
                await self.run_in_request(partial(self._release_capture_scope, root, failed))

    def _release_capture_scope(self, root: Path, failed: bool) -> None:
        if failed:
            for analysis_id, cached in list(self.analyses.items()):
                if any(
                    source.path.is_relative_to(root) or root.is_relative_to(source.path)
                    for source in cached.sources
                ):
                    del self.analyses[analysis_id]
        self._capture_reservations.pop(root, None)
        if not any(
            source.path.is_relative_to(root) or root.is_relative_to(source.path)
            for cached in self.analyses.values()
            if cached.preserved is None
            for source in cached.sources
        ):
            shutil.rmtree(root, ignore_errors=True)
