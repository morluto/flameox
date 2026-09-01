"""Bounded process-lifespan analysis runtime."""

from __future__ import annotations

import base64
import csv
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import ijson
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import TypeAdapter

from flameox import __version__
from flameox.canonical import canonical_bytes
from flameox.command_binding import ExecutableResolver
from flameox.executable_models import ResolvedExecutable
from flameox.execution import (
    ExecutionRequest,
    ProcessExecutionError,
    ResourcePolicy,
    SubprocessBroker,
)
from flameox.providers.aiperf import AIPerfProvider
from flameox.providers.benchmarks import BenchmarkProvider
from flameox.providers.contracts import (
    ProviderAnalysis,
    ProviderFailure,
    canonical_provider_projection,
)
from flameox.providers.cpu import CpuProfileProvider
from flameox.providers.inference_exports import InferenceExportProvider
from flameox.providers.kernel_evidence import KernelEvidenceProvider
from flameox.providers.memray import MemrayProvider
from flameox.providers.nsight_compute import NsightComputeProvider, find_report_interface
from flameox.providers.nsight_systems import NsightSystemsParquetProvider
from flameox.providers.nvbench import NvbenchProvider
from flameox.providers.otlp import OtlpProvider
from flameox.providers.perfetto import PerfettoProvider
from flameox.providers.reliability import ReliabilityProvider
from flameox.providers.source_evidence import SourceEvidenceProvider
from flameox.providers.structured_workers import StructuredWorkerProviders
from flameox.providers.xctrace import XctraceProvider
from flameox.repository import (
    EvidenceRepository,
    NativeArtifact,
    RepositoryError,
    sha256_file,
)
from flameox.runtime_contracts import (
    CAPABILITIES,
    CAPABILITY_BY_ID,
    CAPTURE_PROBES,
    FORMAT_PROBES,
    MAX_INPUTS,
    MAX_ROWS,
    MAX_SNIFF_INPUTS,
    AnalysisResult,
    BenchmarkSamplesCaptureArguments,
    Capability,
    CaptureArguments,
    CaptureTarget,
    ComputeSanitizerCaptureArguments,
    CoverageCaptureArguments,
    EmptyArguments,
    EvidenceSource,
    ExperimentCase,
    ExperimentDesign,
    MemrayCaptureArguments,
    NsightComputeCaptureArguments,
    NsightSystemsCaptureArguments,
    PathSource,
    PerfCaptureArguments,
    PreviewArguments,
    ProviderProbe,
    PyperfCaptureArguments,
    PySpyCaptureArguments,
    RequestLimits,
    RocprofCaptureArguments,
    RuntimeFailure,
    Source,
    StrictModel,
    TorchProfilerCaptureArguments,
    XctraceCaptureArguments,
)
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.harness import IsolatedWorkerHarness, WorkerRuntimeConfig


@dataclass(slots=True)
class ResolvedSource:
    path: Path
    sha256: str
    size_bytes: int
    format: str
    producer: str | None
    role: str

    def public(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "producer": self.producer,
            "role": self.role,
        }


@dataclass(slots=True)
class CachedAnalysis:
    result: dict[str, Any]
    sources: list[ResolvedSource]
    manifest_body: dict[str, Any]
    preserved: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CaptureInvocation:
    argv: tuple[str, ...]
    environment: dict[str, str]
    artifacts: tuple[tuple[Path, str, str], ...]


def _perf_permission_limited(executable_path: str) -> bool:
    try:
        probe = subprocess.run(
            [executable_path, "stat", "-e", "task-clock", "--", "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=2,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    diagnostic = probe.stderr.casefold()
    return probe.returncode != 0 and any(
        marker in diagnostic for marker in ("permission", "not permitted", "perf_event_paranoid")
    )


def _permission_guidance(provider_id: str, executable_path: str) -> tuple[str, str]:
    if provider_id == "perf" and os.access(executable_path, os.X_OK):
        return (
            "perf events are blocked by the host permission policy.",
            "Grant CAP_PERFMON or adjust kernel.perf_event_paranoid outside Flameox, then retry.",
        )
    return (
        f"Required executable is not executable: {executable_path}",
        "Grant execute permission outside Flameox, then retry.",
    )


class AnalysisRuntime:
    """Own registry, broker, scratch, conversions, and the session analysis cache."""

    def __init__(
        self,
        project_root: Path,
        *,
        limits: RequestLimits | None = None,
        scratch_max_bytes: int = 1024**3,
        scratch_max_files: int = 8192,
    ) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.limits = limits or RequestLimits()
        self.session_id = f"{os.getpid()}-{secrets.token_hex(8)}"
        self._temporary = tempfile.TemporaryDirectory(prefix="flameox-session-")
        self.scratch = Path(self._temporary.name)
        self.scratch_max_bytes, self.scratch_max_files = scratch_max_bytes, scratch_max_files
        self.broker = SubprocessBroker()
        self.workers = IsolatedWorkerHarness(
            WorkerRuntimeConfig(
                project_root=self.project_root,
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
        self.memray = MemrayProvider(self.workers, self.project_root)
        self.nvbench = NvbenchProvider()
        self.nsight_compute = NsightComputeProvider(self.workers)
        self.nsight_systems = NsightSystemsParquetProvider()
        self.otlp = OtlpProvider(self.workers)
        self.reliability = ReliabilityProvider()
        self.source_evidence = SourceEvidenceProvider(self.workers, self.project_root)
        self.structured_workers = StructuredWorkerProviders(self.workers, self.project_root)
        self.xctrace = XctraceProvider()
        self.repository = EvidenceRepository(self.project_root, self.session_id)
        self.analyses: dict[str, CachedAnalysis] = {}
        self.conversions: dict[tuple[str, str], Path] = {}

    def close(self) -> None:
        self._temporary.cleanup()

    def discover_capabilities(
        self,
        intent: str | None = None,
        sources: Sequence[Source] | None = None,
        *,
        include_unavailable: bool = False,
        limit: int = 10,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 50:
            raise RuntimeFailure("INVALID_INPUT", "limit must be between 1 and 50")
        supplied = sources or []
        if len(supplied) > MAX_SNIFF_INPUTS:
            raise RuntimeFailure("INVALID_INPUT", "discovery accepts at most 16 sources")
        sniffed = [self._sniff_source(item) for item in supplied]
        formats = {str(item["format"]) for item in sniffed}
        terms = set((intent or "").lower().replace("_", " ").replace(".", " ").split())
        ranked: list[tuple[int, Capability, dict[str, Any]]] = []
        for capability in CAPABILITIES:
            state = self._availability(capability, formats)
            score = 3 * len(terms.intersection(capability.id.replace(".", " ").split())) + 5 * len(
                formats.intersection(capability.formats)
            )
            if not terms and not formats:
                score = 1
            if state["available"] or include_unavailable:
                ranked.append((score, capability, state))
        ranked.sort(key=lambda item: (-item[0], item[1].id))
        return {
            "sniffed_sources": sniffed,
            "capabilities": [
                {
                    "id": cap.id,
                    "summary": cap.summary,
                    "accepted_formats": list(cap.formats),
                    "providers": state["providers"],
                    "available": state["available"],
                    "overhead": "Bounded local analysis; capture overhead is provider-specific.",
                    "limitations": state["limitations"],
                    "remediation": state["remediation"],
                }
                for _, cap, state in ranked[:limit]
            ],
        }

    def inspect_capabilities(self, capability_ids: list[str]) -> dict[str, Any]:
        if not 1 <= len(capability_ids) <= 16:
            raise RuntimeFailure("INVALID_INPUT", "capability_ids must contain 1 to 16 entries")
        if unknown := [item for item in capability_ids if item not in CAPABILITY_BY_ID]:
            raise RuntimeFailure("UNKNOWN_CAPABILITY", f"Unknown capability: {unknown[0]}")
        return {
            "capabilities": [
                {
                    "id": cap.id,
                    "summary": cap.summary,
                    "source_modes": ["path", "evidence"],
                    "accepted_formats": list(cap.formats),
                    "providers": self._availability(cap, set())["providers"],
                    "capture_providers": self._capture_provider_details(cap),
                    "argument_schema": cap.model.model_json_schema(mode="validation"),
                    "example": {
                        "capability_id": cap.id,
                        "sources": [{"kind": "path", "path": "/absolute/path/to/artifact"}],
                        "arguments": {},
                    },
                    "limits": self.limits.model_dump(mode="json"),
                    "capture_semantics": (
                        "Typed argv is executed through the bounded subprocess broker."
                    ),
                    "limitations": [cap.limitation],
                }
                for cap in (CAPABILITY_BY_ID[item] for item in capability_ids)
            ]
        }

    def _capture_provider_details(self, capability: Capability) -> list[dict[str, Any]]:
        providers: list[tuple[str, type[StrictModel], str]] = []
        if capability.id == "artifact.preview":
            providers.append(("direct", EmptyArguments, "stdout and stderr text"))
        if "pyperf" in capability.formats:
            providers.append(("pyperf", PyperfCaptureArguments, "native pyperf JSON suite"))
        if "py-spy" in capability.formats:
            providers.append(("py-spy", PySpyCaptureArguments, "py-spy Speedscope profile"))
        if "perf-data" in capability.formats:
            providers.append(("perf", PerfCaptureArguments, "native perf.data profile"))
        if "cpuprofile" in capability.formats:
            providers.append(("node-cpu-profile", EmptyArguments, "native V8 CPU profile"))
        if "memray" in capability.formats:
            providers.append(("memray", MemrayCaptureArguments, "native Memray allocation file"))
        if "pytorch" in capability.formats:
            providers.append(
                (
                    "torch-profiler",
                    TorchProfilerCaptureArguments,
                    "native PyTorch Chrome trace",
                )
            )
        if "samples" in capability.formats:
            providers.append(
                (
                    "benchmark-samples",
                    BenchmarkSamplesCaptureArguments,
                    "native Flameox benchmark samples",
                )
            )
        if "nvbench" in capability.formats:
            providers.append(("nvbench", EmptyArguments, "native NVBench JSON-bin directory"))
        if "compute-sanitizer" in capability.formats:
            providers.append(
                (
                    "compute-sanitizer",
                    ComputeSanitizerCaptureArguments,
                    "native Compute Sanitizer log",
                )
            )
        if "nsys-rep" in capability.formats:
            providers.append(
                (
                    "nsight-systems",
                    NsightSystemsCaptureArguments,
                    "native Nsight Systems report",
                )
            )
        if "nsight-compute" in capability.formats:
            providers.append(
                (
                    "nsight-compute",
                    NsightComputeCaptureArguments,
                    "native Nsight Compute report",
                )
            )
        if "rocprof-pftrace" in capability.formats:
            providers.append(("rocprofv3", RocprofCaptureArguments, "native ROCprof PFTrace"))
        if "xctrace" in capability.formats:
            providers.append(("xctrace", XctraceCaptureArguments, "native Apple .trace bundle"))
        if "coverage" in capability.formats:
            providers.append(("coverage", CoverageCaptureArguments, "native coverage.py data file"))
        if "pytest" in capability.formats:
            providers.append(("pytest", EmptyArguments, "bounded native pytest event stream"))
        return [
            {
                "id": provider_id,
                "capture_argument_schema": model.model_json_schema(mode="validation"),
                "artifact": artifact,
                "availability": self._probe_provider(CAPTURE_PROBES[provider_id]),
            }
            for provider_id, model, artifact in providers
        ]

    def analyze(
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
            raise RuntimeFailure("UNKNOWN_CAPABILITY", f"Unknown capability: {capability_id}")
        selected_limits = limits.lowered_against(self.limits) if limits else self.limits
        validated = TypeAdapter(capability.model).validate_python(arguments)
        resolved = self._resolve_sources(sources, selected_limits)
        if capability.id != "artifact.preview" and (
            bad := next(
                (item.format for item in resolved if item.format not in capability.formats), None
            )
        ):
            raise RuntimeFailure("UNSUPPORTED_FORMAT", f"Unsupported format: {bad}")
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
        }
        default_offset = validated.offset if isinstance(validated, PreviewArguments) else 0
        offset = self._decode_continuation(continuation, identity, default_offset)
        analysis_request = {**identity, "offset": offset}
        analysis_id = hashlib.sha256(
            self.session_id.encode() + canonical_bytes(analysis_request)
        ).hexdigest()
        if cached := self.analyses.get(analysis_id):
            return self._copy_result(cached.result)
        provider_analysis = canonical_provider_projection(
            self._provider_analysis(
                capability_id,
                resolved,
                validated.model_dump(mode="json"),
                # Projection providers must see the same bounded prefix on every page.
                # Asking for offset + page size changes aggregates and mixed metric quotas.
                max_rows=MAX_ROWS + 1,
                limits=selected_limits,
            )
        )
        if provider_analysis is None and capability.id != "artifact.preview":
            raise RuntimeFailure(
                "UNSUPPORTED_FORMAT",
                f"No typed {capability_id} provider accepts the supplied inputs",
            )
        if provider_analysis is None:
            rows, observed, complete = self._read_rows(resolved, offset, selected_limits.max_rows)
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
        else:
            provider_rows = provider_analysis.blocks[-1]["rows"]
            if not isinstance(provider_rows, list):
                raise RuntimeFailure("DECODE_FAILURE", "Provider table block is invalid")
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
            current_digest, current_size, _ = self._hash_path(
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
        self.analyses[analysis_id] = CachedAnalysis(
            self._copy_result(validated_result), resolved, body
        )
        return self._copy_result(validated_result)

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
        selected_limits = limits.lowered_against(self.limits) if limits else self.limits
        capability = CAPABILITY_BY_ID.get(capability_id)
        if capability is None:
            raise RuntimeFailure("UNKNOWN_CAPABILITY", f"Unknown capability: {capability_id}")
        capture_arguments = self._capture_arguments(target.provider_id, target.capture_arguments)
        TypeAdapter(capability.model).validate_python(target.analysis_arguments)
        if (mode == "experiment") != (experiment is not None):
            raise RuntimeFailure(
                "INVALID_INPUT", "experiment mode and design must be supplied together"
            )
        cases = experiment.cases if experiment else [ExperimentCase(name="single")]
        blocks = experiment.blocks if experiment else 1
        total = len(cases) * blocks
        self._reserve_capture_capacity(
            total,
            provider_id=target.provider_id,
            has_oracle=experiment is not None and experiment.semantic_oracle is not None,
            limits=selected_limits,
        )
        captured: list[ResolvedSource] = []
        analysis_sources: list[ResolvedSource] = []
        executions: list[dict[str, Any]] = []
        request_scratch = self.scratch / f"capture-{secrets.token_hex(12)}"
        request_scratch.mkdir()
        sequence = 0
        for block in range(blocks):
            block_cases = list(cases)
            if experiment is not None:
                random.Random(experiment.seed + block).shuffle(block_cases)
            for case in block_cases:
                sequence += 1
                if progress:
                    await progress(sequence - 1, total, f"capture {case.name} block {block + 1}")
                argv = case.argv or target.argv
                environment = {**target.environment, **case.environment}
                if len(environment) > 32:
                    raise RuntimeFailure(
                        "INVALID_INPUT",
                        "merged capture environment must contain at most 32 entries",
                    )
                cwd = self._resolve_project_cwd(target.cwd)
                directory = request_scratch / f"case-{sequence:04d}"
                directory.mkdir()
                invocation, binding = self._bind_capture_invocation(
                    target.provider_id,
                    argv,
                    environment,
                    capture_arguments,
                    directory,
                    cwd=cwd,
                    request_scratch=request_scratch,
                )
                self._check_capture_provenance_capacity(
                    target=target,
                    mode=mode,
                    experiment=experiment,
                    executions=executions,
                    case=case,
                    block=block + 1,
                    argv=argv,
                    capture_argv=invocation.argv,
                    cwd=cwd,
                    limit=selected_limits.max_provenance_bytes,
                )
                request = ExecutionRequest(
                    argv=invocation.argv,
                    executable_binding=binding,
                    cwd=cwd,
                    environment_allowlist=("PATH",),
                    environment_overrides=invocation.environment,
                    allowed_working_roots=(self.project_root,),
                    timeout_seconds=selected_limits.timeout_seconds,
                    max_output_bytes=selected_limits.max_output_bytes,
                    resource_policy=self._resource_policy(selected_limits, writable_root=directory),
                )
                failure_code: str | None = None
                try:
                    outcome = await self.broker.run(request)
                    stdout = outcome.stdout
                    stderr = outcome.stderr
                    process = outcome.process
                    containment = outcome.containment.value
                except ProcessExecutionError as error:
                    stdout = error.stdout or b""
                    stderr = error.stderr or b""
                    process = error.process
                    containment = "broker"
                    failure_code = error.code.value
                for role, content in (("stdout", stdout), ("stderr", stderr)):
                    path = directory / f"{role}.txt"
                    path.write_bytes(content)
                    digest, size = sha256_file(path)
                    resolved_output = ResolvedSource(
                        path,
                        digest,
                        size,
                        "text",
                        target.provider_id,
                        f"capture-{sequence:04d}/{role}",
                    )
                    captured.append(resolved_output)
                    if target.provider_id == "direct":
                        analysis_sources.append(resolved_output)
                missing_artifact_roles: list[str] = []
                for path, format_name, role in invocation.artifacts:
                    native = self._resolve_capture_artifact(
                        path,
                        format_name=format_name,
                        role=role,
                        provider_id=target.provider_id,
                        limits=selected_limits,
                    )
                    if native is None:
                        missing_artifact_roles.append(role)
                        continue
                    captured_native = ResolvedSource(
                        native.path,
                        native.sha256,
                        native.size_bytes,
                        native.format,
                        native.producer,
                        f"capture-{sequence:04d}/{native.role}",
                    )
                    captured.append(captured_native)
                    analysis_sources.append(captured_native)
                oracle: dict[str, Any] | None = None
                if experiment is not None and experiment.semantic_oracle is not None:
                    oracle_argv = experiment.semantic_oracle
                    oracle_environment = {
                        **environment,
                        "FLAMEOX_CAPTURE_STDOUT": str(directory / "stdout.txt"),
                        "FLAMEOX_CAPTURE_STDERR": str(directory / "stderr.txt"),
                    }
                    oracle_binding = self._require_capture_tool(
                        oracle_argv[0],
                        cwd=cwd,
                        environment={**os.environ, **oracle_environment},
                        request_scratch=request_scratch,
                    )
                    oracle_request = ExecutionRequest(
                        argv=tuple(oracle_argv),
                        executable_binding=oracle_binding,
                        cwd=cwd,
                        environment_allowlist=("PATH",),
                        environment_overrides=oracle_environment,
                        allowed_working_roots=(self.project_root,),
                        timeout_seconds=selected_limits.timeout_seconds,
                        max_output_bytes=selected_limits.max_output_bytes,
                        resource_policy=self._resource_policy(
                            selected_limits, writable_root=directory
                        ),
                    )
                    try:
                        oracle_outcome = await self.broker.run(oracle_request)
                        oracle_stdout = oracle_outcome.stdout
                        oracle_stderr = oracle_outcome.stderr
                        oracle_process = oracle_outcome.process
                        oracle_failure_code: str | None = None
                    except ProcessExecutionError as error:
                        oracle_stdout = error.stdout or b""
                        oracle_stderr = error.stderr or b""
                        oracle_process = error.process
                        oracle_failure_code = error.code.value
                    oracle_exit_code = getattr(oracle_process.termination, "exit_code", None)
                    for role, content in (
                        ("oracle_stdout", oracle_stdout),
                        ("oracle_stderr", oracle_stderr),
                    ):
                        path = directory / f"{role}.txt"
                        path.write_bytes(content)
                        digest, size = sha256_file(path)
                        captured.append(
                            ResolvedSource(
                                path,
                                digest,
                                size,
                                "text",
                                target.provider_id,
                                f"capture-{sequence:04d}/{role}",
                            )
                        )
                    oracle = {
                        "argv": oracle_argv,
                        "returncode": oracle_exit_code,
                        "status": "passed" if oracle_exit_code == 0 else "failed",
                        "failure_code": oracle_failure_code,
                    }
                self._assert_scratch_capacity()
                termination = process.termination
                exit_code = getattr(termination, "exit_code", None)
                status = "succeeded" if exit_code == 0 else "failed"
                if status == "succeeded" and missing_artifact_roles:
                    status = "failed"
                    failure_code = "CAPTURE_ARTIFACT_MISSING"
                if oracle is not None and oracle["status"] == "failed":
                    status = "failed"
                    failure_code = "SEMANTIC_ORACLE_FAILED"
                executions.append(
                    {
                        "case": case.name,
                        "block": block + 1,
                        "argv": argv,
                        "capture_argv": list(invocation.argv),
                        "cwd": str(cwd),
                        "returncode": exit_code,
                        "status": status,
                        "failure_code": failure_code,
                        "missing_artifact_roles": missing_artifact_roles,
                        "semantic_oracle": oracle,
                        "wall_time_ns": process.wall_time_ns,
                        "containment": containment,
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
                    await progress(sequence, total, f"captured {case.name} block {block + 1}")
        effective_capability_id, analysis_sources = self._capture_analysis_sources(
            capability_id, analysis_sources, captured
        )
        try:
            result = self.analyze(
                effective_capability_id,
                [
                    PathSource(path=str(item.path), format=item.format, producer=item.producer)
                    for item in analysis_sources
                ],
                target.analysis_arguments,
                limits=selected_limits,
            )
        except RuntimeFailure as error:
            if not preserve:
                raise
            result = self._capture_failure_result(
                target=target,
                capability_id=capability_id,
                mode=mode,
                experiment=experiment,
                executions=executions,
                captured=captured,
                limits=selected_limits,
                failure=error,
            )
            result["preserved"] = self.preserve_evidence(str(result["analysis_id"]))
            return result
        cached = self.analyses[str(result["analysis_id"])]
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
        return result

    def _capture_failure_result(
        self,
        *,
        target: CaptureTarget,
        capability_id: str,
        mode: str,
        experiment: ExperimentDesign | None,
        executions: list[dict[str, Any]],
        captured: list[ResolvedSource],
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
                "Native capture succeeded, but requested analysis failed; no analysis claims "
                "are available."
            ],
            "continuation": None,
            "capture": {
                "mode": mode,
                "requested_capability_id": capability_id,
                "executions": executions,
            },
            "analysis_failure": failure_body,
        }
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
        self.analyses[analysis_id] = CachedAnalysis(validated, captured, manifest_body)
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

    def _check_capture_provenance_capacity(
        self,
        *,
        target: CaptureTarget,
        mode: str,
        experiment: ExperimentDesign | None,
        executions: Sequence[Mapping[str, Any]],
        limit: int,
        case: ExperimentCase | None = None,
        block: int | None = None,
        argv: Sequence[str] | None = None,
        capture_argv: Sequence[str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        candidate = list(executions)
        if case is not None:
            candidate.append(
                {
                    "case": case.name,
                    "block": block,
                    "argv": list(argv or ()),
                    "capture_argv": list(capture_argv or ()),
                    "cwd": str(cwd) if cwd is not None else str(self.project_root),
                    "returncode": None,
                    "status": "pending",
                    "failure_code": None,
                    "missing_artifact_roles": [],
                    "semantic_oracle": None,
                    "wall_time_ns": None,
                    "containment": "broker",
                }
            )
        if (
            len(canonical_bytes(self._capture_request_body(target, mode, experiment, candidate)))
            > limit
        ):
            raise RuntimeFailure(
                "LIMIT_EXCEEDED",
                "Capture provenance exceeds max_provenance_bytes",
            )

    @staticmethod
    def _require_host_tool(
        executable: str, *, cwd: Path, environment: Mapping[str, str]
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
            raise RuntimeFailure(code, error.message) from error

    def _require_capture_tool(
        self,
        executable: str,
        *,
        cwd: Path,
        environment: Mapping[str, str],
        request_scratch: Path,
    ) -> ResolvedExecutable:
        try:
            return self._require_host_tool(executable, cwd=cwd, environment=environment)
        except RuntimeFailure:
            shutil.rmtree(request_scratch, ignore_errors=True)
            raise

    def _bind_capture_invocation(
        self,
        provider_id: str,
        target_argv: list[str],
        environment: dict[str, str],
        arguments: CaptureArguments,
        directory: Path,
        *,
        cwd: Path,
        request_scratch: Path,
    ) -> tuple[CaptureInvocation, ResolvedExecutable]:
        try:
            invocation = self._capture_invocation(
                provider_id, target_argv, environment, arguments, directory
            )
            binding = self._require_capture_tool(
                invocation.argv[0],
                cwd=cwd,
                environment={**os.environ, **invocation.environment},
                request_scratch=request_scratch,
            )
        except OSError:
            shutil.rmtree(request_scratch, ignore_errors=True)
            raise
        return invocation, binding

    def _resolve_capture_artifact(
        self,
        path: Path,
        *,
        format_name: str,
        role: str,
        provider_id: str,
        limits: RequestLimits,
    ) -> ResolvedSource | None:
        if not path.exists() or (not path.is_file() and not path.is_dir()):
            return None
        digest, size, file_count = self._hash_path(
            path,
            max_bytes=limits.max_input_bytes,
            max_files=limits.max_input_files,
        )
        if file_count == 0:
            return None
        return ResolvedSource(path, digest, size, format_name, provider_id, role)

    @staticmethod
    def _capture_analysis_sources(
        capability_id: str,
        analysis_sources: list[ResolvedSource],
        captured: list[ResolvedSource],
    ) -> tuple[str, list[ResolvedSource]]:
        if analysis_sources:
            return capability_id, analysis_sources
        return "artifact.preview", [
            source for source in captured if source.role.endswith(("/stdout", "/stderr"))
        ]

    @staticmethod
    def _capture_arguments(provider_id: str, arguments: Mapping[str, Any]) -> CaptureArguments:
        models: dict[str, type[CaptureArguments]] = {
            "direct": EmptyArguments,
            "pyperf": PyperfCaptureArguments,
            "py-spy": PySpyCaptureArguments,
            "perf": PerfCaptureArguments,
            "node-cpu-profile": EmptyArguments,
            "memray": MemrayCaptureArguments,
            "torch-profiler": TorchProfilerCaptureArguments,
            "nvbench": EmptyArguments,
            "compute-sanitizer": ComputeSanitizerCaptureArguments,
            "nsight-systems": NsightSystemsCaptureArguments,
            "nsight-compute": NsightComputeCaptureArguments,
            "rocprofv3": RocprofCaptureArguments,
            "xctrace": XctraceCaptureArguments,
            "coverage": CoverageCaptureArguments,
            "benchmark-samples": BenchmarkSamplesCaptureArguments,
            "observations": EmptyArguments,
            "pytest": EmptyArguments,
        }
        model = models.get(provider_id)
        if model is None:
            raise RuntimeFailure("UNKNOWN_CAPABILITY", f"Unknown capture provider: {provider_id}")
        return model.model_validate(arguments)

    @staticmethod
    def _capture_invocation(
        provider_id: str,
        target_argv: list[str],
        environment: dict[str, str],
        arguments: CaptureArguments,
        directory: Path,
    ) -> CaptureInvocation:
        if provider_id in {
            "direct",
            "node-cpu-profile",
            "torch-profiler",
            "compute-sanitizer",
            "nsight-systems",
            "nsight-compute",
            "perf",
            "rocprofv3",
            "xctrace",
        }:
            return AnalysisRuntime._special_capture_invocation(
                provider_id, target_argv, environment, arguments, directory
            )
        if provider_id == "pyperf" and isinstance(arguments, PyperfCaptureArguments):
            output = directory / "benchmark.json"
            argv = (
                sys.executable,
                "-m",
                "pyperf",
                "command",
                "--quiet",
                "--output",
                str(output),
                "--processes",
                str(arguments.processes),
                "--values",
                str(arguments.values),
                "--warmups",
                str(arguments.warmups),
                "--loops",
                str(arguments.loops),
                "--min-time",
                str(arguments.min_time),
                "--name",
                arguments.name,
                *target_argv,
            )
            return CaptureInvocation(argv, environment, ((output, "pyperf", "benchmark"),))
        if provider_id == "py-spy" and isinstance(arguments, PySpyCaptureArguments):
            output = directory / "profile.speedscope.json"
            pyspy_options = ["--rate", str(arguments.rate)]
            if arguments.gil:
                pyspy_options.append("--gil")
            if arguments.native:
                pyspy_options.append("--native")
            return CaptureInvocation(
                (
                    "py-spy",
                    "record",
                    "--format",
                    "speedscope",
                    "--output",
                    str(output),
                    *pyspy_options,
                    "--",
                    *target_argv,
                ),
                environment,
                ((output, "py-spy", "cpu-profile"),),
            )
        if provider_id == "memray" and isinstance(arguments, MemrayCaptureArguments):
            if len(target_argv) < 2:
                raise RuntimeFailure(
                    "INVALID_INPUT",
                    "memray capture requires a Python interpreter followed by a script, -m, or -c",
                )
            output = directory / "memory.bin"
            memray_options = ["--output", str(output)]
            if arguments.native:
                memray_options.append("--native")
            return CaptureInvocation(
                (
                    target_argv[0],
                    "-m",
                    "memray",
                    "run",
                    *memray_options,
                    *target_argv[1:],
                ),
                environment,
                ((output, "memray", "memory"),),
            )
        if provider_id == "nvbench" and isinstance(arguments, EmptyArguments):
            output_root = directory / "nvbench"
            output_root.mkdir()
            output = output_root / "results.json"
            return CaptureInvocation(
                (*target_argv, "--jsonbin", str(output)),
                environment,
                ((output_root, "nvbench", "benchmark"),),
            )
        if provider_id == "benchmark-samples" and isinstance(
            arguments, BenchmarkSamplesCaptureArguments
        ):
            output = directory / "benchmark-samples.json"
            capture_environment = {
                **environment,
                "FLAMEOX_BENCHMARK_OUTPUT": str(output),
            }
            if arguments.torch_benchmark is not None:
                capture_environment["FLAMEOX_TORCH_BENCHMARK_CONFIG"] = json.dumps(
                    arguments.torch_benchmark.model_dump(mode="json"),
                    separators=(",", ":"),
                    sort_keys=True,
                )
            return CaptureInvocation(
                tuple(target_argv),
                capture_environment,
                ((output, "samples", "benchmark"),),
            )
        if provider_id == "observations" and isinstance(arguments, EmptyArguments):
            output = directory / "observations.jsonl"
            return CaptureInvocation(
                tuple(target_argv),
                {**environment, "FLAMEOX_OBSERVATIONS_PATH": str(output)},
                ((output, "observations", "observations"),),
            )
        if provider_id == "pytest" and isinstance(arguments, EmptyArguments):
            return AnalysisRuntime._pytest_capture_invocation(target_argv, environment, directory)
        if provider_id == "coverage" and isinstance(arguments, CoverageCaptureArguments):
            return AnalysisRuntime._coverage_capture_invocation(
                target_argv, environment, arguments, directory
            )
        raise RuntimeFailure(
            "INVALID_INPUT", f"Invalid arguments for capture provider: {provider_id}"
        )

    @staticmethod
    def _pytest_capture_invocation(
        target_argv: list[str], environment: dict[str, str], directory: Path
    ) -> CaptureInvocation:
        executable = Path(target_argv[0]).name
        if not executable.startswith("python") or target_argv[1:3] != ["-m", "pytest"]:
            raise RuntimeFailure("INVALID_INPUT", "pytest capture requires a python -m pytest argv")
        output = directory / "pytest.jsonl"
        runner = Path(__file__).with_name("pytest_runner.py").resolve()
        return CaptureInvocation(
            (target_argv[0], str(runner), "--output", str(output), "--", *target_argv[3:]),
            environment,
            ((output, "pytest", "reliability"),),
        )

    @staticmethod
    def _coverage_capture_invocation(
        target_argv: list[str],
        environment: dict[str, str],
        arguments: CoverageCaptureArguments,
        directory: Path,
    ) -> CaptureInvocation:
        if len(target_argv) < 2:
            raise RuntimeFailure(
                "INVALID_INPUT",
                "coverage capture requires a Python interpreter followed by a script or -m",
            )
        output = directory / ".coverage"
        empty_config = directory / "coverage.ini"
        empty_config.write_text("[run]\n")
        options: list[str] = [
            "--data-file",
            str(output),
            "--rcfile",
            str(empty_config),
        ]
        if arguments.branch:
            options.append("--branch")
        for name, values in (
            ("source", arguments.source),
            ("include", arguments.include),
            ("omit", arguments.omit),
        ):
            if values:
                options.extend((f"--{name}", ",".join(values)))
        argv = (
            target_argv[0],
            "-m",
            "coverage",
            "run",
            *options,
            *target_argv[1:],
        )
        return CaptureInvocation(argv, environment, ((output, "coverage", "coverage"),))

    @staticmethod
    def _special_capture_invocation(
        provider_id: str,
        target_argv: list[str],
        environment: dict[str, str],
        arguments: CaptureArguments,
        directory: Path,
    ) -> CaptureInvocation:
        if provider_id == "direct" and isinstance(arguments, EmptyArguments):
            return CaptureInvocation(tuple(target_argv), environment, ())
        if provider_id == "node-cpu-profile" and isinstance(arguments, EmptyArguments):
            output = directory / "profile.cpuprofile"
            return CaptureInvocation(
                (
                    target_argv[0],
                    "--cpu-prof",
                    f"--cpu-prof-dir={directory}",
                    f"--cpu-prof-name={output.name}",
                    *target_argv[1:],
                ),
                environment,
                ((output, "cpuprofile", "cpu-profile"),),
            )
        if provider_id == "torch-profiler" and isinstance(arguments, TorchProfilerCaptureArguments):
            output_root = directory / "torch-profiler"
            output_root.mkdir()
            config = {
                "mode": "sdk",
                "activities": arguments.activities,
                "schedule": {
                    "wait": arguments.wait,
                    "warmup": arguments.warmup,
                    "active": arguments.active,
                    "repeat": 1,
                    "skip_first": arguments.skip_first,
                },
                "record_shapes": arguments.record_shapes,
                "profile_memory": arguments.profile_memory,
                "with_stack": arguments.with_stack,
                "with_flops": arguments.with_flops,
                "with_modules": arguments.with_modules,
            }
            output = output_root / "torch-trace-cycle-0000.json"
            return CaptureInvocation(
                tuple(target_argv),
                {
                    **environment,
                    "FLAMEOX_TORCH_PROFILER_CONFIG": json.dumps(
                        config, separators=(",", ":"), sort_keys=True
                    ),
                    "FLAMEOX_TORCH_PROFILER_OUTPUT_ROOT": str(output_root),
                },
                ((output, "pytorch", "trace"),),
            )
        if provider_id == "compute-sanitizer" and isinstance(
            arguments, ComputeSanitizerCaptureArguments
        ):
            output = directory / "compute-sanitizer.log"
            return CaptureInvocation(
                (
                    "compute-sanitizer",
                    "--tool",
                    arguments.tool,
                    "--xml",
                    "--save",
                    str(output),
                    "--error-exitcode",
                    "99",
                    *target_argv,
                ),
                environment,
                ((output, "compute-sanitizer", "sanitizer"),),
            )
        if provider_id == "perf" and isinstance(arguments, PerfCaptureArguments):
            output = directory / "perf.data"
            return CaptureInvocation(
                (
                    "perf",
                    "record",
                    "--freq",
                    str(arguments.frequency),
                    "--call-graph",
                    arguments.call_graph,
                    "--output",
                    str(output),
                    "--",
                    *target_argv,
                ),
                environment,
                ((output, "perf-data", "cpu-profile"),),
            )
        if provider_id == "nsight-systems" and isinstance(arguments, NsightSystemsCaptureArguments):
            output_stem = directory / "nsight-systems"
            output = output_stem.with_suffix(".nsys-rep")
            return CaptureInvocation(
                (
                    "nsys",
                    "profile",
                    f"--trace={','.join(arguments.trace)}",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--resolve-symbols=false",
                    "--force-overwrite=true",
                    "--output",
                    str(output_stem),
                    *target_argv,
                ),
                environment,
                ((output, "nsys-rep", "trace"),),
            )
        if provider_id == "nsight-compute" and isinstance(arguments, NsightComputeCaptureArguments):
            output = directory / "nsight-compute.ncu-rep"
            ncu_options: list[str] = [
                "--export",
                str(output),
                "--force-overwrite",
                "--replay-mode",
                arguments.replay_mode,
                "--launch-skip",
                str(arguments.launch_skip),
                "--launch-count",
                str(arguments.launch_count),
            ]
            for section in arguments.section:
                ncu_options.extend(("--section", section))
            return CaptureInvocation(
                ("ncu", *ncu_options, *target_argv),
                environment,
                ((output, "nsight-compute", "kernel-metrics"),),
            )
        if provider_id == "rocprofv3" and isinstance(arguments, RocprofCaptureArguments):
            output_root = directory / "rocprof"
            output_root.mkdir()
            output = output_root / "rocprofv3_results.pftrace"
            options: list[str] = []
            for enabled, flag in (
                (arguments.hip_trace, "--hip-trace"),
                (arguments.kernel_trace, "--kernel-trace"),
                (arguments.memory_copy_trace, "--memory-copy-trace"),
                (arguments.memory_allocation_trace, "--memory-allocation-trace"),
                (arguments.scratch_memory_trace, "--scratch-memory-trace"),
                (arguments.marker_trace, "--marker-trace"),
            ):
                if enabled:
                    options.append(flag)
            return CaptureInvocation(
                (
                    "rocprofv3",
                    "--output-format",
                    "pftrace",
                    "-o",
                    "rocprofv3",
                    "-d",
                    str(output_root),
                    *options,
                    "--",
                    *target_argv,
                ),
                environment,
                ((output, "rocprof-pftrace", "trace"),),
            )
        if provider_id == "xctrace" and isinstance(arguments, XctraceCaptureArguments):
            output = directory / "capture.trace"
            return CaptureInvocation(
                (
                    "xcrun",
                    "xctrace",
                    "record",
                    "--template",
                    arguments.template,
                    "--output",
                    str(output),
                    "--launch",
                    "--",
                    *target_argv,
                ),
                environment,
                ((output, "xctrace", "trace"),),
            )
        raise RuntimeFailure(
            "INVALID_INPUT", f"Invalid arguments for capture provider: {provider_id}"
        )

    def _reserve_capture_capacity(
        self,
        total: int,
        *,
        provider_id: str,
        has_oracle: bool,
        limits: RequestLimits,
    ) -> None:
        provider_files = 2 if provider_id == "coverage" else int(provider_id != "direct")
        files_per_capture = 2 + provider_files + (2 if has_oracle else 0)
        bounded_outputs = 1 + int(provider_id != "direct") + int(has_oracle)
        used_bytes, used_files = self._scratch_usage()
        if used_bytes + total * bounded_outputs * limits.max_output_bytes > self.scratch_max_bytes:
            raise RuntimeFailure(
                "LIMIT_EXCEEDED", "Capture would exceed the session scratch ceiling"
            )
        if used_files + total * files_per_capture > self.scratch_max_files:
            raise RuntimeFailure(
                "LIMIT_EXCEEDED", "Capture would exceed the session scratch file ceiling"
            )

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
                    raise RuntimeFailure(exc.code, exc.message) from exc
                cached.preserved = None
            except OSError as exc:
                raise RuntimeFailure("REPOSITORY_IO_FAILURE", str(exc)) from exc
            else:
                return dict(cached.preserved)
        if cached.preserved is None:
            artifacts: list[NativeArtifact] = []
            for source in cached.sources:
                if source.path.is_file():
                    artifacts.append(
                        NativeArtifact(
                            source.path,
                            source.role,
                            source.sha256,
                            source.size_bytes,
                            source.format,
                            source.producer,
                        )
                    )
                    continue
                current_digest, current_size, _ = self._hash_path(source.path)
                if (current_digest, current_size) != (source.sha256, source.size_bytes):
                    raise RuntimeFailure(
                        "MISSING_OR_CHANGED_INPUT",
                        f"Input changed before preservation: {source.path}",
                    )
                for path in self._directory_files(source.path):
                    digest, size = sha256_file(path)
                    relative = path.relative_to(source.path).as_posix()
                    artifacts.append(
                        NativeArtifact(
                            path,
                            f"{source.role}:{relative}",
                            digest,
                            size,
                            source.format,
                            source.producer,
                        )
                    )
            try:
                cached.preserved = self.repository.preserve(
                    manifest_body=cached.manifest_body,
                    artifacts=artifacts,
                    analysis=self._durable_analysis(cached.result),
                )
            except RepositoryError as exc:
                raise RuntimeFailure(exc.code, exc.message) from exc
            except OSError as exc:
                raise RuntimeFailure("REPOSITORY_IO_FAILURE", str(exc)) from exc
        return dict(cached.preserved)

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
            raise RuntimeFailure(exc.code, exc.message) from exc
        except OSError as exc:
            raise RuntimeFailure("REPOSITORY_IO_FAILURE", str(exc)) from exc

    def read_evidence(self, evidence_id: str) -> dict[str, Any]:
        try:
            return self.repository.read(evidence_id)
        except RepositoryError as exc:
            raise RuntimeFailure(exc.code, exc.message) from exc
        except OSError as exc:
            raise RuntimeFailure("REPOSITORY_IO_FAILURE", str(exc)) from exc

    def read_evidence_agent_projection(self, evidence_id: str) -> dict[str, Any]:
        try:
            return self.repository.read_agent_projection(evidence_id)
        except RepositoryError as exc:
            raise RuntimeFailure(exc.code, exc.message) from exc
        except OSError as exc:
            raise RuntimeFailure("REPOSITORY_IO_FAILURE", str(exc)) from exc

    def _resolve_sources(
        self, sources: Sequence[Source], limits: RequestLimits
    ) -> list[ResolvedSource]:
        if not 1 <= len(sources) <= MAX_INPUTS:
            raise RuntimeFailure("INVALID_INPUT", "sources must contain 1 to 32 entries")
        result: list[ResolvedSource] = []
        total_size = 0
        total_files = 0
        for source_index, source in enumerate(sources):
            if isinstance(source, EvidenceSource):
                resolved_evidence = self._resolve_evidence_source(source)
                total_size += resolved_evidence.size_bytes
                total_files += (
                    1
                    if resolved_evidence.path.is_file()
                    else len(self._directory_files(resolved_evidence.path))
                )
                if total_size > limits.max_input_bytes:
                    raise RuntimeFailure("LIMIT_EXCEEDED", "Evidence input exceeds max_input_bytes")
                if total_files > limits.max_input_files:
                    raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_files")
                result.append(resolved_evidence)
                continue
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
            digest, size, file_count = self._hash_path(
                path,
                max_bytes=limits.max_input_bytes - total_size,
                max_files=limits.max_input_files - total_files,
            )
            total_size += size
            total_files += file_count
            if source.expected_sha256 and digest != source.expected_sha256:
                raise RuntimeFailure("MISSING_OR_CHANGED_INPUT", f"SHA-256 mismatch: {source.path}")
            result.append(
                ResolvedSource(
                    path,
                    digest,
                    size,
                    source.format or self._sniff(path),
                    source.producer,
                    "input" if len(sources) == 1 else f"input-{source_index + 1:04d}",
                )
            )
        return result

    def _resolve_evidence_source(self, source: EvidenceSource) -> ResolvedSource:
        manifest = self.read_evidence(source.evidence_id)
        artifacts = manifest["body"].get("artifacts", [])
        if not isinstance(artifacts, list):
            raise RuntimeFailure("REPOSITORY_CORRUPTION", "Evidence artifacts are invalid")
        role = source.artifact_role or self._default_evidence_role(artifacts)
        selected = [
            item
            for item in artifacts
            if isinstance(item, dict)
            and (item.get("role") == role or str(item.get("role", "")).startswith(role + ":"))
        ]
        if not selected:
            raise RuntimeFailure(
                "MISSING_EVIDENCE", "The requested evidence artifact role is absent"
            )
        if len(selected) > 1 or any(
            str(item.get("role", "")).startswith(role + ":") for item in selected
        ):
            return self._materialize_evidence_bundle(source.evidence_id, role, selected)
        artifact = selected[0]
        digest = str(artifact["sha256"])
        path = self.repository.root / "artifacts" / "sha256" / digest[:2] / digest / "payload"
        return ResolvedSource(
            path,
            digest,
            int(artifact["size_bytes"]),
            str(artifact["format"]),
            str(artifact["producer"]) if artifact.get("producer") is not None else None,
            str(artifact.get("role", "input")),
        )

    @staticmethod
    def _default_evidence_role(artifacts: list[Any]) -> str:
        roots = {
            str(item["role"]).split(":", 1)[0]
            for item in artifacts
            if isinstance(item, dict)
            and isinstance(item.get("role"), str)
            and not str(item["role"]).endswith(
                ("/stdout", "/stderr", "/oracle_stdout", "/oracle_stderr")
            )
        }
        if len(roots) != 1:
            raise RuntimeFailure(
                "INVALID_INPUT",
                "artifact_role is required when evidence contains multiple logical sources",
            )
        return roots.pop()

    def _materialize_evidence_bundle(
        self, evidence_id: str, role: str, artifacts: list[dict[str, Any]]
    ) -> ResolvedSource:
        bundle_key = hashlib.sha256(f"{evidence_id}:{role}".encode()).hexdigest()
        destination = self.scratch / "evidence-sources" / bundle_key
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(
                tempfile.mkdtemp(prefix=f"{destination.name}.partial-", dir=destination.parent)
            )
            try:
                for artifact in artifacts:
                    artifact_role = str(artifact["role"])
                    if not artifact_role.startswith(role + ":"):
                        raise RuntimeFailure(
                            "REPOSITORY_CORRUPTION", "Evidence bundle member role is invalid"
                        )
                    relative = Path(artifact_role[len(role) + 1 :])
                    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                        raise RuntimeFailure(
                            "REPOSITORY_CORRUPTION", "Evidence bundle member path is invalid"
                        )
                    digest = str(artifact["sha256"])
                    payload = (
                        self.repository.root
                        / "artifacts"
                        / "sha256"
                        / digest[:2]
                        / digest
                        / "payload"
                    )
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(payload, target)
                try:
                    stage.rename(destination)
                except OSError as error:
                    if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        digest, size, _file_count = self._hash_path(destination)
        formats = {str(item["format"]) for item in artifacts}
        producers = {
            str(item["producer"]) if item.get("producer") is not None else None
            for item in artifacts
        }
        if len(formats) != 1 or len(producers) != 1:
            raise RuntimeFailure(
                "REPOSITORY_CORRUPTION", "Evidence bundle metadata is inconsistent"
            )
        return ResolvedSource(destination, digest, size, formats.pop(), producers.pop(), role)

    def _read_rows(
        self, sources: list[ResolvedSource], offset: int, limit: int
    ) -> tuple[list[dict[str, Any]], int, bool]:
        rows: list[dict[str, Any]] = []
        observed = 0
        for source in sources:
            for row in self._iter_rows(source.path, source.format):
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
        sources: list[ResolvedSource],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        limits: RequestLimits,
    ) -> ProviderAnalysis | None:
        try:
            if (
                capability_id == "cpu.hotspots"
                and len(sources) == 1
                and (
                    cpu_profile := self.cpu_profiles.analyze(
                        *(
                            self._perf_collapsed(sources[0], limits)
                            if sources[0].format == "perf-data"
                            else (sources[0].path, sources[0].format)
                        ),
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
                in {"failures.summary", "coverage.summary", "static.performance_candidates"}
            ):
                return self.reliability.analyze(
                    sources[0].path, sources[0].format, max_rows=max_rows
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
        sources: list[ResolvedSource],
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

    def _perf_collapsed(self, source: ResolvedSource, limits: RequestLimits) -> tuple[Path, str]:
        binding = ExecutableResolver().require_host_tool(
            "perf", cwd=self.project_root, environment=dict(os.environ)
        )
        provider_version = binding.identity.sha256
        key = (source.sha256, f"perf-script:{provider_version}")
        cached = self.conversions.get(key)
        if cached is not None and cached.is_file():
            return cached, "perf"
        conversion_root = self.scratch / "conversions"
        conversion_root.mkdir(exist_ok=True)
        output = conversion_root / f"perf-{source.sha256[:20]}-{provider_version[:12]}.folded"
        request = ExecutionRequest(
            argv=(str(binding.invocation_path), "script", "--input", str(source.path)),
            executable_binding=binding,
            cwd=self.project_root,
            environment_allowlist=("PATH",),
            allowed_working_roots=(self.project_root,),
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
        self._write_collapsed_perf_script(outcome.stdout, output)
        self.conversions[key] = output
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
                fields = raw_line.strip().split()
                if len(fields) >= 2:
                    symbol = fields[1].split("+0x", 1)[0]
                    if symbol and symbol != "[unknown]":
                        frames.append(symbol.replace(";", ":"))
        except UnicodeDecodeError as error:
            raise RuntimeFailure("DECODE_FAILURE", "perf script output is not UTF-8") from error
        finish()
        if not stacks:
            raise RuntimeFailure("DECODE_FAILURE", "perf script returned no stack samples")
        output.write_text(
            "".join(f"{';'.join(stack)} {count}\n" for stack, count in sorted(stacks.items()))
        )

    def _nsys_parquetdir(self, source: ResolvedSource, limits: RequestLimits) -> tuple[Path, str]:
        binding = ExecutableResolver().require_host_tool(
            "nsys", cwd=self.project_root, environment=dict(os.environ)
        )
        provider_version = binding.identity.sha256
        key = (source.sha256, provider_version)
        cached = self.conversions.get(key)
        if cached is not None and cached.is_dir():
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
            cwd=self.project_root,
            environment_allowlist=("PATH",),
            allowed_working_roots=(self.project_root,),
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
        self.conversions[key] = exported
        return exported, provider_version

    def _xctrace_toc(self, source: ResolvedSource, limits: RequestLimits) -> tuple[Path, str]:
        binding = ExecutableResolver().require_host_tool(
            "xcrun", cwd=self.project_root, environment=dict(os.environ)
        )
        provider_version = binding.identity.sha256
        key = (source.sha256, f"xctrace-toc:{provider_version}")
        cached = self.conversions.get(key)
        if cached is not None and cached.is_file():
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
            cwd=self.project_root,
            environment_allowlist=("PATH",),
            allowed_working_roots=(self.project_root,),
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
        self.conversions[key] = output
        return output, provider_version

    def _resource_policy(self, limits: RequestLimits, *, writable_root: Path) -> ResourcePolicy:
        _used_bytes, used_files = self._scratch_usage()
        remaining_files = self.scratch_max_files - used_files - 4
        if remaining_files < 1:
            raise RuntimeFailure(
                "LIMIT_EXCEEDED", "Capture would exceed the session scratch file ceiling"
            )
        return ResourcePolicy(
            filesystem_path=self.scratch,
            writable_roots=(writable_root,),
            minimum_free_bytes=64 * 1024 * 1024,
            maximum_rss_bytes=limits.max_memory_bytes,
            max_observed_files=min(limits.max_input_files, remaining_files),
            maximum_writable_growth_bytes=limits.max_output_bytes,
        )

    def _iter_rows(self, path: Path, format_name: str) -> Iterator[dict[str, Any]]:
        if path.is_dir():
            for item in self._directory_files(path):
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
            with path.open("rb") as stream:
                prefix = stream.read(1)
                stream.seek(0)
                if prefix == b"[":
                    for value in ijson.items(stream, "item", use_float=True):
                        yield value if isinstance(value, dict) else {"value": value}
                    return
            yield from self._iter_json_object(path)
        else:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    yield {"line": number, "text": line.rstrip("\n")}

    @staticmethod
    def _iter_json_object(path: Path) -> Iterator[dict[str, Any]]:
        entries: list[tuple[str, str, Any]] = []
        current_key: str | None = None
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream, use_float=True):
                if prefix == "" and event == "map_key":
                    current_key = str(value)
                elif current_key is not None and prefix == current_key:
                    key = current_key
                    if event == "start_array":
                        entries.append(("array", key, None))
                        current_key = None
                    if event in {"string", "number", "boolean", "null"}:
                        entries.append(("scalar", key, value))
                        current_key = None
                    elif event == "start_map":
                        entries.append(("object", key, None))
                        current_key = None
        if any(kind == "array" for kind, _key, _value in entries):
            entries = [entry for entry in entries if entry[0] != "object"]
        for kind, key, value in entries:
            if kind == "array":
                with path.open("rb") as stream:
                    for item in ijson.items(stream, f"{key}.item", use_float=True):
                        yield (
                            {"section": key, **item}
                            if isinstance(item, dict)
                            else {"section": key, "value": item}
                        )
            elif kind == "object":
                yield {"key": key, "value_type": "object"}
            else:
                yield {"key": key, "value": value}

    def _shrink_result(
        self, result: dict[str, Any], limit: int, identity: Mapping[str, Any], offset: int
    ) -> None:
        rows = result["blocks"][1]["rows"]
        while len(canonical_bytes(result)) > limit and rows:
            rows.pop()
            result["coverage"].update(rows_returned=len(rows), complete=False)
            result["truncation"] = {"reason": "result_bytes", "next_offset": offset + len(rows)}
            result["continuation"] = self._encode_continuation(identity, offset + len(rows))
        if len(canonical_bytes(result)) > limit:
            raise RuntimeFailure("LIMIT_EXCEEDED", "Result metadata exceeds max_result_bytes")

    def _bound_capture_result(
        self, result: dict[str, Any], limit: int, identity: Mapping[str, Any], offset: int
    ) -> None:
        if len(canonical_bytes(result)) <= limit:
            return
        capture = cast(dict[str, Any], result["capture"])
        executions = cast(list[dict[str, Any]], capture["executions"])
        compact = [
            {
                "case": item["case"],
                "block": item["block"],
                "returncode": item["returncode"],
                "status": item["status"],
                "failure_code": item["failure_code"],
                "wall_time_ns": item["wall_time_ns"],
                "containment": item["containment"],
                "semantic_oracle": (
                    {
                        "status": item["semantic_oracle"]["status"],
                        "returncode": item["semantic_oracle"]["returncode"],
                        "failure_code": item["semantic_oracle"]["failure_code"],
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
                    "decision": decision,
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
            "request": hashlib.sha256(canonical_bytes(identity)).hexdigest(),
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
            if (
                not secrets.compare_digest(value["checksum"], expected)
                or payload["request"] != hashlib.sha256(canonical_bytes(identity)).hexdigest()
            ):
                raise ValueError
            return int(payload["offset"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeFailure(
                "INVALID_INPUT", "Continuation does not match this request and its inputs"
            ) from exc

    def _sniff_source(self, source: Source) -> dict[str, Any]:
        if isinstance(source, EvidenceSource):
            manifest = self.read_evidence(source.evidence_id)
            artifacts = manifest["body"].get("artifacts", [])
            if not isinstance(artifacts, list):
                raise RuntimeFailure("REPOSITORY_CORRUPTION", "Evidence artifacts are invalid")
            role = source.artifact_role or self._default_evidence_role(artifacts)
            selected = [
                item
                for item in artifacts
                if isinstance(item, dict)
                and (item.get("role") == role or str(item.get("role", "")).startswith(role + ":"))
            ]
            formats = {str(item.get("format")) for item in selected}
            if len(formats) != 1:
                raise RuntimeFailure(
                    "REPOSITORY_CORRUPTION",
                    "Evidence bundle metadata is inconsistent",
                )
            return {"kind": "evidence", "evidence_id": source.evidence_id, "format": formats.pop()}
        return {
            "kind": "path",
            "path": source.path,
            "format": source.format or self._sniff(Path(source.path)),
        }

    @staticmethod
    def _sniff(path: Path) -> str:
        known = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".parquet": "parquet",
            ".cpuprofile": "cpuprofile",
            ".pftrace": "perfetto",
            ".perfetto-trace": "perfetto",
            ".trace": "xctrace",
            ".bin": "memray",
            ".nsys-rep": "nsys-rep",
            ".ncu-rep": "nsight-compute",
            ".ncu-repz": "nsight-compute",
            ".sarif": "sarif",
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
        return (
            "coverage"
            if path.name == ".coverage" and header.startswith(b"SQLite format 3")
            else "parquet"
            if header.startswith(b"PAR1")
            else "json"
            if header.startswith((b"{", b"["))
            else "text"
        )

    def _availability(self, capability: Capability, formats: set[str]) -> dict[str, Any]:
        source_issues: list[str] = []
        if formats and not formats.intersection(capability.formats):
            source_issues.append("No supplied artifact matches a declared format.")
        selected_formats = (
            formats.intersection(capability.formats) if formats else set(capability.formats)
        )
        probes = [probe for probe in FORMAT_PROBES if selected_formats.intersection(probe.formats)]
        providers = [self._probe_provider(probe) for probe in probes]
        provider_issues = [
            limitation
            for provider in providers
            if not provider["available"]
            for limitation in provider["limitations"]
        ]
        remediation = list(
            dict.fromkeys(item for provider in providers for item in provider["remediation"])
        )
        return {
            "available": not source_issues and any(provider["available"] for provider in providers),
            "providers": providers,
            "limitations": source_issues + provider_issues or [capability.limitation],
            "remediation": remediation,
        }

    @staticmethod
    def _probe_provider(probe: ProviderProbe) -> dict[str, Any]:
        limitations: list[str] = []
        remediation: list[str] = []
        wrong_platform = bool(probe.platforms and sys.platform not in probe.platforms)
        if wrong_platform:
            limitations.append(f"Provider is not supported on {sys.platform}.")

        missing_package: str | None = None
        version: str | None = None
        unsupported_version = False
        if probe.module is not None:
            try:
                installed = importlib.util.find_spec(probe.module) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                installed = False
            if not installed:
                missing_package = probe.distribution or probe.module
                limitations.append(f"Optional package is not installed: {missing_package}")
                if probe.setup_provider is not None:
                    remediation.append(
                        f"Run flameox setup --provider {probe.setup_provider}, then restart "
                        "the server."
                    )
                else:
                    remediation.append(
                        f"Install {missing_package} outside Flameox, then restart the server."
                    )
            elif probe.distribution is not None:
                try:
                    version = importlib.metadata.version(probe.distribution)
                except importlib.metadata.PackageNotFoundError:
                    missing_package = probe.distribution
                    limitations.append(f"Package metadata is unavailable for: {probe.distribution}")
                if version is not None and probe.supported_versions is not None:
                    try:
                        unsupported_version = Version(version) not in SpecifierSet(
                            probe.supported_versions
                        )
                    except InvalidVersion:
                        unsupported_version = True
                    if unsupported_version:
                        limitations.append(
                            f"Provider version {version} is outside {probe.supported_versions}."
                        )

        executable_path: str | None = None
        if probe.configured_path_environment is not None:
            configured = os.environ.get(probe.configured_path_environment)
            if configured and Path(configured).is_file():
                executable_path = configured
        if executable_path is None and probe.executable is not None:
            executable_path = shutil.which(probe.executable)
        missing_executable = (
            probe.executable if probe.executable is not None and executable_path is None else None
        )
        permission_limited = bool(
            executable_path is not None and not os.access(executable_path, os.X_OK)
        )
        if probe.id == "perf" and executable_path is not None and not permission_limited:
            permission_limited = _perf_permission_limited(executable_path)
        if missing_executable is not None:
            limitations.append(f"Required executable is not on PATH: {missing_executable}")
            remediation.append(
                f"Install {missing_executable} with the system/vendor package manager, "
                "then restart Flameox."
            )
        if permission_limited:
            assert executable_path is not None
            limitation, repair = _permission_guidance(probe.id, executable_path)
            limitations.append(limitation)
            remediation.append(repair)

        missing_resource: str | None = None
        if (
            probe.id == "nsight-compute"
            and not wrong_platform
            and executable_path is not None
            and find_report_interface(Path(executable_path)) is None
        ):
            missing_resource = "ncu_report.py"
            limitations.append("The official Nsight Compute ncu_report.py interface is missing.")
            remediation.append(
                "Install Nsight Compute with its extras/python interface outside Flameox, "
                "then restart the server."
            )

        unavailable = bool(
            wrong_platform
            or missing_package
            or unsupported_version
            or missing_executable
            or permission_limited
            or missing_resource
        )
        return {
            "id": probe.id,
            "available": not unavailable,
            "version": version
            or (__version__ if probe.module is None and probe.executable is None else None),
            "formats": list(probe.formats),
            "platforms": list(probe.platforms),
            "missing_package": missing_package,
            "missing_executable": missing_executable,
            "missing_resource": missing_resource,
            "permission_limited": permission_limited,
            "unsupported_version": unsupported_version,
            "limitations": limitations,
            "remediation": remediation,
        }

    def _resolve_project_cwd(self, value: str) -> Path:
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else self.project_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.project_root)
        except (OSError, ValueError) as exc:
            raise RuntimeFailure(
                "INVALID_INPUT", "cwd must be an existing project-contained directory"
            ) from exc
        if not resolved.is_dir():
            raise RuntimeFailure("INVALID_INPUT", "cwd must be a directory")
        return resolved

    @staticmethod
    def _hash_path(
        path: Path,
        *,
        max_bytes: int | None = None,
        max_files: int | None = None,
    ) -> tuple[str, int, int]:
        if not path.is_file() and not path.is_dir():
            raise RuntimeFailure(
                "INVALID_INPUT", f"Source is not a regular file or directory: {path}"
            )
        if path.is_file():
            digest, size = sha256_file(path)
            if max_bytes is not None and size > max_bytes:
                raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_bytes")
            if max_files is not None and max_files < 1:
                raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_files")
            return digest, size, 1
        directory_digest, size = hashlib.sha256(), 0
        files = AnalysisRuntime._directory_files(path, max_files=max_files)
        for item in files:
            item_digest, item_size = sha256_file(item)
            if max_bytes is not None and size + item_size > max_bytes:
                raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_bytes")
            directory_digest.update(
                item.relative_to(path).as_posix().encode() + bytes.fromhex(item_digest)
            )
            size += item_size
        return directory_digest.hexdigest(), size, len(files)

    @staticmethod
    def _directory_files(path: Path, *, max_files: int | None = None) -> list[Path]:
        files: list[Path] = []
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise RuntimeFailure(
                    "INVALID_INPUT", f"Directory sources cannot contain symlinks: {item}"
                )
            if not item.is_file() and not item.is_dir():
                raise RuntimeFailure(
                    "INVALID_INPUT", f"Directory sources cannot contain special files: {item}"
                )
            if item.is_file():
                files.append(item)
                if max_files is not None and len(files) > max_files:
                    raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_files")
        return files

    def _scratch_usage(self) -> tuple[int, int]:
        files = [item for item in self.scratch.rglob("*") if item.is_file()]
        return sum(item.stat().st_size for item in files), len(files)

    def _assert_scratch_capacity(self) -> None:
        used_bytes, used_files = self._scratch_usage()
        if used_bytes > self.scratch_max_bytes or used_files > self.scratch_max_files:
            raise RuntimeFailure("LIMIT_EXCEEDED", "Capture exceeded the session scratch ceiling")
