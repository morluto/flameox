"""Thin MCP transport for the process-lifespan Flameox runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, cast

from mcp.server import CacheHint, MCPServer
from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, ContentBlock, ResourceLink, TextContent, ToolAnnotations
from pydantic import Field

import flameox.providers.environment as provider_setup
from flameox import __version__
from flameox.canonical import canonical_bytes
from flameox.mcp.capability_tools import AnalysisRequest, CaptureRequest, ExperimentExecution
from flameox.mcp.result_contracts import (
    AnalysisEnvelope,
    AnalysisOutcome,
    PreparationOutcome,
    PreservationOutcome,
    QueryOutcome,
    RescueOutcome,
)
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    LOWERCASE_SHA256_PATTERN,
    MAX_ROWS,
    CaptureTarget,
    RequestLimits,
    RuntimeFailure,
)

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
CAPTURE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)
PRESERVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
PREPARE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)


def _analysis_summary(value: dict[str, Any], *, resource: ResourceLink | None) -> str:
    coverage = cast(dict[str, Any], value["coverage"])
    state = "complete" if coverage["complete"] is True else "bounded or incomplete"
    truncation = value.get("truncation")
    provider_limited = isinstance(truncation, dict) and truncation.get("reason") == "provider_limit"
    result_limited = isinstance(truncation, dict) and truncation.get("reason") == "result_bytes"
    if isinstance(value.get("next_page"), dict):
        next_action = "call analyze with the exact next_page arguments"
        if value.get("capture") is not None:
            next_action += "; do not rerun capture"
    elif resource is not None or isinstance(value.get("preserved"), dict):
        next_action = "follow the returned evidence resource"
    elif provider_limited:
        next_action = "narrow the semantic query or recapture; no continuation is available"
    elif result_limited:
        next_action = "use a smaller page or simpler options and restart the analysis"
    else:
        next_action = "preserve the session analysis if durable evidence is needed"
    return (
        f"{value['capability_id']}: analysis {state}; analysis_id={value['analysis_id']}; "
        f"next: {next_action}. Full bounded evidence is in structuredContent."
    )


def _preparation_summary(value: dict[str, Any]) -> str:
    next_action = value["next_action"]
    action = "reconnect using the returned launcher" if next_action else "continue capture"
    if isinstance(next_action, dict) and next_action["necessity"] == "conditional":
        action = "verify the active environment; reconnect only if support is absent"
    if value["external_requirements"]:
        action = (
            action + "; " if next_action else ""
        ) + "satisfy the listed external requirements before capture; host readiness is unknown"
    if value["workload_requirements"]:
        action += "; verify the listed requirements in the workload interpreter"
    return f"Provider preparation completed; next: {action}. Details are in structuredContent."


def _evidence_summary(value: dict[str, Any], *, rescued: bool) -> str:
    if rescued:
        action = "restart or reconnect with the returned FLAMEOX_DATA_DIR"
        if value.get("next_page") is not None:
            action += ", then use this refreshed next_page"
        verb = "rescued"
    else:
        action = (
            "use this refreshed evidence-backed next_page"
            if value.get("next_page") is not None
            else "follow the returned evidence resource"
        )
        verb = "preserved"
    return (
        f"Evidence {value['evidence_id']} {verb} with {value['artifact_count']} artifact(s); "
        f"next: {action}. Full details are in structuredContent."
    )


def _query_summary(value: dict[str, Any]) -> str:
    action = "call the exact next_page" if value.get("next_page") else "query complete"
    return (
        f"Found {len(value['evidence'])} evidence record(s); next: {action}. "
        "Full details are in structuredContent."
    )


def _success(
    value: dict[str, Any], *, summary: str, resource: ResourceLink | None = None
) -> CallToolResult:
    content: list[ContentBlock] = [TextContent(type="text", text=summary)]
    if resource is not None:
        content.append(resource)
    return CallToolResult(content=content, structured_content=value)


def _failure(error: RuntimeFailure, *, resource: ResourceLink | None = None) -> CallToolResult:
    detail = {"code": error.code, "message": error.message, "details": error.details}
    content: list[ContentBlock] = [
        TextContent(type="text", text=json.dumps(detail, sort_keys=True))
    ]
    if resource is not None:
        content.append(resource)
    return CallToolResult(
        is_error=True,
        content=content,
        structured_content=detail,
    )


def _attach_next_page(
    value: dict[str, Any],
    request: dict[str, Any] | None,
    *,
    max_result_bytes: int,
) -> None:
    """Project an internal continuation into one bounded, executable MCP call."""

    if request is None:
        return
    limits = request.pop("limits")
    if "continuation" in value:
        value["continuation"] = None
    value["next_page"] = {
        "tool": "analyze",
        "arguments": {"request": request, "page_size": limits["max_rows"]},
    }
    if len(canonical_bytes(value)) <= max_result_bytes:
        return

    # The runtime bounded the evidence before the transport-only handoff existed. If the
    # complete request cannot fit, preserve the evidence bound and report a non-resumable
    # byte truncation instead of returning an oversized or partial invocation.
    value["next_page"] = None
    truncation = value.get("truncation")
    next_offset = truncation.get("next_offset", 0) if isinstance(truncation, dict) else 0
    value["truncation"] = {"reason": "result_bytes", "next_offset": next_offset}


def create_server(
    *,
    evidence_directory: Path | None = None,
    limits: RequestLimits | None = None,
) -> MCPServer[AnalysisRuntime]:
    """Create a process-lifespan runtime over explicit artifacts and targets."""

    def runtime(ctx: Context[Any]) -> AnalysisRuntime:
        return cast(AnalysisRuntime, ctx.request_context.lifespan_context)

    @asynccontextmanager
    async def lifespan(_: MCPServer[AnalysisRuntime]) -> AsyncIterator[AnalysisRuntime]:
        active_runtime = AnalysisRuntime(evidence_directory=evidence_directory, limits=limits)
        try:
            if active_runtime.repository.exists:
                active_runtime.repository.cleanup_abandoned_staging()
            yield active_runtime
        finally:
            active_runtime.close()

    server = MCPServer(
        "flameox",
        version=__version__,
        description="Bounded local runtime evidence without a prerequisite workspace.",
        instructions=(
            "Pass explicit artifact paths or a typed direct target. Analysis and capture are "
            "session-local unless preserve_evidence is called. If a capture reports a missing "
            "Flameox-managed provider, call prepare_providers with the complete desired provider "
            "set and reconnect with its returned launcher. Profiles are exploratory: use "
            "capture_and_analyze with execution.kind=experiment for baseline/candidate cases "
            "measured in "
            "randomized paired blocks with a wall-clock effect and semantic oracle. Use "
            "benchmark.scaling for measurements across a declared numeric input axis, and "
            "the comparison analysis capabilities for compatible artifacts captured separately. "
            "Flameox "
            "never installs host tools, searches parent directories, accepts shell strings, or "
            "creates durable jobs."
        ),
        lifespan=lifespan,
        cache_hints={
            "tools/list": CacheHint(ttl_ms=3_600_000, scope="public"),
            "resources/list": CacheHint(ttl_ms=3_600_000, scope="public"),
            "resources/read": CacheHint(ttl_ms=86_400_000, scope="private"),
            "resources/templates/list": CacheHint(ttl_ms=3_600_000, scope="public"),
        },
    )

    @server.tool(annotations=PREPARE)
    async def prepare_providers(
        ctx: Context[AnalysisRuntime],
        provider_ids: Annotated[
            list[str],
            Field(
                description="Complete desired set of Flameox-managed and host provider IDs.",
                min_length=1,
                max_length=16,
            ),
        ],
        timeout_seconds: Annotated[
            int,
            Field(
                description="Maximum uvx environment preparation time in seconds.",
                ge=1,
                le=provider_setup.MAX_PREPARATION_TIMEOUT_SECONDS,
            ),
        ] = provider_setup.DEFAULT_PREPARATION_TIMEOUT_SECONDS,
    ) -> Annotated[CallToolResult, PreparationOutcome]:
        """Prepare managed providers and return any required MCP reconnection action."""

        try:
            await ctx.report_progress(0.0, message="preparing selected providers")
            preparation = await runtime(ctx).dependencies.prepare(provider_ids, timeout_seconds)
        except provider_setup.ProviderSelectionFailure as error:
            return _failure(RuntimeFailure("INVALID_INPUT", str(error)))
        except provider_setup.SetupFailure as error:
            return _failure(RuntimeFailure("SETUP_FAILURE", str(error)))

        next_action = None
        if preparation.restart_required is not False:
            next_action = {
                "kind": "reconnect_mcp",
                "necessity": "required" if preparation.restart_required else "conditional",
                "message": (
                    "The active server does not satisfy the requested dependency contract. "
                    if preparation.restart_required
                    else "The active server dependency identity could not be verified. "
                )
                + (
                    "Preserve needed session analyses before reconnecting with the complete "
                    "returned launcher. Reconnection ends session-local evidence access."
                ),
            }
        value = {
            "requested_providers": preparation.requested_providers,
            "prepared_managed_providers": preparation.prepared_managed_providers,
            "external_requirements": [
                {
                    "provider_id": requirement.provider_id,
                    "guidance": requirement.guidance,
                }
                for requirement in preparation.external_requirements
            ],
            "preparation": {"status": preparation.preparation_status},
            "launcher": {
                "command": preparation.launcher_command,
                "args": preparation.launcher_args,
            },
            "next_action": next_action,
            "activation_status": preparation.activation_status,
            "workload_requirements": [
                {"provider_id": item.provider_id, "guidance": item.guidance}
                for item in preparation.workload_requirements
            ],
        }
        return _success(value, summary=_preparation_summary(value))

    @server.tool(annotations=READ_ONLY)
    async def analyze(
        request: AnalysisRequest,
        ctx: Context[AnalysisRuntime],
        page_size: Annotated[
            int,
            Field(description="Maximum evidence rows returned on this page.", ge=1, le=MAX_ROWS),
        ] = 100,
    ) -> Annotated[CallToolResult, AnalysisOutcome]:
        """Analyze existing native artifacts with one typed evidence capability."""

        try:
            await ctx.report_progress(0.0, message=f"analyzing {request.capability_id}")
            value, next_request = await runtime(ctx).run_in_request(
                partial(
                    runtime(ctx).analyze_page,
                    request.capability_id,
                    request.sources,
                    request.options.model_dump(),
                    limits=RequestLimits(max_rows=page_size),
                    continuation=request.continuation,
                )
            )
            _attach_next_page(
                value, next_request, max_result_bytes=runtime(ctx).limits.max_result_bytes
            )
            return _success(value, summary=_analysis_summary(value, resource=None))
        except RuntimeFailure as error:
            return _failure(error)
        except OSError:
            return _failure(
                RuntimeFailure("DECODE_FAILURE", "Input could not be read during analysis.")
            )
        except (ValueError, json.JSONDecodeError):
            return _failure(
                RuntimeFailure("DECODE_FAILURE", "Input could not be decoded during analysis.")
            )
        except Exception:
            return _failure(RuntimeFailure("ANALYSIS_FAILURE", "Analysis failed unexpectedly."))

    @server.tool(annotations=CAPTURE)
    async def capture_and_analyze(
        request: CaptureRequest,
        ctx: Context[AnalysisRuntime],
        page_size: Annotated[
            int,
            Field(description="Maximum evidence rows returned on this page.", ge=1, le=MAX_ROWS),
        ] = 100,
    ) -> Annotated[CallToolResult, AnalysisOutcome]:
        """Execute a typed target, capture native artifacts, and analyze them."""

        async def progress(current: int, total: int, message: str) -> None:
            await ctx.report_progress(float(current), float(total), message)

        target = CaptureTarget(
            **request.target.model_dump(),
            provider_id=request.provider.kind,
            capture_arguments=request.provider.options.model_dump(),
            analysis_arguments=request.options.model_dump(),
        )
        experiment = (
            request.execution.design if isinstance(request.execution, ExperimentExecution) else None
        )
        try:
            value, next_request = await runtime(ctx).capture_analysis_page(
                target,
                request.capability_id,
                mode=request.execution.kind,
                experiment=experiment,
                limits=RequestLimits(max_rows=page_size),
                progress=progress,
                preserve=request.preserve,
            )
            _attach_next_page(
                value, next_request, max_result_bytes=runtime(ctx).limits.max_result_bytes
            )
            failed = [
                item for item in value["capture"]["executions"] if item["status"] != "succeeded"
            ]
            preserved = value.get("preserved")
            link = None
            if isinstance(preserved, dict):
                link = ResourceLink(
                    type="resource_link",
                    uri=preserved["uri"],
                    name=f"Evidence {preserved['evidence_id']}",
                    mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
                )
            if value["capture"]["outcome"]["status"] != "succeeded":
                partial_evidence = AnalysisEnvelope.model_validate(value).model_dump(mode="json")
                return _failure(
                    RuntimeFailure(
                        "EXECUTION_FAILURE",
                        "One or more capture executions failed; consult exit attribution "
                        "and preserved diagnostics before inferring workload failure.",
                        details={
                            "partial_evidence": partial_evidence,
                            "failed_executions": failed,
                        },
                    ),
                    resource=link,
                )
            analysis_failure = value.get("analysis_failure")
            if isinstance(analysis_failure, dict):
                partial_evidence = AnalysisEnvelope.model_validate(value).model_dump(mode="json")
                return _failure(
                    RuntimeFailure(
                        str(analysis_failure["code"]),
                        "Capture completed, but requested analysis failed. "
                        "Preserve the returned analysis_id if not already preserved, then "
                        "read the evidence resource and retry "
                        "analyze with its analysis_sources "
                        "after addressing analysis_failure; do not rerun the target.",
                        details={"partial_evidence": partial_evidence},
                    ),
                    resource=link,
                )
            return _success(
                value,
                summary=_analysis_summary(value, resource=link),
                resource=link,
            )
        except RuntimeFailure as error:
            return _failure(error)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _failure(
                RuntimeFailure(
                    "EXECUTION_FAILURE",
                    "Capture failed unexpectedly without trustworthy evidence.",
                )
            )

    @server.tool(annotations=PRESERVE)
    async def preserve_evidence(
        analysis_id: Annotated[
            str,
            Field(description="Session analysis handle returned by an analysis or capture tool."),
        ],
        ctx: Context[AnalysisRuntime],
    ) -> Annotated[CallToolResult, PreservationOutcome]:
        """Idempotently preserve one session analysis and its native artifacts."""
        try:
            value, next_request = await runtime(ctx).run_in_request(
                partial(runtime(ctx).preserve_evidence_page, analysis_id)
            )
            _attach_next_page(
                value, next_request, max_result_bytes=runtime(ctx).limits.max_result_bytes
            )
            link = ResourceLink(
                type="resource_link",
                uri=value["uri"],
                name=f"Evidence {value['evidence_id']}",
                mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
            )
            return _success(
                value,
                summary=_evidence_summary(value, rescued=False),
                resource=link,
            )
        except RuntimeFailure as error:
            return _failure(error)

    @server.tool(annotations=PRESERVE)
    async def rescue_evidence(
        analysis_id: Annotated[
            str,
            Field(description="Live session analysis handle to rescue before reconnecting."),
        ],
        destination: Annotated[
            str,
            Field(
                description=(
                    "Explicit absolute path to a distinct empty evidence directory. The active "
                    "configured repository is not changed."
                ),
                min_length=1,
                max_length=4096,
            ),
        ],
        ctx: Context[AnalysisRuntime],
    ) -> Annotated[CallToolResult, RescueOutcome]:
        """Rescue one live session analysis to a distinct empty store before restart."""
        try:
            value, next_request = await runtime(ctx).run_in_request(
                partial(runtime(ctx).rescue_evidence_page, analysis_id, destination)
            )
            _attach_next_page(
                value, next_request, max_result_bytes=runtime(ctx).limits.max_result_bytes
            )
            return _success(value, summary=_evidence_summary(value, rescued=True))
        except RuntimeFailure as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def query_evidence(
        ctx: Context[AnalysisRuntime],
        evidence_kind: Annotated[
            str | None, Field(description="Exact preserved evidence-kind filter.")
        ] = None,
        capability_id: Annotated[
            str | None, Field(description="Exact capability ID filter.")
        ] = None,
        provider_id: Annotated[str | None, Field(description="Exact provider ID filter.")] = None,
        input_sha256: Annotated[
            str | None,
            Field(
                description="Lowercase SHA-256 digest of a contributing input.",
                pattern=LOWERCASE_SHA256_PATTERN,
            ),
        ] = None,
        created_after: Annotated[
            datetime | None, Field(description="Inclusive lower creation-time bound with timezone.")
        ] = None,
        created_before: Annotated[
            datetime | None, Field(description="Inclusive upper creation-time bound with timezone.")
        ] = None,
        page_size: Annotated[
            int,
            Field(description="Maximum matching manifests returned on this page.", ge=1, le=200),
        ] = 50,
        cursor: Annotated[
            str | None,
            Field(
                description="Opaque cursor from the preceding query page; reuse the same filters."
            ),
        ] = None,
    ) -> Annotated[CallToolResult, QueryOutcome]:
        """Search an immutable, request-pinned manifest inventory in deterministic order."""
        try:
            value = await runtime(ctx).run_in_request(
                partial(
                    runtime(ctx).query_evidence,
                    evidence_kind=evidence_kind,
                    capability_id=capability_id,
                    provider_id=provider_id,
                    input_sha256=input_sha256,
                    created_after=created_after,
                    created_before=created_before,
                    limit=page_size,
                    cursor=cursor,
                )
            )
            if continuation := value.pop("continuation", None):
                value["next_page"] = {
                    "tool": "query_evidence",
                    "arguments": {
                        "evidence_kind": evidence_kind,
                        "capability_id": capability_id,
                        "provider_id": provider_id,
                        "input_sha256": input_sha256,
                        "created_after": created_after,
                        "created_before": created_before,
                        "page_size": page_size,
                        "cursor": continuation,
                    },
                }
            return _success(value, summary=_query_summary(value))
        except RuntimeFailure as error:
            return _failure(error)

    @server.resource(
        "flameox://evidence/{evidence_id}",
        name="immutable-evidence-manifest",
        description="Redacted projection of an immutable Flameox evidence manifest.",
        mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
    )
    async def evidence_manifest(evidence_id: str, ctx: Context) -> str:
        try:
            manifest = await runtime(ctx).run_in_request(
                partial(runtime(ctx).read_evidence_agent_projection, evidence_id)
            )
        except RuntimeFailure as error:
            # Resource handlers raise so missing resources are protocol errors, not content.
            raise FileNotFoundError(f"{error.code}: {error.message}") from error
        return json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    return server


def run_server(*, limits: RequestLimits | None = None) -> None:
    create_server(limits=limits).run()
