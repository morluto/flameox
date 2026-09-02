"""Thin MCP transport for the process-lifespan Flameox runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, ContentBlock, ResourceLink, TextContent, ToolAnnotations
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    JsonValue,
    RootModel,
)
from pydantic.json_schema import JsonSchemaValue

import flameox.setup as provider_setup
from flameox import __version__
from flameox.mcp.capability_tools import (
    Execution,
    ExperimentExecution,
    analysis_tool_name,
    capture_provider_type,
    capture_tool_name,
)
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime_contracts import (
    CAPABILITIES,
    Capability,
    CaptureTarget,
    DirectTarget,
    RequestLimits,
    RuntimeFailure,
    Source,
    compatible_capture_providers,
)
from flameox.stateless import AnalysisRuntime

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
CAPTURE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
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


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolFailureEnvelope(_Envelope):
    code: str
    message: str
    details: dict[str, JsonValue]


class CoverageEnvelope(_Envelope):
    rows_returned: int
    rows_observed: int
    complete: bool


class AnalysisEnvelope(_Envelope):
    analysis_id: str
    capability_id: str
    provider: dict[str, JsonValue]
    inputs: list[dict[str, JsonValue]]
    blocks: list[dict[str, JsonValue]]
    coverage: CoverageEnvelope
    truncation: dict[str, JsonValue] | None
    limitations: list[str]
    continuation: str | None


class ExternalRequirementEnvelope(BaseModel):
    provider_id: str
    guidance: str


class PreparationStatusEnvelope(BaseModel):
    status: Literal["prepared", "not_applicable"]


class LauncherEnvelope(BaseModel):
    command: str
    args: list[str]


class PreparationEnvelope(_Envelope):
    requested_providers: list[str]
    prepared_managed_providers: list[str]
    external_requirements: list[ExternalRequirementEnvelope]
    preparation: PreparationStatusEnvelope
    launcher: LauncherEnvelope
    restart_required: bool


class PreservationEnvelope(_Envelope):
    evidence_id: str
    uri: str
    artifact_count: int


class QueryEnvelope(_Envelope):
    evidence: list[dict[str, JsonValue]]
    continuation: str | None
    inventory_digest: str


class _ObjectOutcome(RootModel[Any]):
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema["type"] = "object"
        return schema


class PreparationOutcome(_ObjectOutcome):
    root: PreparationEnvelope | ToolFailureEnvelope


class AnalysisOutcome(_ObjectOutcome):
    root: AnalysisEnvelope | ToolFailureEnvelope


class PreservationOutcome(_ObjectOutcome):
    root: PreservationEnvelope | ToolFailureEnvelope


class QueryOutcome(_ObjectOutcome):
    root: QueryEnvelope | ToolFailureEnvelope


def _success(value: dict[str, Any], *, resource: ResourceLink | None = None) -> CallToolResult:
    content: list[ContentBlock] = [
        TextContent(type="text", text=json.dumps(value, sort_keys=True, separators=(",", ":")))
    ]
    if resource is not None:
        content.append(resource)
    return CallToolResult(content=content, structured_content=value)


def _failure(error: RuntimeFailure) -> CallToolResult:
    detail = {"code": error.code, "message": error.message, "details": error.details}
    return CallToolResult(
        is_error=True,
        content=[TextContent(type="text", text=json.dumps(detail, sort_keys=True))],
        structured_content=detail,
    )


def create_server(
    project_root: Path | None = None,
    *,
    limits: RequestLimits | None = None,
) -> MCPServer[AnalysisRuntime]:
    """Create a server whose fixed project root defaults to the startup cwd."""

    root = (project_root or Path.cwd()).resolve(strict=True)
    active_runtime: AnalysisRuntime | None = None

    def runtime() -> AnalysisRuntime:
        if active_runtime is None:
            raise RuntimeError("The Flameox server lifespan is not active.")
        return active_runtime

    @asynccontextmanager
    async def lifespan(_: MCPServer[AnalysisRuntime]) -> AsyncIterator[AnalysisRuntime]:
        nonlocal active_runtime
        active_runtime = AnalysisRuntime(root, limits=limits)
        try:
            if active_runtime.repository.exists:
                active_runtime.repository.cleanup_abandoned_staging()
            yield active_runtime
        finally:
            active_runtime.close()
            active_runtime = None

    server = MCPServer(
        "flameox",
        version=__version__,
        description="Bounded local runtime evidence without a prerequisite workspace.",
        instructions=(
            "Pass explicit artifact paths or a typed direct target. Analysis and capture are "
            "session-local unless preserve_evidence is called. If a capture reports a missing "
            "Flameox-managed provider, call prepare_providers with the complete desired provider "
            "set and reconnect with its returned launcher. Flameox never installs host tools, "
            "searches parent directories, accepts shell strings, or creates durable jobs."
        ),
        lifespan=lifespan,
    )

    @server.tool(annotations=PREPARE, structured_output=True)
    async def prepare_providers(
        provider_ids: Annotated[list[str], Field(min_length=1, max_length=16)],
        timeout_seconds: Annotated[
            int, Field(ge=1, le=provider_setup.MAX_PREPARATION_TIMEOUT_SECONDS)
        ] = provider_setup.DEFAULT_PREPARATION_TIMEOUT_SECONDS,
    ) -> Annotated[CallToolResult, PreparationOutcome]:
        """Prepare selected managed providers; report host-tool setup without changing it."""

        try:
            preparation = await anyio.to_thread.run_sync(
                provider_setup.prepare_providers, provider_ids, root, timeout_seconds
            )
        except provider_setup.ProviderSelectionFailure as error:
            return _failure(RuntimeFailure("INVALID_INPUT", str(error)))
        except provider_setup.SetupFailure as error:
            return _failure(RuntimeFailure("SETUP_FAILURE", str(error)))

        return _success(
            {
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
                "restart_required": preparation.restart_required,
            }
        )

    def analysis_handler(capability: Capability) -> Callable[..., Awaitable[CallToolResult]]:
        async def handler(
            sources: Annotated[list[Source], Field(min_length=1, max_length=32)],
            options: Any,
            limits: RequestLimits | None = None,
            continuation: str | None = None,
        ) -> Annotated[CallToolResult, AnalysisOutcome]:
            try:
                return _success(
                    runtime().analyze(
                        capability.id,
                        sources,
                        options.model_dump(),
                        limits=limits,
                        continuation=continuation,
                    )
                )
            except RuntimeFailure as error:
                return _failure(error)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                return _failure(RuntimeFailure("DECODE_FAILURE", str(error)))

        # MCP SDK 2.0 derives schemas and diagnostic names from the registered callable.
        handler.__name__ = analysis_tool_name(capability)
        minimum_sources = 2 if capability.id.endswith(".compare") else 1
        handler.__annotations__["sources"] = Annotated[
            list[Source], Field(min_length=minimum_sources, max_length=32)
        ]
        handler.__annotations__["options"] = capability.model
        return handler

    def capture_handler(
        capability: Capability,
        provider_type: Any,
    ) -> Callable[..., Awaitable[CallToolResult]]:
        async def handler(
            target: DirectTarget,
            provider: Any,
            options: Any,
            execution: Execution,
            ctx: Context[AnalysisRuntime],
            limits: RequestLimits | None = None,
            preserve: bool = False,
        ) -> Annotated[CallToolResult, AnalysisOutcome]:
            async def progress(current: int, total: int, message: str) -> None:
                await ctx.report_progress(float(current), float(total), message)

            target = CaptureTarget(
                **target.model_dump(),
                provider_id=provider.kind,
                capture_arguments=provider.options.model_dump(),
                analysis_arguments=options.model_dump(),
            )
            experiment = execution.design if isinstance(execution, ExperimentExecution) else None
            try:
                value = await runtime().capture_and_analyze(
                    target,
                    capability.id,
                    mode=execution.kind,
                    experiment=experiment,
                    limits=limits,
                    progress=progress,
                    preserve=preserve,
                )
                failed = [
                    item for item in value["capture"]["executions"] if item["status"] != "succeeded"
                ]
                if failed:
                    return _failure(
                        RuntimeFailure(
                            "EXECUTION_FAILURE",
                            "One or more captured targets exited unsuccessfully.",
                            details={"partial_evidence": value, "failed_executions": failed},
                        )
                    )
                preserved = value.get("preserved")
                link = None
                if isinstance(preserved, dict):
                    link = ResourceLink(
                        type="resource_link",
                        uri=preserved["uri"],
                        name=f"Evidence {preserved['evidence_id']}",
                        mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
                    )
                return _success(value, resource=link)
            except RuntimeFailure as error:
                return _failure(error)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                return _failure(
                    RuntimeFailure("EXECUTION_FAILURE", str(error) or type(error).__name__)
                )

        handler.__name__ = capture_tool_name(capability)
        handler.__annotations__["provider"] = provider_type
        handler.__annotations__["options"] = capability.model
        return handler

    for capability in CAPABILITIES:
        server.add_tool(
            analysis_handler(capability),
            name=analysis_tool_name(capability),
            title=capability.summary,
            description=(
                f"{capability.summary} Accepts: {', '.join(capability.formats)}. "
                f"Limitation: {capability.limitation}"
            ),
            annotations=READ_ONLY,
            structured_output=True,
        )
        provider_type = capture_provider_type(capability)
        if provider_type is not None:
            providers = compatible_capture_providers(capability)
            server.add_tool(
                capture_handler(capability, provider_type),
                name=capture_tool_name(capability),
                title=f"Capture and {capability.summary.lower()}",
                description=(
                    f"Execute typed argv and {capability.summary.lower()} Compatible providers: "
                    f"{', '.join(provider.id for provider in providers)}."
                ),
                annotations=CAPTURE,
                structured_output=True,
            )

    @server.tool(annotations=PRESERVE, structured_output=True)
    async def preserve_evidence(
        analysis_id: str,
    ) -> Annotated[CallToolResult, PreservationOutcome]:
        """Idempotently preserve one session analysis and its native artifacts."""
        try:
            value = runtime().preserve_evidence(analysis_id)
            link = ResourceLink(
                type="resource_link",
                uri=value["uri"],
                name=f"Evidence {value['evidence_id']}",
                mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
            )
            return _success(value, resource=link)
        except RuntimeFailure as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    async def query_evidence(
        evidence_kind: str | None = None,
        capability_id: str | None = None,
        provider_id: str | None = None,
        input_sha256: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, QueryOutcome]:
        """Search an immutable, request-pinned manifest inventory in deterministic order."""
        try:
            return _success(
                runtime().query_evidence(
                    evidence_kind=evidence_kind,
                    capability_id=capability_id,
                    provider_id=provider_id,
                    input_sha256=input_sha256,
                    created_after=created_after,
                    created_before=created_before,
                    limit=limit,
                    cursor=cursor,
                )
            )
        except RuntimeFailure as error:
            return _failure(error)

    @server.resource(
        "flameox://evidence/{evidence_id}",
        name="immutable-evidence-manifest",
        description="Redacted projection of an immutable Flameox evidence manifest.",
        mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
    )
    async def evidence_manifest(evidence_id: str) -> str:
        try:
            manifest = runtime().read_evidence_agent_projection(evidence_id)
        except RuntimeFailure as error:
            # Resource handlers raise so missing resources are protocol errors, not content.
            raise FileNotFoundError(f"{error.code}: {error.message}") from error
        return json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    return server


def run_server(project_root: Path | None = None, *, limits: RequestLimits | None = None) -> None:
    create_server(project_root, limits=limits).run()
