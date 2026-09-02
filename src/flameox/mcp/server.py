"""Thin MCP transport for the process-lifespan Flameox runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

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
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue

from flameox import __version__
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentDesign,
    RequestLimits,
    RuntimeFailure,
    Source,
)
from flameox.setup import (
    PYTHON_PROVIDER_EXTRAS,
    SYSTEM_PROVIDER_GUIDANCE,
    SetupFailure,
    install_providers,
    mcp_launcher,
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


class DiscoveryEnvelope(_Envelope):
    sniffed_sources: list[dict[str, JsonValue]]
    capabilities: list[dict[str, JsonValue]]


class InspectionEnvelope(_Envelope):
    capabilities: list[dict[str, JsonValue]]


class ExternalRequirementEnvelope(BaseModel):
    provider_id: str
    guidance: str


class InstallationEnvelope(BaseModel):
    status: Literal["installed", "already_configured", "not_applicable"]
    command: list[str]


class LauncherEnvelope(BaseModel):
    command: str
    args: list[str]


class PreparationEnvelope(_Envelope):
    requested_providers: list[str]
    configured_providers: list[str]
    external_requirements: list[ExternalRequirementEnvelope]
    installation: InstallationEnvelope
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


class DiscoveryOutcome(_ObjectOutcome):
    root: DiscoveryEnvelope | ToolFailureEnvelope


class InspectionOutcome(_ObjectOutcome):
    root: InspectionEnvelope | ToolFailureEnvelope


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
            "session-local unless preserve_evidence is called. Use prepare_capabilities only when "
            "discovery reports a missing Flameox-managed provider; reconnect with its returned "
            "launcher after installation. Flameox never installs host tools, searches parent "
            "directories, accepts shell strings, or creates durable jobs."
        ),
        lifespan=lifespan,
    )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    async def discover_capabilities(
        intent: str | None = None,
        sources: list[Source] | None = None,
        include_unavailable: bool = False,
        limit: int = 10,
    ) -> Annotated[CallToolResult, DiscoveryOutcome]:
        """Rank capabilities by intent and bounded source sniffing; never install providers."""
        try:
            return _success(
                runtime().discover_capabilities(
                    intent, sources, include_unavailable=include_unavailable, limit=limit
                )
            )
        except (RuntimeFailure, ValidationError) as error:
            if isinstance(error, ValidationError):
                return _failure(RuntimeFailure("INVALID_INPUT", str(error)))
            return _failure(error)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    async def inspect_capabilities(
        capability_ids: list[str],
    ) -> Annotated[CallToolResult, InspectionOutcome]:
        """Batch-inspect 1-16 capability contracts, providers, limits, and examples."""
        try:
            return _success(runtime().inspect_capabilities(capability_ids))
        except RuntimeFailure as error:
            return _failure(error)

    @server.tool(annotations=PREPARE, structured_output=True)
    async def prepare_capabilities(
        provider_ids: Annotated[list[str], Field(min_length=1, max_length=16)],
    ) -> Annotated[CallToolResult, PreparationOutcome]:
        """Install selected managed providers; report host-tool setup without changing it."""

        requested = list(dict.fromkeys(provider_ids))
        unknown = sorted(
            set(requested).difference(PYTHON_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
        )
        if unknown:
            supported = ", ".join(sorted(PYTHON_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
            return _failure(
                RuntimeFailure(
                    "INVALID_INPUT",
                    f"Unknown provider {unknown[0]!r}; choose one of: {supported}",
                )
            )

        managed = [item for item in requested if item in PYTHON_PROVIDER_EXTRAS]
        external = [
            {"provider_id": item, "guidance": SYSTEM_PROVIDER_GUIDANCE[item]}
            for item in requested
            if item in SYSTEM_PROVIDER_GUIDANCE
        ]
        try:
            installation = await asyncio.to_thread(install_providers, managed)
        except SetupFailure as error:
            return _failure(RuntimeFailure("SETUP_FAILURE", str(error)))

        launcher_command, launcher_args = mcp_launcher(installation.providers)
        installation_status: Literal["installed", "already_configured", "not_applicable"]
        if not managed:
            installation_status = "not_applicable"
        elif installation.changed:
            installation_status = "installed"
        else:
            installation_status = "already_configured"
        return _success(
            {
                "requested_providers": requested,
                "configured_providers": installation.providers,
                "external_requirements": external,
                "installation": {
                    "status": installation_status,
                    "command": installation.command,
                },
                "launcher": {"command": launcher_command, "args": launcher_args},
                "restart_required": installation.changed,
            }
        )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    async def analyze(
        capability_id: str,
        sources: list[Source],
        arguments: dict[str, Any],
        limits: RequestLimits | None = None,
        continuation: str | None = None,
    ) -> Annotated[CallToolResult, AnalysisOutcome]:
        """Analyze 1-32 explicit path or evidence sources without durable writes."""
        try:
            return _success(
                runtime().analyze(
                    capability_id,
                    sources,
                    arguments,
                    limits=limits,
                    continuation=continuation,
                )
            )
        except RuntimeFailure as error:
            return _failure(error)
        except ValidationError as error:
            return _failure(RuntimeFailure("INVALID_INPUT", str(error)))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return _failure(RuntimeFailure("DECODE_FAILURE", str(error)))

    @server.tool(annotations=CAPTURE, structured_output=True)
    async def capture_and_analyze(
        target: CaptureTarget,
        capability_id: str = "artifact.preview",
        mode: Literal["single", "experiment"] = "single",
        experiment: ExperimentDesign | None = None,
        limits: RequestLimits | None = None,
        preserve: bool = False,
        ctx: Context[AnalysisRuntime] | None = None,
    ) -> Annotated[CallToolResult, AnalysisOutcome]:
        """Validate, execute, and analyze one typed target through the bounded broker."""

        async def progress(current: int, total: int, message: str) -> None:
            if ctx is not None:
                await ctx.report_progress(float(current), float(total), message)

        try:
            value = await runtime().capture_and_analyze(
                target,
                capability_id,
                mode=mode,
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
        except ValidationError as error:
            return _failure(RuntimeFailure("INVALID_INPUT", str(error)))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _failure(RuntimeFailure("EXECUTION_FAILURE", str(error) or type(error).__name__))

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
