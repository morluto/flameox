"""Low-level MCP transport over the process-lifespan Flameox runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import anyio
from mcp import types
from mcp.server import CacheHint, Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp_types import CallToolResult, ContentBlock, ResourceLink, TextContent
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

import flameox.providers.environment as provider_setup
from flameox import __version__
from flameox.mcp.descriptions import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from flameox.mcp.request_contracts import (
    AnalyzeArguments,
    CaptureArguments,
    InspectCapabilitiesArguments,
    PrepareProvidersArguments,
    PreserveArguments,
    QueryArguments,
    RescueArguments,
)
from flameox.mcp.result_contracts import (
    AdjustRequestAction,
    CallToolAction,
    CapabilityDetail,
    CapabilityGetEnvelope,
    CapabilityListEnvelope,
    CapabilityListRecord,
    OperatorAction,
    PreserveThenAnalyzeAction,
    RecoverableEnvelope,
    ToolFailureEnvelope,
)
from flameox.mcp.tool_registry import (
    bind_tool_contracts,
    capability_descriptor,
    capability_detail,
)
from flameox.mcp.validation import normalize_validation_error
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CAPABILITY_BY_ID,
    CAPTURE_PROVIDER_CONTRACTS,
    CaptureTarget,
    RequestLimits,
    RuntimeFailure,
    Source,
    compatible_capture_providers,
)


def _text_result(
    value: Mapping[str, Any],
    *,
    summary: str,
    is_error: bool = False,
    resource: ResourceLink | None = None,
) -> CallToolResult:
    content: list[ContentBlock] = [TextContent(type="text", text=summary)]
    if resource is not None:
        content.append(resource)
    return CallToolResult(
        is_error=is_error,
        content=content,
        structured_content=dict(value),
    )


def _failure_result(value: ToolFailureEnvelope) -> CallToolResult:
    field = ""
    if value.field_path:
        field = f" at {'.'.join(str(item) for item in value.field_path)}"
    return _text_result(
        value.model_dump(mode="json"),
        summary=f"{value.code}{field}: {value.message}",
        is_error=True,
    )


def _runtime_failure(error: RuntimeFailure) -> CallToolResult:
    provider_id = error.details.get("provider_id")
    remediation = " ".join(error.remediation) or error.message
    if error.retryable and isinstance(provider_id, str):
        value = RecoverableEnvelope(
            status="retryable",
            code=error.code,
            message=error.message,
            retryable=True,
            next_action=CallToolAction(
                kind="call_tool",
                tool="prepare_providers",
                arguments=cast(dict[str, JsonValue], {"provider_ids": [provider_id]}),
                then_retry="capture_and_analyze",
                message=remediation,
            ),
            details=error.details,
        )
        return _text_result(value.model_dump(mode="json"), summary=error.message)
    if error.code == "UNAVAILABLE_CAPABILITY":
        value = RecoverableEnvelope(
            status="unavailable",
            code=error.code,
            message=error.message,
            retryable=False,
            next_action=OperatorAction(kind="operator_action", message=remediation)
            if error.remediation
            else None,
            details=error.details,
        )
        return _text_result(value.model_dump(mode="json"), summary=error.message)
    accepted: list[str] | None = None
    field_path: list[str | int] | None = None
    for key in (
        "accepted_values",
        "accepted_formats",
        "accepted_provider_ids",
        "available_capabilities",
    ):
        candidate = error.details.get(key)
        if isinstance(candidate, list) and all(isinstance(item, str) for item in candidate):
            accepted = candidate
            break
    source_index = error.details.get("source_index")
    if isinstance(source_index, int) and "accepted_formats" in error.details:
        field_path = ["request", "sources", source_index, "format"]
    elif "accepted_provider_ids" in error.details:
        field_path = ["request", "provider", "kind"]
    next_action: AdjustRequestAction | OperatorAction | None
    if field_path is not None or accepted is not None:
        next_action = AdjustRequestAction(
            kind="adjust_request", field_path=field_path, message=remediation
        )
    elif error.remediation:
        next_action = OperatorAction(kind="operator_action", message=remediation)
    else:
        next_action = None
    return _failure_result(
        ToolFailureEnvelope(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            field_path=field_path,
            accepted_values=accepted,
            next_action=next_action,
            details=error.details,
        )
    )


def _attach_next_page(value: dict[str, Any], request: dict[str, Any] | None) -> None:
    if request is None:
        value["next_page"] = None
        return
    limits = request.pop("limits")
    if "analysis_id" in value and "capability_id" in value:
        value["continuation"] = None
    else:
        value.pop("continuation", None)
    value["next_page"] = {
        "tool": "analyze",
        "arguments": {"request": request, "page_size": limits["max_rows"]},
    }


def _resource_link(value: Mapping[str, Any]) -> ResourceLink | None:
    preserved = value.get("preserved")
    if not isinstance(preserved, Mapping):
        return None
    return ResourceLink(
        type="resource_link",
        uri=str(preserved["uri"]),
        name=f"Evidence {preserved['evidence_id']}",
        mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
    )


def _summary(value: Mapping[str, Any]) -> str:
    status = value.get("status", "complete")
    if "analysis_id" in value:
        truncation = value.get("truncation")
        if value.get("next_page") is not None:
            action = "call analyze with the exact next_page arguments; do not rerun capture"
        elif isinstance(value.get("preserved"), Mapping):
            action = "follow the returned evidence resource"
        elif isinstance(truncation, Mapping) and truncation.get("reason") == "provider_limit":
            action = "narrow the query or recapture; no continuation is available"
        else:
            action = "preserve the session analysis if durable evidence is needed"
        return (
            f"{value.get('capability_id')}: {status}; analysis_id={value['analysis_id']}; "
            f"next: {action}."
        )
    if "evidence" in value:
        return f"Evidence query {status}; {len(cast(list[Any], value['evidence']))} row(s)."
    return f"Flameox operation {status}."


class FlameoxServer(Server[AnalysisRuntime]):
    """SDK low-level server with small direct-inspection conveniences for tests and CLI."""

    def __init__(
        self, *, evidence_directory: Path | None = None, limits: RequestLimits | None = None
    ) -> None:
        self._evidence_directory = evidence_directory
        self._limits = limits
        self._tool_contracts = bind_tool_contracts(self._dispatch)
        self._tool_contract_by_name = {item.name: item for item in self._tool_contracts}
        super().__init__(
            "flameox",
            version=__version__,
            description=SERVER_DESCRIPTION,
            instructions=SERVER_INSTRUCTIONS,
            lifespan=self._lifespan,
            cache_hints={
                "tools/list": CacheHint(ttl_ms=3_600_000, scope="public"),
                "resources/list": CacheHint(ttl_ms=3_600_000, scope="public"),
                "resources/read": CacheHint(ttl_ms=86_400_000, scope="private"),
                "resources/templates/list": CacheHint(ttl_ms=3_600_000, scope="public"),
            },
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
            on_list_resources=self._on_list_resources,
            on_list_resource_templates=self._on_list_resource_templates,
            on_read_resource=self._on_read_resource,
        )

    @asynccontextmanager
    async def _lifespan(self, _: Server[AnalysisRuntime]) -> AsyncIterator[AnalysisRuntime]:
        active = AnalysisRuntime(evidence_directory=self._evidence_directory, limits=self._limits)
        try:
            if active.repository.exists:
                active.repository.cleanup_abandoned_staging()
            yield active
        finally:
            active.close()

    async def list_tools(self) -> list[types.Tool]:
        return [contract.project() for contract in self._tool_contracts]

    async def list_resources(self) -> list[types.Resource]:
        return []

    async def list_resource_templates(self) -> list[types.ResourceTemplate]:
        return [
            types.ResourceTemplate(
                name="immutable-evidence-manifest",
                uri_template="flameox://evidence/{evidence_id}",
                description="Redacted projection of an immutable Flameox evidence manifest.",
                mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
            )
        ]

    async def _on_list_tools(
        self, _ctx: ServerRequestContext[AnalysisRuntime], _params: Any
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=await self.list_tools(), ttl_ms=3_600_000, cache_scope="public"
        )

    async def _on_list_resources(
        self, _ctx: ServerRequestContext[AnalysisRuntime], _params: Any
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=[])

    async def _on_list_resource_templates(
        self, _ctx: ServerRequestContext[AnalysisRuntime], _params: Any
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resource_templates=await self.list_resource_templates()
        )

    async def _on_call_tool(
        self,
        ctx: ServerRequestContext[AnalysisRuntime],
        params: types.CallToolRequestParams,
    ) -> CallToolResult:
        contract = self._tool_contract_by_name.get(params.name)
        if contract is None:
            return _failure_result(
                ToolFailureEnvelope(
                    code="UNKNOWN_TOOL",
                    message=f"Unknown Flameox tool: {params.name}",
                    accepted_values=sorted(self._tool_contract_by_name),
                )
            )
        try:
            request = contract.input_model.model_validate(params.arguments or {})
        except ValidationError as error:
            return _failure_result(normalize_validation_error(error))
        try:
            result = await contract.handler(request, ctx)
            contract.output_model.model_validate(result.structured_content)
            return result
        except ValidationError as error:
            if error.title == contract.output_model.__name__:
                return _failure_result(
                    ToolFailureEnvelope(
                        code="INTERNAL_CONTRACT_FAILURE",
                        message="Flameox produced a result that violated its public contract.",
                    )
                )
            return _failure_result(normalize_validation_error(error))
        except RuntimeFailure as error:
            return _runtime_failure(error)
        except OSError:
            code = "DECODE_FAILURE" if params.name == "analyze" else "EXECUTION_FAILURE"
            message = (
                "Input could not be read during analysis."
                if params.name == "analyze"
                else "Capture failed unexpectedly without trustworthy evidence."
            )
            return _runtime_failure(RuntimeFailure(code, message))
        except (ValueError, json.JSONDecodeError):
            return _runtime_failure(RuntimeFailure("DECODE_FAILURE", "Input could not be decoded."))
        except asyncio.CancelledError:
            raise
        except Exception:
            code = "ANALYSIS_FAILURE" if params.name == "analyze" else "INTERNAL_FAILURE"
            message = (
                "Analysis failed unexpectedly."
                if params.name == "analyze"
                else "Flameox operation failed unexpectedly."
            )
            return _runtime_failure(RuntimeFailure(code, message))

    async def _dispatch(
        self,
        name: str,
        request: BaseModel,
        ctx: ServerRequestContext[AnalysisRuntime],
    ) -> CallToolResult:
        runtime = ctx.lifespan_context
        if name == "inspect_capabilities":
            return self._inspect(cast(InspectCapabilitiesArguments, request))
        if name == "prepare_providers":
            prepare_args = cast(PrepareProvidersArguments, request)
            await ctx.session.report_progress(0.0, message="preparing selected providers")
            try:
                prepared = await runtime.dependencies.prepare(
                    prepare_args.provider_ids, prepare_args.timeout_seconds
                )
            except provider_setup.ProviderSelectionFailure as error:
                raise RuntimeFailure("INVALID_INPUT", str(error)) from error
            except provider_setup.SetupFailure as error:
                raise RuntimeFailure("SETUP_FAILURE", str(error)) from error
            next_action = None
            if prepared.restart_required is not False:
                next_action = {
                    "kind": "reconnect_mcp",
                    "necessity": "required" if prepared.restart_required else "conditional",
                    "launcher": {
                        "command": prepared.launcher_command,
                        "args": prepared.launcher_args,
                    },
                    "message": (
                        "Preserve needed session analyses, then reconnect with the launcher."
                    ),
                }
            value: dict[str, Any] = {
                "status": "complete",
                "requested_providers": prepared.requested_providers,
                "prepared_managed_providers": prepared.prepared_managed_providers,
                "external_requirements": [
                    {"provider_id": item.provider_id, "guidance": item.guidance}
                    for item in prepared.external_requirements
                ],
                "preparation": {"status": prepared.preparation_status},
                "launcher": {"command": prepared.launcher_command, "args": prepared.launcher_args},
                "next_action": next_action,
                "activation_status": prepared.activation_status,
                "workload_requirements": [
                    {"provider_id": item.provider_id, "guidance": item.guidance}
                    for item in prepared.workload_requirements
                ],
            }
            action = "reconnect using the returned launcher" if next_action else "continue capture"
            if prepared.external_requirements:
                action += "; satisfy the listed external requirements; host readiness is unknown"
            return _text_result(
                value,
                summary=f"Provider preparation completed; next: {action}.",
            )
        if name == "analyze":
            analysis_args = cast(AnalyzeArguments, request)
            capability = CAPABILITY_BY_ID[analysis_args.request.capability_id]
            if (
                not capability.minimum_sources
                <= len(analysis_args.request.sources)
                <= capability.maximum_sources
            ):
                raise RuntimeFailure(
                    "INVALID_INPUT",
                    f"{capability.id} requires {capability.minimum_sources} to "
                    f"{capability.maximum_sources} source(s).",
                    details={
                        "minimum_sources": capability.minimum_sources,
                        "maximum_sources": capability.maximum_sources,
                        "actual_sources": len(analysis_args.request.sources),
                    },
                )
            for source_index, source in enumerate(analysis_args.request.sources):
                if (
                    source.kind == "path"
                    and source.format is not None
                    and source.format not in capability.formats
                ):
                    raise RuntimeFailure(
                        "UNSUPPORTED_FORMAT",
                        f"{capability.id} does not accept artifact format {source.format!r}.",
                        details={
                            "source_index": source_index,
                            "accepted_formats": list(capability.formats),
                        },
                    )
            try:
                options = capability.model.model_validate(
                    analysis_args.request.options
                ).model_dump()
            except ValidationError as error:
                return _failure_result(
                    normalize_validation_error(error, prefix=("request", "options"))
                )
            sources = TypeAdapter(list[Source]).validate_python(
                [item.model_dump(mode="python") for item in analysis_args.request.sources]
            )
            await ctx.session.report_progress(0.0, message=f"analyzing {capability.id}")
            value, next_request = await runtime.run_in_request(
                partial(
                    runtime.analyze_page,
                    capability.id,
                    sources,
                    options,
                    limits=RequestLimits(max_rows=analysis_args.page_size),
                    continuation=analysis_args.request.continuation,
                )
            )
            value["status"] = "complete"
            _attach_next_page(value, next_request)
            return _text_result(value, summary=_summary(value))
        if name == "capture_and_analyze":
            capture_args = cast(CaptureArguments, request)
            capability = CAPABILITY_BY_ID[capture_args.request.capability_id]
            compatible = {item.id for item in compatible_capture_providers(capability)}
            if capture_args.request.provider.kind not in compatible:
                raise RuntimeFailure(
                    "INVALID_INPUT",
                    f"Provider {capture_args.request.provider.kind!r} cannot produce artifacts for "
                    f"{capability.id}.",
                    details={"accepted_provider_ids": sorted(compatible)},
                )
            try:
                provider_options = (
                    CAPTURE_PROVIDER_CONTRACTS[capture_args.request.provider.kind]
                    .argument_model.model_validate(capture_args.request.provider.options)
                    .model_dump()
                )
            except ValidationError as error:
                return _failure_result(
                    normalize_validation_error(error, prefix=("request", "provider", "options"))
                )
            try:
                analysis_options = capability.model.model_validate(
                    capture_args.request.options
                ).model_dump()
            except ValidationError as error:
                return _failure_result(
                    normalize_validation_error(error, prefix=("request", "options"))
                )
            target = CaptureTarget(
                **capture_args.request.target.model_dump(),
                provider_id=capture_args.request.provider.kind,
                capture_arguments=provider_options,
                analysis_arguments=analysis_options,
            )

            async def progress(current: int, total: int, message: str) -> None:
                await ctx.session.report_progress(float(current), float(total), message)

            value, next_request = await runtime.capture_analysis_page(
                target,
                capability.id,
                experiment=capture_args.request.experiment,
                limits=RequestLimits(max_rows=capture_args.page_size),
                progress=progress,
                preserve=capture_args.request.preserve,
            )
            _attach_next_page(value, next_request)
            capture = cast(dict[str, Any], value["capture"])
            outcome = cast(dict[str, Any], capture["outcome"])
            capture["status"] = "complete"
            capture["workload_status"] = outcome["status"]
            value["status"] = (
                "partial"
                if outcome["status"] != "succeeded" or value.get("analysis_failure") is not None
                else "complete"
            )
            value["next_action"] = None
            if value["status"] == "partial":
                value["next_action"] = PreserveThenAnalyzeAction(
                    kind="preserve_then_analyze",
                    preserve_arguments=cast(
                        dict[str, JsonValue], {"analysis_id": value["analysis_id"]}
                    ),
                    message=(
                        "Preserve this trustworthy capture, then retry analyze over its "
                        "analysis sources after addressing the observed failure."
                    ),
                ).model_dump(mode="json")
            link = _resource_link(value)
            return _text_result(value, summary=_summary(value), resource=link)
        if name == "preserve_evidence":
            preserve_args = cast(PreserveArguments, request)
            value, next_request = await runtime.run_in_request(
                partial(runtime.preserve_evidence_page, preserve_args.analysis_id)
            )
            value["status"] = "complete"
            _attach_next_page(value, next_request)
            link = ResourceLink(
                type="resource_link",
                uri=str(value["uri"]),
                name=f"Evidence {value['evidence_id']}",
                mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
            )
            action = (
                "use this refreshed evidence-backed next_page"
                if value.get("next_page") is not None
                else "follow the returned evidence resource"
            )
            return _text_result(
                value,
                summary=(f"Evidence {value['evidence_id']} preserved; next: {action}."),
                resource=link,
            )
        if name == "rescue_evidence":
            rescue_args = cast(RescueArguments, request)
            value, next_request = await runtime.run_in_request(
                partial(
                    runtime.rescue_evidence_page,
                    rescue_args.analysis_id,
                    rescue_args.destination,
                )
            )
            value["status"] = "complete"
            _attach_next_page(value, next_request)
            return _text_result(
                value,
                summary=(
                    f"Evidence {value['evidence_id']} rescued; next: restart or reconnect "
                    "with the returned FLAMEOX_DATA_DIR."
                ),
            )
        if name == "query_evidence":
            query_args = cast(QueryArguments, request)
            value = await runtime.run_in_request(
                partial(
                    runtime.query_evidence,
                    evidence_kind=query_args.evidence_kind,
                    capability_id=query_args.capability_id,
                    provider_id=query_args.provider_id,
                    input_sha256=query_args.input_sha256,
                    created_after=query_args.created_after,
                    created_before=query_args.created_before,
                    limit=query_args.page_size,
                    cursor=query_args.cursor,
                )
            )
            continuation = value.pop("continuation", None)
            value["status"] = "complete"
            value["next_page"] = None
            if continuation:
                next_arguments = query_args.model_dump(mode="json")
                next_arguments["cursor"] = continuation
                value["next_page"] = {"tool": "query_evidence", "arguments": next_arguments}
            return _text_result(value, summary=_summary(value))
        raise RuntimeFailure("UNKNOWN_TOOL", f"Unknown Flameox tool: {name}")

    @staticmethod
    def _inspect(args: InspectCapabilitiesArguments) -> CallToolResult:
        selected = args.root
        if selected.mode == "get":
            detail = CapabilityDetail.model_validate(
                capability_detail(CAPABILITY_BY_ID[selected.capability_id])
            )
            value = CapabilityGetEnvelope(mode="get", capabilities=[detail]).model_dump(
                mode="json"
            )
            return _text_result(value, summary="Found 1 matching capability contract.")

        capabilities: list[CapabilityListRecord] = []
        for capability in CAPABILITY_BY_ID.values():
            providers = compatible_capture_providers(capability)
            if (
                selected.artifact_format is not None
                and selected.artifact_format not in capability.formats
            ):
                continue
            if (
                selected.capture_supported is not None
                and bool(providers) != selected.capture_supported
            ):
                continue
            capabilities.append(
                CapabilityListRecord.model_validate(capability_descriptor(capability))
            )
        value = CapabilityListEnvelope(mode="list", capabilities=capabilities).model_dump(
            mode="json"
        )
        return _text_result(
            value, summary=f"Found {len(capabilities)} matching capability contract(s)."
        )

    async def _on_read_resource(
        self,
        ctx: ServerRequestContext[AnalysisRuntime],
        params: types.ReadResourceRequestParams,
    ) -> types.ReadResourceResult:
        parsed = urlparse(params.uri)
        evidence_id = parsed.path.removeprefix("/")
        if parsed.scheme != "flameox" or parsed.netloc != "evidence" or not evidence_id:
            raise FileNotFoundError("Unknown Flameox resource URI")
        try:
            manifest = await ctx.lifespan_context.run_in_request(
                partial(ctx.lifespan_context.read_evidence_agent_projection, evidence_id)
            )
        except RuntimeFailure as error:
            raise FileNotFoundError(f"{error.code}: {error.message}") from error
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=params.uri,
                    mime_type=AGENT_EVIDENCE_MEDIA_TYPE,
                    text=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                )
            ]
        )


def create_server(
    *, evidence_directory: Path | None = None, limits: RequestLimits | None = None
) -> FlameoxServer:
    return FlameoxServer(evidence_directory=evidence_directory, limits=limits)


def run_server(*, limits: RequestLimits | None = None) -> None:
    async def serve() -> None:
        server = create_server(limits=limits)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    anyio.run(serve)
