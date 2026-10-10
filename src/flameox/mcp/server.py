"""Low-level MCP transport over the process-lifespan Flameox runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, cast

import anyio
from jsonschema.exceptions import ValidationError as SchemaValidationError
from mcp import types
from mcp.server import CacheHint, Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel, JsonValue, ValidationError

import flameox.providers.environment as provider_setup
from flameox import __version__
from flameox.mcp.descriptions import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from flameox.mcp.request_contracts import (
    AnalysisArguments,
    CaptureArguments,
    InspectEvidenceArguments,
    PrepareProvidersArguments,
    PreserveArguments,
    QueryArguments,
    RescueArguments,
)
from flameox.mcp.result_contracts import (
    AdjustRequestAction,
    CallToolAction,
    OperatorAction,
    PreserveThenAnalyzeAction,
    RecoverableEnvelope,
    ToolFailureEnvelope,
    WaitAndRetryAction,
)
from flameox.mcp.tool_registry import tool_contracts
from flameox.mcp.validation import normalize_schema_error, normalize_validation_error
from flameox.repository import RepositoryError
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    OPERATION_BY_CAPTURE_TOOL,
    OPERATION_BY_NAME,
    CaptureTarget,
    RequestLimits,
    RuntimeFailure,
)


def _tool_result(
    value: Mapping[str, Any],
    *,
    is_error: bool = False,
) -> CallToolResult:
    result = dict(value)
    return CallToolResult(
        is_error=is_error,
        content=[TextContent(type="text", text=json.dumps(result, separators=(",", ":")))],
        structured_content=result,
    )


def _failure_result(value: ToolFailureEnvelope) -> CallToolResult:
    return _tool_result(
        value.model_dump(mode="json"),
        is_error=True,
    )


def _failure_details(details: Mapping[str, Any]) -> dict[str, Any]:
    # Native decoder text remains in local diagnostics, outside the MCP contract.
    return {key: value for key, value in details.items() if key != "decoder_stderr"}


def _runtime_failure(
    error: RuntimeFailure,
    *,
    tool: str | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> CallToolResult:
    provider_id = error.details.get("provider_id")
    remediation = " ".join(error.remediation) or error.message
    if error.retryable:
        retry_action: CallToolAction | WaitAndRetryAction
        if isinstance(provider_id, str) and tool in OPERATION_BY_CAPTURE_TOOL:
            retry_action = CallToolAction(
                kind="call_tool",
                tool="prepare_providers",
                arguments=cast(dict[str, JsonValue], {"provider_ids": [provider_id]}),
                then_retry=tool,
                message=remediation,
            )
        else:
            retry_after_ms = error.details.get("retry_after_ms")
            retry_action = WaitAndRetryAction(
                kind="wait_and_retry",
                retry_after_ms=retry_after_ms
                if isinstance(retry_after_ms, int) and retry_after_ms >= 0
                else None,
                message=remediation,
            )
        value = RecoverableEnvelope(
            status="retryable",
            code=error.code,
            message=error.message,
            retryable=True,
            next_action=retry_action,
            details=_failure_details(error.details),
        )
        return _tool_result(value.model_dump(mode="json"))
    if error.code == "UNAVAILABLE_CAPABILITY":
        value = RecoverableEnvelope(
            status="unavailable",
            code=error.code,
            message=error.message,
            retryable=False,
            next_action=OperatorAction(kind="operator_action", message=remediation)
            if error.remediation
            else None,
            details=_failure_details(error.details),
        )
        return _tool_result(value.model_dump(mode="json"))
    accepted: list[str] | None = None
    field_path: list[str | int] | None = None
    for key in (
        "accepted_values",
        "accepted_formats",
        "accepted_provider_ids",
        "available_operations",
    ):
        candidate = error.details.get(key)
        if isinstance(candidate, list) and all(isinstance(item, str) for item in candidate):
            accepted = candidate
            break
    source_index = error.details.get("source_index")
    safe_retry = error.details.get("safe_retry")
    if (
        error.details.get("scope") == "request_limit"
        and isinstance(safe_retry, Mapping)
        and len(safe_retry) == 1
    ):
        safe_field = next(iter(safe_retry))
        if isinstance(safe_field, str):
            field_path = ["limits", safe_field]
            if (
                safe_field == "max_rows"
                and arguments is not None
                and arguments.get("page_size") is not None
            ):
                ceiling = safe_retry[safe_field]
                field_path = ["page_size"]
                adjustment = f"page_size={ceiling}"
                limits = arguments.get("limits")
                if isinstance(limits, Mapping) and safe_field in limits:
                    adjustment += f" and limits.{safe_field}={ceiling}"
                remediation = (
                    f"Retry with {adjustment}. To raise the server ceiling, restart "
                    "or reconnect Flameox with the desired --limits setting."
                )
    elif isinstance(source_index, int) and "accepted_formats" in error.details:
        field_path = ["sources", source_index, "format"]
    elif "accepted_provider_ids" in error.details:
        field_path = ["provider", "kind"]
    elif isinstance(error.details.get("option"), str):
        field_path = [error.details["option"]]
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
            details=_failure_details(error.details),
        )
    )


def _attach_next_page(value: dict[str, Any], request: dict[str, Any] | None) -> None:
    if request is None:
        value["next_page"] = None
        return
    arguments = {
        **request["options"],
        "sources": request["sources"],
        "continuation": request["continuation"],
        "page_size": request["limits"]["max_rows"],
        "limits": request["limits"],
    }
    if "analysis_id" in value and "operation" in value:
        value["continuation"] = None
    else:
        value.pop("continuation", None)
    value["next_page"] = {
        "tool": request["operation"],
        "arguments": arguments,
    }


def _workload_status(executions: list[dict[str, Any]]) -> str:
    workload_returncodes = [
        execution.get("workload_returncode")
        for execution in executions
        if execution.get("returncode_scope") == "workload"
    ]
    if any(item is not None and item != 0 for item in workload_returncodes):
        return "failed"
    if len(workload_returncodes) == len(executions) and all(
        item == 0 for item in workload_returncodes
    ):
        return "succeeded"
    return "unknown"


class FlameoxServer(Server[AnalysisRuntime]):
    """SDK low-level server with small direct-inspection conveniences for tests and CLI."""

    def __init__(
        self, *, evidence_directory: Path | None = None, limits: RequestLimits | None = None
    ) -> None:
        self._evidence_directory = evidence_directory
        self._limits = limits
        self._tool_contracts = tool_contracts()
        self._tool_contract_by_name = {item.name: item for item in self._tool_contracts}
        super().__init__(
            "flameox",
            version=__version__,
            description=SERVER_DESCRIPTION,
            instructions=SERVER_INSTRUCTIONS,
            lifespan=self._lifespan,
            cache_hints={
                "tools/list": CacheHint(ttl_ms=3_600_000, scope="public"),
            },
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    @asynccontextmanager
    async def _lifespan(self, _: Server[AnalysisRuntime]) -> AsyncIterator[AnalysisRuntime]:
        active = AnalysisRuntime(evidence_directory=self._evidence_directory, limits=self._limits)
        try:
            try:
                if active.repository.exists:
                    active.repository.cleanup_abandoned_staging()
            except (RepositoryError, OSError):
                pass  # Optional preservation failures belong to explicit repository operations.
            yield active
        finally:
            active.close()

    async def list_tools(self) -> list[types.Tool]:
        return [contract.project() for contract in self._tool_contracts]

    async def _on_list_tools(
        self, _ctx: ServerRequestContext[AnalysisRuntime], _params: Any
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await self.list_tools())

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
            arguments = params.arguments or {}
            request = contract.input_model.model_validate(arguments)
            contract.input_validator.validate(arguments)
        except ValidationError as error:
            return _failure_result(normalize_validation_error(error))
        except SchemaValidationError as error:
            return _failure_result(normalize_schema_error(error))
        try:
            value = await self._dispatch(params.name, request, ctx)
            try:
                contract.output_model.model_validate(value)
            except ValidationError:
                return _failure_result(
                    ToolFailureEnvelope(
                        code="INTERNAL_CONTRACT_FAILURE",
                        message="Flameox produced a result that violated its public contract.",
                    )
                )
            return _tool_result(value)
        except ValidationError as error:
            return _failure_result(normalize_validation_error(error))
        except RuntimeFailure as error:
            return _runtime_failure(error, tool=params.name, arguments=arguments)
        except OSError:
            if params.name in OPERATION_BY_NAME:
                code = "DECODE_FAILURE"
                message = "Input could not be read during analysis."
            elif params.name in OPERATION_BY_CAPTURE_TOOL:
                code = "EXECUTION_FAILURE"
                message = "Capture failed unexpectedly without trustworthy evidence."
            else:
                code = "IO_FAILURE"
                message = "Flameox could not complete the operation's local file access."
            return _runtime_failure(RuntimeFailure(code, message))
        except ValueError:
            return _runtime_failure(RuntimeFailure("DECODE_FAILURE", "Input could not be decoded."))
        except asyncio.CancelledError:
            raise
        except Exception:
            code = "ANALYSIS_FAILURE" if params.name in OPERATION_BY_NAME else "INTERNAL_FAILURE"
            message = (
                "Analysis failed unexpectedly."
                if params.name in OPERATION_BY_NAME
                else "Flameox operation failed unexpectedly."
            )
            return _runtime_failure(RuntimeFailure(code, message))

    async def _dispatch(
        self,
        name: str,
        request: BaseModel,
        ctx: ServerRequestContext[AnalysisRuntime],
    ) -> dict[str, Any]:
        runtime = ctx.lifespan_context
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
            return value
        if name in OPERATION_BY_NAME:
            analysis_args = cast(AnalysisArguments, request)
            spec = OPERATION_BY_NAME[name]
            options = analysis_args.model_dump(include=set(spec.model.model_fields))
            await ctx.session.report_progress(0.0, message=f"analyzing {spec.name}")
            value, next_request = await runtime.run_in_request(
                partial(
                    runtime.analyze_page,
                    spec.name,
                    analysis_args.sources,
                    options,
                    limits=analysis_args.request_limits(),
                    continuation=analysis_args.continuation,
                )
            )
            value["status"] = "complete"
            _attach_next_page(value, next_request)
            return value
        if name in OPERATION_BY_CAPTURE_TOOL:
            capture_args = cast(CaptureArguments, request)
            spec = OPERATION_BY_CAPTURE_TOOL[name]
            provider_options = capture_args.provider.model_dump()
            provider_id = provider_options.pop("kind")
            analysis_options = capture_args.model_dump(include=set(spec.model.model_fields))
            target = CaptureTarget(
                **capture_args.target.model_dump(),
                provider_id=provider_id,
                capture_arguments=provider_options,
                analysis_arguments=analysis_options,
            )

            async def progress(current: int, total: int, message: str) -> None:
                await ctx.session.report_progress(float(current), float(total), message)

            value, next_request = await runtime.capture_analysis_page(
                target,
                spec.name,
                experiment=getattr(capture_args, "experiment", None),
                limits=capture_args.request_limits(),
                progress=progress,
                preserve=capture_args.preserve,
            )
            _attach_next_page(value, next_request)
            capture = cast(dict[str, Any], value["capture"])
            outcome = cast(dict[str, Any], capture["outcome"])
            capture["status"] = "complete"
            capture["workload_status"] = _workload_status(
                cast(list[dict[str, Any]], capture["executions"])
            )
            value["status"] = (
                "partial"
                if outcome["status"] != "succeeded" or value.get("analysis_failure") is not None
                else "complete"
            )
            value["next_action"] = None
            analysis_failure = value.get("analysis_failure")
            if analysis_failure is not None:
                analysis_failure["details"] = _failure_details(analysis_failure["details"])
            can_reanalyze = (
                analysis_failure is not None
                and analysis_failure["details"]["analysis_source_count"] > 0
            )
            if value["status"] == "partial" and value.get("preserved") is not None:
                if analysis_failure is not None and not can_reanalyze:
                    message = (
                        "Inspect the preserved capture diagnostics. No native analysis inputs "
                        "were retained; address the collector failure before a new capture."
                    )
                elif can_reanalyze:
                    message = (
                        "Inspect this preserved capture and its replay sources. "
                        "Retry analysis after addressing the observed analysis failure; "
                        "reanalysis does not rerun the workload or change its exit outcome."
                    )
                elif capture["workload_status"] == "failed":
                    message = (
                        "Inspect the preserved workload failure. "
                        "Reanalysis does not change its exit outcome."
                    )
                else:
                    message = (
                        "Inspect the preserved collector or experiment failure. "
                        "The failure was not attributed to the workload."
                    )
                value["next_action"] = CallToolAction(
                    kind="call_tool",
                    tool="inspect_evidence",
                    arguments=cast(
                        dict[str, JsonValue],
                        {"evidence_id": value["preserved"]["evidence_id"]},
                    ),
                    then_retry=spec.name if can_reanalyze else None,
                    message=message,
                ).model_dump(mode="json")
            elif can_reanalyze:
                value["next_action"] = PreserveThenAnalyzeAction(
                    kind="preserve_then_analyze",
                    preserve_arguments=cast(
                        dict[str, JsonValue], {"analysis_id": value["analysis_id"]}
                    ),
                    message=(
                        "Preserve this trustworthy capture, then retry "
                        f"{spec.name} over its "
                        "analysis sources after addressing the observed failure."
                    ),
                ).model_dump(mode="json")
            elif value["status"] == "partial":
                workload_status = capture["workload_status"]
                if analysis_failure is not None:
                    message = (
                        "Preserve the capture diagnostics. No native analysis inputs were "
                        "retained; address the collector failure before a new capture."
                    )
                elif workload_status == "failed":
                    message = (
                        "Preserve the observed workload failure for inspection. Reanalysis does "
                        "not change its exit outcome; a new capture can observe target changes."
                    )
                else:
                    message = (
                        "Preserve the capture for inspection. The failure was not attributed "
                        "to the workload; inspect the collector or experiment outcome "
                        "before retrying."
                    )
                value["next_action"] = CallToolAction(
                    kind="call_tool",
                    tool="preserve_evidence",
                    arguments=cast(dict[str, JsonValue], {"analysis_id": value["analysis_id"]}),
                    message=message,
                ).model_dump(mode="json")
            return value
        if name == "preserve_evidence":
            preserve_args = cast(PreserveArguments, request)
            value, next_request = await runtime.run_in_request(
                partial(runtime.preserve_evidence_page, preserve_args.analysis_id)
            )
            value["status"] = "complete"
            _attach_next_page(value, next_request)
            return value
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
            return value
        if name == "inspect_evidence":
            inspect_args = cast(InspectEvidenceArguments, request)
            value = await runtime.run_in_request(
                partial(runtime.read_evidence_agent_projection, inspect_args.evidence_id)
            )
            value["status"] = "complete"
            return value
        if name == "query_evidence":
            query_args = cast(QueryArguments, request)
            value = await runtime.run_in_request(
                partial(
                    runtime.query_evidence,
                    evidence_kind=query_args.evidence_kind,
                    operation=query_args.operation,
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
            return value
        raise RuntimeFailure("UNKNOWN_TOOL", f"Unknown Flameox tool: {name}")


def run_server(*, limits: RequestLimits | None = None) -> None:
    async def serve() -> None:
        server = FlameoxServer(limits=limits)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    anyio.run(serve)
