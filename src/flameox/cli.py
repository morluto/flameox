from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

import anyio
import typer
from mcp import Client
from pydantic import BaseModel, TypeAdapter, ValidationError

from flameox import __version__, setup_ui
from flameox.action_graph import ARTIFACT_PREVIEW_MAX_BYTES, ARTIFACT_PREVIEW_MAX_LINES
from flameox.adapters import (
    AdapterRegistry,
    BenchmarkSamplesExtractor,
    ComputeSanitizerExtractor,
    CoverageExtractor,
    InferenceArtifactExtractor,
    KernelValidationExtractor,
    MemrayExtractor,
    NsightComputeExtractor,
    NsightSystemsExtractor,
    NvbenchExtractor,
    ObservationExtractor,
    PerfettoExtractor,
    PyPerfExtractor,
    PytestExtractor,
    PythonStartupExtractor,
    SetupClient,
    V8CpuProfExtractor,
    V8HeapProfExtractor,
)
from flameox.adapters.memray import memray_extraction_limits
from flameox.analysis import RecipeService
from flameox.application import (
    AnalysisMaterializationService,
    ArtifactPipelineService,
    ArtifactService,
    ArtifactTextPreview,
    CapabilityService,
    CaptureService,
    CompactionService,
    CompareRunSetsRequest,
    ComparisonService,
    ConfigurationOperation,
    ConfigureInferenceScenarioRequest,
    ConfigureInferenceServerRequest,
    CreateInvestigationRequest,
    DrilldownService,
    EvidenceLookupService,
    EvidenceQueryService,
    EvidenceSummaryRequest,
    EvidenceSummaryService,
    ExecutionPolicy,
    ExperimentService,
    FaultExperimentService,
    FindingService,
    FreezeRunIdsRequest,
    GarbageCollector,
    ImportArtifactRequest,
    ImportProfile,
    ImportService,
    InferenceEndpointType,
    InferenceProfilingService,
    InferenceReplayService,
    InferenceScenarioProvider,
    InferenceServerMode,
    InferenceServerProvider,
    IntegrityLevel,
    IntegrityService,
    InvestigationService,
    KernelBuildImportService,
    KernelValidationCompareRequest,
    KernelValidationComparisonService,
    KernelValidationRegistrationService,
    LifecycleEvidenceService,
    MaterializeAnalysisRequest,
    NativeViewerService,
    NvbenchImportService,
    OtlpTraceService,
    PipelineFilter,
    QualifyArtifactImportRequest,
    QuarantineService,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RecoveryService,
    RegisterKernelValidationRequest,
    RegisterPipelineRequest,
    RunDiscoveryService,
    RunFilter,
    RunSetService,
    SetupOperation,
    SetupService,
    SummaryExcerptPolicy,
    SummarySensitiveContextPolicy,
    TraceWindowService,
    WorkloadService,
    XctraceImportRequest,
    XctraceService,
    parse_inference_scenario_config,
    parse_inference_server_config,
    workspace_status,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    EvidenceReferenceType,
    Sensitivity,
)
from flameox.mcp import create_server, run_server
from flameox.storage import RunStore, Workspace

_MATERIALIZE_ANALYSIS_REQUEST_ADAPTER: TypeAdapter[MaterializeAnalysisRequest] = TypeAdapter(
    MaterializeAnalysisRequest
)

app = typer.Typer(
    name="flameox",
    help="Collect and query local runtime evidence.",
    no_args_is_help=True,
)
catalog_app = typer.Typer(help="Validate and rebuild the analytical catalog.")
runs_app = typer.Typer(help="List and inspect execution and import runs.")
extract_app = typer.Typer(help="Extract normalized evidence from native artifacts.")
mcp_app = typer.Typer(help="Run or inspect the local MCP adapter.")
capture_app = typer.Typer(help="Plan and execute named profiler captures.")
workload_app = typer.Typer(help="Create, inspect, and execute named workloads.")
investigations_app = typer.Typer(help="Create and inspect diagnostic investigations.")
hypotheses_app = typer.Typer(help="Record and inspect falsifiable hypotheses.")
findings_app = typer.Typer(help="Record and inspect evidence-linked findings.")
run_sets_app = typer.Typer(help="Freeze immutable cohorts of runs.")
analyze_app = typer.Typer(help="Run evidence analyses.")
artifacts_app = typer.Typer(help="List and inspect immutable artifacts.")
config_app = typer.Typer(help="Inspect effective workspace policy.")
measurements_app = typer.Typer(help="Run bounded curated measurement queries.")
evidence_app = typer.Typer(help="Retrieve one typed immutable evidence reference.")
experiment_app = typer.Typer(help="Plan, run, and inspect controlled experiments.")
fault_app = typer.Typer(help="Plan, run, and inspect loopback Toxiproxy fault experiments.")
inference_app = typer.Typer(help="Configure, plan, and run local inference scenarios.")
stacks_app = typer.Typer(help="Inspect bounded call relationships and stacks.")
trace_app = typer.Typer(help="Inspect bounded temporal trace windows.")
adapters_app = typer.Typer(help="Discover and approve third-party adapter entry points.")
pipelines_app = typer.Typer(help="Register and compare immutable artifact pipelines.")
app.add_typer(catalog_app, name="catalog")
app.add_typer(runs_app, name="runs")
app.add_typer(extract_app, name="extract")
app.add_typer(mcp_app, name="mcp")
app.add_typer(capture_app, name="capture")
app.add_typer(workload_app, name="workload")
app.add_typer(investigations_app, name="investigations")
app.add_typer(hypotheses_app, name="hypotheses")
app.add_typer(findings_app, name="findings")
app.add_typer(run_sets_app, name="run-sets")
app.add_typer(analyze_app, name="analyze")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(config_app, name="config")
app.add_typer(measurements_app, name="measurements")
app.add_typer(evidence_app, name="evidence")
app.add_typer(experiment_app, name="experiment")
app.add_typer(fault_app, name="fault")
app.add_typer(inference_app, name="inference")
app.add_typer(stacks_app, name="stacks")
app.add_typer(trace_app, name="trace")
app.add_typer(adapters_app, name="adapters")
app.add_typer(pipelines_app, name="pipelines")

WorkspaceOption = Annotated[
    Path | None,
    typer.Option("--workspace", help="Explicit .diagnostics workspace path."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit the structured JSON result."),
]


@dataclass(frozen=True, slots=True)
class _CliDefaults:
    workspace: Path | None = None
    project_root: Path | None = None
    json_output: bool = False
    quiet: bool = False
    timeout_seconds: float | None = None


_CLI_DEFAULTS: ContextVar[_CliDefaults | None] = ContextVar("flameox_cli_defaults")


def _cli_defaults() -> _CliDefaults:
    return _CLI_DEFAULTS.get(None) or _CliDefaults()


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the installed flameox version.",
    ),
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Default explicit .diagnostics workspace for this invocation.",
        ),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Default project root used for workspace discovery.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit structured JSON from the selected command."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress human-readable success output."),
    ] = False,
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        typer.Option("--log-level", help="Set local diagnostic log verbosity."),
    ] = "WARNING",
    timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            min=0.001,
            help="Bound asynchronous CLI operations in seconds.",
        ),
    ] = None,
) -> None:
    """Collect and query local runtime evidence."""
    logging.basicConfig(level=getattr(logging, log_level))
    _CLI_DEFAULTS.set(
        _CliDefaults(
            workspace=workspace,
            project_root=project_root,
            json_output=json_output,
            quiet=quiet,
            timeout_seconds=timeout_seconds,
        )
    )


def _workspace(explicit: Path | None) -> Workspace:
    defaults = _cli_defaults()
    return Workspace.discover(
        defaults.project_root or Path.cwd(),
        explicit=explicit or defaults.workspace,
    )


def _emit(value: BaseModel | dict[str, Any] | list[Any], *, as_json: bool) -> None:
    defaults = _cli_defaults()
    as_json = as_json or defaults.json_output
    if isinstance(value, BaseModel):
        payload: Any = value.model_dump(mode="json")
    else:
        payload = value
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if defaults.quiet:
        return
    if isinstance(payload, dict):
        for key, item in payload.items():
            typer.echo(f"{key}: {_terminal_text(item)}")
        return
    for item in payload:
        typer.echo(_terminal_text(item))


def _terminal_text(value: object) -> str:
    if isinstance(value, str):
        return value.encode("unicode_escape", errors="backslashreplace").decode("ascii")
    return repr(value)


def _run_async[T](operation: Callable[[], Awaitable[T]]) -> T:
    async def bounded() -> T:
        timeout = _cli_defaults().timeout_seconds
        if timeout is None:
            return await operation()
        try:
            with anyio.fail_after(timeout):
                return await operation()
        except TimeoutError as exc:
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                "The CLI operation exceeded its explicit timeout.",
            ) from exc

    return anyio.run(bounded)


def _request[RequestT: BaseModel](model: type[RequestT], value: str) -> RequestT:
    return _adapt_request(TypeAdapter(model), value)


def _adapt_request[RequestT](adapter: TypeAdapter[RequestT], value: str) -> RequestT:
    try:
        if value.startswith("@"):
            return adapter.validate_json(Path(value[1:]).read_text())
        return adapter.validate_json(value)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Structured input is invalid: {exc}") from exc


def _fail(error: DomainError, *, as_json: bool = False) -> NoReturn:
    if as_json or _cli_defaults().json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "result": None,
                    "error": error.to_detail(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo(f"{error.code.value}: {_terminal_text(error.message)}", err=True)
        if error.run_id is not None:
            typer.echo(f"  run_id: {_terminal_text(error.run_id)}", err=True)
        for key in ("partial_artifact_ids", "partial_artifact_run_ids"):
            if key in error.details:
                typer.echo(f"  {key}: {_terminal_text(error.details[key])}", err=True)
        for remediation in error.remediation:
            typer.echo(f"  {_terminal_text(remediation)}", err=True)
    exit_codes = {
        ErrorCode.INVALID_ARGUMENTS: 2,
        ErrorCode.WORKSPACE_NOT_FOUND: 2,
        ErrorCode.WORKSPACE_INVALID: 5,
        ErrorCode.RUN_NOT_FOUND: 1,
        ErrorCode.CAPABILITY_UNAVAILABLE: 3,
        ErrorCode.INVALID_CAPTURE_PLAN: 9,
        ErrorCode.EXECUTION_REFUSED: 9,
        ErrorCode.PROCESS_FAILED: 4,
        ErrorCode.PROCESS_TIMEOUT: 8,
        ErrorCode.PROCESS_CANCELLED: 8,
        ErrorCode.ARTIFACT_TOO_LARGE: 5,
        ErrorCode.STORAGE_QUOTA_EXCEEDED: 5,
        ErrorCode.ARTIFACT_INTEGRITY_FAILED: 5,
        ErrorCode.ARTIFACT_PARSE_FAILED: 5,
        ErrorCode.EVIDENCE_SCHEMA_MISMATCH: 5,
        ErrorCode.COMPARISON_INVALID: 6,
        ErrorCode.QUERY_BUDGET_EXCEEDED: 5,
        ErrorCode.WRITE_LOCK_TIMEOUT: 7,
        ErrorCode.SENSITIVE_ARTIFACT_REFUSED: 9,
        ErrorCode.REVISION_CONFLICT: 7,
        ErrorCode.STALE_CURSOR: 7,
    }
    raise typer.Exit(exit_codes.get(error.code, 1))


def _validation_error(error: ValidationError) -> DomainError:
    fields = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    ]
    return DomainError(
        ErrorCode.INVALID_ARGUMENTS,
        "Inference configuration input is invalid.",
        details={"validation_errors": fields},
        remediation=("Correct the reported fields and retry the same configuration command.",),
    )


@app.command("init")
def initialize(
    project_root: Annotated[
        Path | None,
        typer.Argument(help="Project root in which to create .diagnostics."),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Initialize a local flameox workspace."""
    try:
        initialized = Workspace.initialize(
            project_root or _cli_defaults().project_root or Path("."),
            workspace_root=workspace or _cli_defaults().workspace,
        )
        result = workspace_status(initialized)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@app.command("qualify-artifact-import")
def qualify_artifact_import(
    source_run_id: Annotated[str, typer.Argument(help="Run that owns the preserved artifact.")],
    artifact_id: Annotated[str, typer.Argument(help="Artifact to validate and qualify.")],
    profile: Annotated[ImportProfile, typer.Option("--profile", case_sensitive=False)],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Create a qualified import run from preserved immutable evidence."""
    try:
        result = ImportService(_workspace(workspace)).qualify_artifact(
            QualifyArtifactImportRequest(
                run_id=source_run_id,
                artifact_id=artifact_id,
                profile=profile,
            )
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("setup")
def setup(
    claude: Annotated[bool, typer.Option("--claude", help="Configure Claude Code.")] = False,
    cursor: Annotated[bool, typer.Option("--cursor", help="Configure Cursor.")] = False,
    opencode: Annotated[bool, typer.Option("--opencode", help="Configure OpenCode.")] = False,
    codex: Annotated[bool, typer.Option("--codex", help="Configure Codex.")] = False,
    gemini: Annotated[bool, typer.Option("--gemini", help="Configure Gemini CLI.")] = False,
    antigravity: Annotated[
        bool,
        typer.Option("--antigravity", help="Configure Antigravity."),
    ] = False,
    all_clients: Annotated[
        bool,
        typer.Option("--all", help="Select every supported MCP client."),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Select every currently detected MCP client."),
    ] = False,
    remove: Annotated[
        bool,
        typer.Option("--remove", help="Remove flameox from the selected MCP clients."),
    ] = False,
    rollback: Annotated[
        str | None,
        typer.Option("--rollback", metavar="VERSION", help="Activate an installed version."),
    ] = None,
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help="Verify connected client launchers and the active MCP runtime.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Apply an explicit plan without confirmation."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the exact plan without making changes."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Install a managed runtime and connect local MCP clients."""
    if os.environ.get("FLAMEOX_NPM_BOOTSTRAP") == "1" and not json_output:
        typer.echo("Managed runtime ready. Starting flameox setup...", err=True)
    service = SetupService(
        home=Path(os.environ["FLAMEOX_SETUP_HOME"]) if "FLAMEOX_SETUP_HOME" in os.environ else None,
        data_root=Path(os.environ["FLAMEOX_SETUP_DATA_ROOT"])
        if "FLAMEOX_SETUP_DATA_ROOT" in os.environ
        else None,
        jsonc_helper=Path(os.environ["FLAMEOX_SETUP_JSONC_HELPER"])
        if "FLAMEOX_SETUP_JSONC_HELPER" in os.environ
        else None,
        node_executable=os.environ.get("FLAMEOX_SETUP_NODE", "node"),
        uv_executable=os.environ.get("FLAMEOX_SETUP_UV", "uv"),
    )
    try:
        inspection = service.inspect()
    except DomainError as error:
        _fail(error)
    explicit_clients = setup_ui.selected_clients(
        (
            (SetupClient.CLAUDE, claude),
            (SetupClient.CURSOR, cursor),
            (SetupClient.OPENCODE, opencode),
            (SetupClient.CODEX, codex),
            (SetupClient.GEMINI, gemini),
            (SetupClient.ANTIGRAVITY, antigravity),
        )
    )
    action_count = sum((remove, rollback is not None, verify))
    if action_count > 1:
        _fail(
            DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Choose only one of --remove, --rollback, or --verify.",
            )
        )
    explicit_selection = bool(explicit_clients or all_clients or refresh)
    if sum((bool(explicit_clients), all_clients, refresh)) > 1:
        _fail(
            DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Choose client flags, --all, or --refresh; do not combine them.",
            )
        )
    if verify and explicit_selection:
        _fail(
            DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "--verify does not accept client selections.",
            )
        )
    if yes and not (explicit_selection or rollback is not None or verify):
        _fail(
            DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "--yes requires explicit clients, --all, --refresh, --rollback, or --verify.",
            )
        )

    operation = (
        SetupOperation.VERIFY
        if verify
        else SetupOperation.ROLLBACK
        if rollback is not None
        else SetupOperation.REMOVE
        if remove
        else SetupOperation.CONFIGURE
    )
    version: str | None = rollback if rollback is not None else __version__
    clients = explicit_clients
    if all_clients:
        clients = tuple(SetupClient)
    elif refresh:
        clients = inspection.detected_clients

    interactive = not (
        explicit_selection
        or rollback is not None
        or verify
        or dry_run
        or json_output
        or _cli_defaults().json_output
        or yes
    )
    try:
        if interactive:
            if not sys.stdin.isatty():
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Interactive setup requires a terminal.",
                    remediation=("Select clients explicitly and pass --yes or --dry-run.",),
                )
            setup_ui.print_banner(inspection, bootstrap_version=__version__)
            if remove:
                action = SetupOperation.REMOVE.value
            elif inspection.active_version is None:
                action = SetupOperation.CONFIGURE.value
            else:
                action = setup_ui.choose_action(inspection, __version__)
            if action == "exit":
                raise typer.Exit
            if action == "update":
                operation = SetupOperation.CONFIGURE
                clients = inspection.configured_clients
            else:
                operation = SetupOperation(action)
            if operation is SetupOperation.VERIFY:
                clients = ()
            elif operation is SetupOperation.ROLLBACK:
                version = setup_ui.choose_rollback_version(inspection, inspection.active_version)
                clients = inspection.configured_clients
            elif action != "update":
                clients = setup_ui.choose_clients(
                    inspection,
                    remove=operation is SetupOperation.REMOVE,
                )
                version = (
                    None
                    if operation is SetupOperation.REMOVE
                    else setup_ui.effective_runtime_version(inspection, __version__)
                )
        elif operation is SetupOperation.ROLLBACK:
            clients = clients or inspection.configured_clients

        resolved = service.plan(operation=operation, clients=clients, version=version)
        if dry_run:
            _emit(resolved.public, as_json=json_output)
            return
        if json_output or _cli_defaults().json_output:
            if not yes:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "JSON setup requires --yes or --dry-run.",
                )
        elif not yes:
            setup_ui.print_plan(resolved.public)
            if not setup_ui.confirm_apply():
                raise DomainError(ErrorCode.PROCESS_CANCELLED, "Setup cancelled.")
        report = _run_async(lambda: service.apply(resolved))
    except DomainError as error:
        _fail(error)
    if json_output or _cli_defaults().json_output:
        _emit(report, as_json=True)
    else:
        setup_ui.print_report(report)


@app.command("status")
def status(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show workspace and corpus status."""
    try:
        result = workspace_status(_workspace(workspace))
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@app.command("capabilities")
def capabilities(
    active: Annotated[
        bool,
        typer.Option("--active", help="Execute bounded brokered capability probes."),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Ignore process-local active-probe cache."),
    ] = False,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Passively report installed collectors and analysis libraries."""
    selected: Workspace | None
    if workspace is None:
        try:
            selected = _workspace(None)
        except DomainError as error:
            if error.code is not ErrorCode.WORKSPACE_NOT_FOUND:
                _fail(error)
            selected = None
    else:
        try:
            selected = _workspace(workspace)
        except DomainError as error:
            _fail(error)
    service = CapabilityService(selected)
    if active:

        async def probe() -> BaseModel:
            return await service.list_active(refresh=refresh)

        result: BaseModel = _run_async(probe)
    else:
        result = service.list()
    _emit(result, as_json=json_output)


@adapters_app.command("list")
def adapters_list(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Discover third-party entry points without importing their code."""
    try:
        result = AdapterRegistry(_workspace(workspace)).discover()
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@adapters_app.command("approve")
def adapters_approve(
    distribution: Annotated[str, typer.Argument(help="Installed distribution name.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Approve the exact installed identity of one adapter distribution."""
    try:
        result = AdapterRegistry(_workspace(workspace)).approve(distribution)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@adapters_app.command("revoke")
def adapters_revoke(
    distribution: Annotated[str, typer.Argument(help="Approved distribution name.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Revoke a local third-party adapter approval."""
    try:
        result = AdapterRegistry(_workspace(workspace)).revoke(distribution)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@app.command("validate")
def validate(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Hash every artifact and Parquet file."),
    ] = False,
) -> None:
    """Validate the immutable corpus without repairing it."""
    try:
        result = IntegrityService(_workspace(workspace)).validate(
            IntegrityLevel.FULL if full else IntegrityLevel.QUICK
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)
    if not result.valid:
        raise typer.Exit(1)


@app.command("gc")
def garbage_collect(
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Move the displayed candidates to recoverable trash."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Display candidates without moving them."),
    ] = False,
    purge: Annotated[
        str | None,
        typer.Option(
            "--purge",
            metavar="TRASH_MANIFEST",
            help="Permanently delete one expired trash manifest.",
        ),
    ] = None,
    restore: Annotated[
        str | None,
        typer.Option(
            "--restore",
            metavar="TRASH_MANIFEST",
            help="Restore one recoverable trash manifest.",
        ),
    ] = None,
    minimum_age_hours: Annotated[
        int,
        typer.Option("--minimum-age-hours", min=1),
    ] = 24,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Plan garbage collection; mutation requires explicit --apply."""
    try:
        collector = GarbageCollector(_workspace(workspace))
        selected = sum((apply, dry_run, purge is not None, restore is not None))
        if selected > 1:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Choose only one of --dry-run, --apply, --purge, or --restore.",
            )
        if purge is not None:
            result: BaseModel = collector.purge(purge)
        elif restore is not None:
            result = collector.restore(restore)
        else:
            plan = collector.plan(minimum_age_hours=minimum_age_hours)
            result = collector.apply(plan) if apply else plan
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("recover")
def recover(
    quarantine_id: Annotated[
        str | None,
        typer.Option("--quarantine", help="Restore one quarantined item by ID."),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Recover interrupted operations, or restore one quarantined item."""
    try:
        selected_workspace = _workspace(workspace)
        result = (
            QuarantineService(selected_workspace).restore(quarantine_id)
            if quarantine_id is not None
            else RecoveryService(selected_workspace).recover()
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@config_app.command("show")
def config_show(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show strict workspace policy after validation."""
    try:
        result = _workspace(workspace).config
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("import")
def import_artifact(
    path: Annotated[Path, typer.Argument(help="Artifact file to import.")],
    kind: Annotated[
        ArtifactKind,
        typer.Option("--kind", case_sensitive=False),
    ] = ArtifactKind.COLLECTOR_METADATA,
    media_type: Annotated[
        str | None,
        typer.Option("--media-type"),
    ] = None,
    sensitivity: Annotated[
        Sensitivity,
        typer.Option("--sensitivity", case_sensitive=False),
    ] = Sensitivity.INTERNAL,
    producer: Annotated[str | None, typer.Option("--producer")] = None,
    producer_version: Annotated[str | None, typer.Option("--producer-version")] = None,
    profile: Annotated[
        ImportProfile | None,
        typer.Option("--profile", case_sensitive=False),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Import one immutable native artifact as a new import run."""
    try:
        result = ImportService(_workspace(workspace)).import_artifact(
            ImportArtifactRequest(
                path=path,
                kind=kind,
                media_type=media_type,
                sensitivity=sensitivity,
                producer=producer,
                producer_version=producer_version,
                profile=profile,
                allow_external_path=True,
            )
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("register-kernel-validation")
def register_kernel_validation(
    run_id: Annotated[str, typer.Argument(help="Succeeded execution run that produced the file.")],
    path: Annotated[Path, typer.Argument(help="flameox.kernel-validation.v2 JSON file.")],
    expected_run_revision: Annotated[int, typer.Option("--expected-run-revision", min=0)],
    sensitivity: Annotated[
        Sensitivity,
        typer.Option("--sensitivity", case_sensitive=False),
    ] = Sensitivity.INTERNAL,
    pipeline_id: Annotated[str | None, typer.Option("--pipeline-id")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Attach immutable kernel-validation evidence to its exact producing run."""
    try:
        result = KernelValidationRegistrationService(_workspace(workspace)).register(
            RegisterKernelValidationRequest(
                run_id=run_id,
                expected_run_revision=expected_run_revision,
                path=path,
                sensitivity=sensitivity,
                allow_external_path=True,
                pipeline_id=pipeline_id,
            )
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("import-xctrace")
def import_xctrace(
    path: Annotated[Path, typer.Argument(help="Native Metal System Trace .trace bundle.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Preserve a native Metal trace bundle and its bounded xctrace TOC export."""
    try:
        result = _run_async(
            lambda: XctraceService(_workspace(workspace)).import_trace(
                XctraceImportRequest(trace_path=path, allow_external_path=True),
            )
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("import-kernel-build")
def import_kernel_build(
    path: Annotated[Path, typer.Argument(help="flameox.kernel-build.v1 or v2 manifest to import.")],
    sensitivity: Annotated[
        Sensitivity,
        typer.Option("--sensitivity", case_sensitive=False),
    ] = Sensitivity.INTERNAL,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Import a bounded compiler-artifact bundle and register its pipeline."""
    try:
        result = KernelBuildImportService(_workspace(workspace)).import_manifest(
            path,
            sensitivity=sensitivity,
            allow_external_path=True,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@app.command("import-nvbench")
def import_nvbench(
    path: Annotated[Path, typer.Argument(help="NVBench --json output file to import.")],
    sensitivity: Annotated[
        Sensitivity,
        typer.Option("--sensitivity", case_sensitive=False),
    ] = Sensitivity.INTERNAL,
    expected_sha256: Annotated[
        str | None,
        typer.Option("--expected-sha256", help="Declared SHA-256 digest of the JSON file."),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Import an NVBench JSON and its provider-declared sidecars as one bundle."""
    try:
        result = NvbenchImportService(_workspace(workspace)).import_json(
            path,
            sensitivity=sensitivity,
            allow_external_path=True,
            expected_sha256=expected_sha256,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@pipelines_app.command("list")
def pipeline_list(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    pipeline_name: Annotated[str | None, typer.Option("--pipeline-name")] = None,
    pipeline_schema: Annotated[str | None, typer.Option("--pipeline-schema")] = None,
    producer: Annotated[str | None, typer.Option("--producer")] = None,
    source_artifact_id: Annotated[str | None, typer.Option("--source-artifact-id")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1_000)] = 20,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Discover bounded immutable artifact pipelines."""
    try:
        result = ArtifactPipelineService(_workspace(workspace)).list(
            filter=PipelineFilter(
                run_id=run_id,
                pipeline_name=pipeline_name,
                pipeline_schema=pipeline_schema,
                producer=producer,
                source_artifact_id=source_artifact_id,
            ),
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@pipelines_app.command("show")
def pipeline_show(
    pipeline_id: Annotated[str, typer.Argument()],
    candidate_limit: Annotated[int, typer.Option("--candidate-limit", min=0, max=20)] = 20,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Inspect one pipeline and compatible comparison candidates."""
    try:
        result = ArtifactPipelineService(_workspace(workspace)).get(
            pipeline_id,
            candidate_limit=candidate_limit,
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@pipelines_app.command("register")
def pipeline_register(
    request_path: Annotated[
        Path,
        typer.Argument(help="JSON file containing a RegisterPipelineRequest."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Register a bounded pipeline over artifacts already attached to one run."""
    try:
        request = RegisterPipelineRequest.model_validate_json(request_path.read_text())
        result = ArtifactPipelineService(_workspace(workspace)).register(request)
    except ValidationError as error:
        _fail(_validation_error(error), as_json=json_output)
    except OSError as error:
        _fail(
            DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"Pipeline request could not be read: {error}.",
            ),
            as_json=json_output,
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@pipelines_app.command("compare")
def pipeline_compare(
    baseline_pipeline_id: Annotated[str, typer.Argument()],
    candidate_pipeline_id: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Compare two compatible ordered artifact pipelines."""
    try:
        service = ArtifactPipelineService(_workspace(workspace))
        baseline = service.resolve_reference(baseline_pipeline_id)
        candidate = service.resolve_reference(candidate_pipeline_id)
        result = service.compare(baseline.pipeline_id, candidate.pipeline_id)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@app.command("open")
def open_artifact(
    artifact_id: str,
    launch: Annotated[
        bool,
        typer.Option("--launch", help="Execute the displayed local viewer command."),
    ] = False,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Print or explicitly launch the installed native file viewer."""
    if launch and json_output:
        _fail(
            DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "--launch cannot be combined with --json.",
            )
        )

    async def open_viewer() -> BaseModel:
        service = NativeViewerService(_workspace(workspace))
        if launch:
            return await service.launch(artifact_id)
        return service.plan(artifact_id)

    try:
        result = _run_async(open_viewer)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@workload_app.command("list")
def workload_list(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """List named project workloads and their current definition identities."""
    try:
        service = WorkloadService(_workspace(workspace))
        result = {
            "schema_version": 1,
            "workloads": [
                service.definition(name).model_dump(mode="json") for name in service.names()
            ],
        }
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@workload_app.command("show")
def workload_show(
    name: Annotated[str, typer.Argument(help="Named workload in flameox.toml.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show a workload's canonical identity and active definition state."""
    try:
        result = WorkloadService(_workspace(workspace)).inspect(name)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@workload_app.command("run")
def workload_run(
    name: Annotated[str, typer.Argument(help="Named workload in flameox.toml.")],
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Execute a named workload with process evidence but no profiler."""

    async def run() -> BaseModel:
        service = CaptureService(_workspace(workspace))
        plan = await service.plan(
            workload_name=name,
            adapter="command",
            parameters=_parameter_overrides(parameters),
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        return await service.execute(plan.plan_token)

    try:
        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@inference_app.command("list")
def inference_list(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """List declared inference servers and replay scenarios."""
    try:
        result = WorkloadService(_workspace(workspace)).list_inference()
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("configure-server")
def inference_configure_server(
    name: str,
    mode: InferenceServerMode,
    model: str,
    provider: Annotated[InferenceServerProvider, typer.Option("--provider")] = (
        InferenceServerProvider.VLLM
    ),
    benchmark_python: Annotated[str | None, typer.Option("--benchmark-python")] = None,
    operation: ConfigurationOperation = ConfigurationOperation.CREATE,
    workload: Annotated[str | None, typer.Option("--workload")] = None,
    base_url: Annotated[str, typer.Option("--base-url")] = "http://127.0.0.1:8000",
    model_revision: Annotated[str | None, typer.Option("--model-revision")] = None,
    tokenizer: Annotated[str | None, typer.Option("--tokenizer")] = None,
    tokenizer_revision: Annotated[str | None, typer.Option("--tokenizer-revision")] = None,
    quantization: Annotated[str | None, typer.Option("--quantization")] = None,
    expected_configuration_id: Annotated[
        str | None, typer.Option("--expected-configuration-id")
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Create or replace a typed vLLM server declaration."""
    try:
        result = WorkloadService(_workspace(workspace)).configure_inference_server(
            ConfigureInferenceServerRequest(
                name=name,
                operation=operation,
                expected_configuration_id=expected_configuration_id,
                config=parse_inference_server_config(
                    {
                        "provider": provider,
                        "benchmark_python": benchmark_python,
                        "mode": mode,
                        "workload": workload,
                        "base_url": base_url,
                        "model": model,
                        "model_revision": model_revision,
                        "tokenizer": tokenizer,
                        "tokenizer_revision": tokenizer_revision,
                        "quantization": quantization,
                    }
                ),
            )
        )
    except ValidationError as error:
        _fail(_validation_error(error), as_json=json_output)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("configure-scenario")
def inference_configure_scenario(
    name: str,
    server: str,
    provider: InferenceScenarioProvider,
    operation: ConfigurationOperation = ConfigurationOperation.CREATE,
    endpoint_type: InferenceEndpointType = InferenceEndpointType.CHAT,
    streaming: bool = True,
    trace_artifact_id: Annotated[str | None, typer.Option("--trace-artifact-id")] = None,
    num_prompts: Annotated[int, typer.Option("--num-prompts", min=1)] = 1,
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1)] = None,
    request_rate: Annotated[float | None, typer.Option("--request-rate", min=0.001)] = None,
    burstiness: Annotated[float | None, typer.Option("--burstiness", min=0.001)] = None,
    warmup_request_count: Annotated[int, typer.Option("--warmup-request-count", min=0)] = 0,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
    speedup_ratio: Annotated[float, typer.Option("--speedup-ratio", min=0.001)] = 1.0,
    semantic_oracle_workload: Annotated[
        str | None, typer.Option("--semantic-oracle-workload")
    ] = None,
    random_input_len: Annotated[int | None, typer.Option("--random-input-len", min=1)] = None,
    random_output_len: Annotated[int | None, typer.Option("--random-output-len", min=1)] = None,
    random_range_ratio: Annotated[
        float | None, typer.Option("--random-range-ratio", min=0.001, max=1.0)
    ] = None,
    expected_configuration_id: Annotated[
        str | None, typer.Option("--expected-configuration-id")
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Create or replace a typed inference scenario declaration."""
    try:
        result = WorkloadService(_workspace(workspace)).configure_inference_scenario(
            ConfigureInferenceScenarioRequest(
                name=name,
                operation=operation,
                expected_configuration_id=expected_configuration_id,
                config=parse_inference_scenario_config(
                    {
                        "server": server,
                        "provider": provider,
                        "endpoint_type": endpoint_type,
                        "streaming": streaming,
                        "trace_artifact_id": trace_artifact_id,
                        "num_prompts": num_prompts,
                        "concurrency": concurrency,
                        "request_rate": request_rate,
                        "burstiness": burstiness,
                        "warmup_request_count": warmup_request_count,
                        "seed": seed,
                        "speedup_ratio": speedup_ratio,
                        "semantic_oracle_workload": semantic_oracle_workload,
                        "random_input_len": random_input_len,
                        "random_output_len": random_output_len,
                        "random_range_ratio": random_range_ratio,
                    }
                ),
            )
        )
    except ValidationError as error:
        _fail(_validation_error(error), as_json=json_output)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("plan")
def inference_plan(
    name: Annotated[str, typer.Argument(help="Declared inference scenario name.")],
    timeout_seconds: Annotated[float | None, typer.Option("--timeout")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Probe a local server and construct a replay command without executing it."""
    try:
        result = InferenceReplayService(_workspace(workspace)).plan(
            name, timeout_seconds=timeout_seconds
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("run")
def inference_run(
    name: Annotated[str, typer.Argument(help="Declared inference scenario name.")],
    timeout_seconds: Annotated[float | None, typer.Option("--timeout")] = None,
    expected_plan_id: Annotated[str | None, typer.Option("--expected-plan-id")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Run one planned replay through the bounded subprocess broker."""

    async def run() -> BaseModel:
        service = InferenceReplayService(_workspace(workspace))
        plan = service.plan(
            name,
            timeout_seconds=timeout_seconds,
            expected_plan_id=expected_plan_id,
        )
        return await service.run(plan.plan_token, expected_plan_id=expected_plan_id)

    try:
        result = _run_async(run)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("requests")
def inference_requests(
    run_id: Annotated[str, typer.Argument(help="Run with published inference requests.")],
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Page through bounded prompt-free inference request evidence."""
    try:
        result = EvidenceQueryService(_workspace(workspace)).inference_requests(
            run_id=run_id, limit=limit, cursor=cursor
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("profile-plan")
def inference_profile_plan(
    server: Annotated[str, typer.Argument(help="Managed inference server name.")],
    profiler: Annotated[Literal["torch_profiler", "nsight_systems"], typer.Option("--profiler")],
    nsys_executable: Annotated[Path | None, typer.Option("--nsys-executable")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Build a diagnostic-only managed vLLM profiling plan."""
    try:
        result = InferenceProfilingService(_workspace(workspace)).plan(
            server, profiler=profiler, nsys_executable=nsys_executable
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@inference_app.command("profile-run")
def inference_profile_run(
    server: Annotated[str, typer.Argument(help="Managed inference server name.")],
    scenario: Annotated[str, typer.Argument(help="Declared inference scenario name.")],
    profiler: Annotated[Literal["torch_profiler", "nsight_systems"], typer.Option("--profiler")],
    measurement_run_id: Annotated[
        str,
        typer.Option(
            "--measurement-run-id",
            help="Successful compatible unprofiled inference run to link.",
        ),
    ],
    timeout_seconds: Annotated[float, typer.Option("--timeout", min=1)] = 300,
    nsys_executable: Annotated[Path | None, typer.Option("--nsys-executable")] = None,
    expected_plan_id: Annotated[str | None, typer.Option("--expected-plan-id")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Run one diagnostic profile window linked to a declared scenario."""

    async def run() -> BaseModel:
        service = InferenceProfilingService(_workspace(workspace))
        plan = service.plan(
            server,
            profiler=profiler,
            nsys_executable=nsys_executable,
            expected_plan_id=expected_plan_id,
            scenario_name=scenario,
            measurement_run_id=measurement_run_id,
            timeout_seconds=timeout_seconds,
        )
        return await service.capture(plan.plan_token, expected_plan_id=expected_plan_id)

    try:
        result = _run_async(run)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


def _json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Malformed {label} JSON: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"{label.capitalize()} must be a JSON object.",
        )
    return cast(dict[str, Any], parsed)


def _parameter_overrides(value: str) -> dict[str, Any]:
    return _json_object(value, label="parameter overrides")


@experiment_app.command("plan")
def experiment_plan(
    name: Annotated[str, typer.Argument(help="Named experiment in flameox.toml.")],
    investigation_id: Annotated[str, typer.Option("--investigation")],
    adapter: Annotated[str, typer.Option("--adapter")] = "pyperf",
    hypothesis_id: Annotated[str | None, typer.Option("--hypothesis")] = None,
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Resolve and randomize an experiment without executing workloads."""

    async def plan() -> BaseModel:
        return await ExperimentService(_workspace(workspace)).plan(
            experiment_name=name,
            investigation_id=investigation_id,
            hypothesis_id=hypothesis_id,
            adapter=adapter,
            parameter_overrides=_parameter_overrides(parameters),
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    try:
        result = _run_async(plan)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@experiment_app.command("run")
def experiment_run(
    name: Annotated[str, typer.Argument(help="Named experiment in flameox.toml.")],
    investigation_id: Annotated[str, typer.Option("--investigation")],
    adapter: Annotated[str, typer.Option("--adapter")] = "pyperf",
    hypothesis_id: Annotated[str | None, typer.Option("--hypothesis")] = None,
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Plan and execute every predeclared treatment and block."""

    async def run() -> BaseModel:
        service = ExperimentService(_workspace(workspace))
        plan = await service.plan(
            experiment_name=name,
            investigation_id=investigation_id,
            hypothesis_id=hypothesis_id,
            adapter=adapter,
            parameter_overrides=_parameter_overrides(parameters),
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        return await service.run(plan.plan_token)

    try:
        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@experiment_app.command("show")
def experiment_show(
    experiment_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show one immutable experiment protocol."""
    try:
        result = ExperimentService(_workspace(workspace)).experiments.read(experiment_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@experiment_app.command("trial")
def experiment_trial(
    trial_id: str,
    experiment_id: Annotated[
        str | None,
        typer.Option("--experiment-id", help="Disambiguate a trial reused across experiments."),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show one immutable experiment trial and its structured oracle receipt."""
    try:
        result = ExperimentService(_workspace(workspace)).get_trial(
            trial_id,
            experiment_id=experiment_id,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@fault_app.command("plan")
def fault_plan(
    name: Annotated[str, typer.Argument(help="Named fault experiment in flameox.toml.")],
    investigation_id: Annotated[str, typer.Option("--investigation")],
    hypothesis_id: Annotated[str | None, typer.Option("--hypothesis")] = None,
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Plan a bounded loopback transport-fault experiment."""

    async def plan() -> BaseModel:
        return await FaultExperimentService(_workspace(workspace)).plan(
            experiment_name=name,
            investigation_id=investigation_id,
            hypothesis_id=hypothesis_id,
            parameter_overrides=_parameter_overrides(parameters),
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    try:
        result = _run_async(plan)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@fault_app.command("run")
def fault_run(
    name: Annotated[str | None, typer.Argument(help="Named fault experiment.")] = None,
    investigation_id: Annotated[str | None, typer.Option("--investigation")] = None,
    plan_token: Annotated[
        str | None,
        typer.Option("--plan-token", help="Opaque single-use capability returned by fault plan."),
    ] = None,
    hypothesis_id: Annotated[str | None, typer.Option("--hypothesis")] = None,
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Run a planned fault experiment, or plan one from its declared name."""

    async def run() -> BaseModel:
        service = FaultExperimentService(_workspace(workspace))
        selected_plan = plan_token
        if selected_plan is None:
            if name is None or investigation_id is None:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "fault run requires --plan-token or NAME with --investigation.",
                )
            planned = await service.plan(
                experiment_name=name,
                investigation_id=investigation_id,
                hypothesis_id=hypothesis_id,
                parameter_overrides=_parameter_overrides(parameters),
                execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
            )
            selected_plan = planned.plan_token
        elif selected_plan.startswith("sha256:"):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "--plan-token requires the opaque capability, not the public plan_id.",
                remediation=("Pass the plan_token returned by flameox fault plan.",),
            )
        return await service.run(selected_plan)

    try:
        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@fault_app.command("show")
def fault_show(
    result_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show one completed fault experiment result."""
    try:
        result = FaultExperimentService(_workspace(workspace)).show(result_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@capture_app.command("plan")
def capture_plan(
    adapter: Annotated[str, typer.Argument(help="Registered capture adapter.")],
    workload_name: Annotated[str, typer.Option("--workload")],
    parameters: Annotated[
        str,
        typer.Option("--parameters", help="JSON object of declared scalar overrides."),
    ] = "{}",
    adapter_options: Annotated[
        str,
        typer.Option(
            "--adapter-options",
            help="JSON object of adapter-specific capture options.",
        ),
    ] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Resolve a capture plan; adapters may run a bounded compatibility probe."""

    async def plan(values: dict[str, Any], options: dict[str, Any]) -> BaseModel:
        return await CaptureService(_workspace(workspace)).plan(
            workload_name=workload_name,
            adapter=adapter,
            parameters=values,
            adapter_options=options,
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    try:
        values = _parameter_overrides(parameters)
        options = _json_object(adapter_options, label="adapter options")
        result = _run_async(lambda: plan(values, options))
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@capture_app.command("run")
def capture_run(
    adapter: Annotated[str, typer.Argument(help="Registered capture adapter.")],
    workload_name: Annotated[str, typer.Option("--workload")],
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    adapter_options: Annotated[str, typer.Option("--adapter-options")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Plan and execute one named workload capture."""

    async def run(values: dict[str, Any], options: dict[str, Any]) -> BaseModel:
        service = CaptureService(_workspace(workspace))
        plan = await service.plan(
            workload_name=workload_name,
            adapter=adapter,
            parameters=values,
            adapter_options=options,
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        return await service.execute(plan.plan_token)

    try:
        values = _parameter_overrides(parameters)
        options = _json_object(adapter_options, label="adapter options")
        result = _run_async(lambda: run(values, options))
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@capture_app.command("execute")
def capture_execute(
    plan_token: Annotated[
        str,
        typer.Option(
            "--plan-token",
            envvar="FLAMEOX_PLAN_TOKEN",
            help=(
                "Opaque single-use capability returned by capture plan; use "
                "FLAMEOX_PLAN_TOKEN to avoid shell history."
            ),
        ),
    ],
    expected_plan_id: Annotated[
        str | None,
        typer.Option(
            "--expected-plan-id",
            help="Public plan identity that must match before the capability is consumed.",
        ),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Execute one previously reviewed capture plan without replanning."""

    async def execute() -> BaseModel:
        return await CaptureService(_workspace(workspace)).execute(
            plan_token,
            expected_plan_id=expected_plan_id,
        )

    try:
        result = _run_async(execute)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@catalog_app.command("rebuild")
def catalog_rebuild(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Rebuild the disposable DuckDB catalog."""
    try:
        selected = _workspace(workspace)
        catalog = Catalog(selected)
        catalog.rebuild()
        result = catalog.status()
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@catalog_app.command("compact")
def catalog_compact(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Replace reachable small generations with one immutable generation."""
    try:
        result = asyncio.run(CompactionService(_workspace(workspace)).compact())
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@catalog_app.command("status")
def catalog_status(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show disposable catalog metadata."""
    try:
        result = Catalog(_workspace(workspace)).status()
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@catalog_app.command("validate")
def catalog_validate(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Validate the catalog shell and authoritative corpus inventory."""
    try:
        selected = _workspace(workspace)
        integrity = IntegrityService(selected).validate(IntegrityLevel.QUICK)
        result = {
            "catalog": Catalog(selected).status(),
            "integrity": integrity.model_dump(mode="json"),
        }
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)
    if not integrity.valid:
        raise typer.Exit(1)


@runs_app.command("list")
def runs_list(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 100,
) -> None:
    """List runs from one pinned corpus snapshot."""
    try:
        result = RunDiscoveryService(_workspace(workspace)).list(filter=RunFilter(), limit=limit)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(help="Run identifier.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show the current projection of one run manifest."""
    try:
        result = RunStore(_workspace(workspace)).read(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@artifacts_app.command("list")
def artifacts_list(
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 100,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """List bounded content objects with effective sensitivity."""
    try:
        result = ArtifactService(_workspace(workspace)).list(limit=limit)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@artifacts_app.command("show")
def artifacts_show(
    artifact_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show content identity and contextual registrations, not binary bytes."""
    try:
        result = ArtifactService(_workspace(workspace)).get(artifact_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@artifacts_app.command("preview")
def artifacts_preview(
    artifact_id: str,
    max_bytes: Annotated[int, typer.Option("--max-bytes", min=1, max=ARTIFACT_PREVIEW_MAX_BYTES)],
    max_lines: Annotated[int, typer.Option("--max-lines", min=1, max=ARTIFACT_PREVIEW_MAX_LINES)],
    offset: Annotated[int, typer.Option("--offset", min=0)] = 0,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Read an explicitly bounded text projection of eligible output evidence."""
    try:
        result: ArtifactTextPreview = ArtifactService(_workspace(workspace)).preview_text(
            artifact_id,
            offset=offset,
            max_bytes=max_bytes,
            max_lines=max_lines,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@investigations_app.command("create")
def investigations_create(
    structured_input: Annotated[
        str,
        typer.Argument(help="JSON request or @path/to/request.json."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Create an investigation with a concrete diagnostic question."""
    try:
        result = InvestigationService(_workspace(workspace)).create(
            _request(CreateInvestigationRequest, structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@investigations_app.command("list")
def investigations_list(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """List bounded current investigation projections."""
    try:
        values = InvestigationService(_workspace(workspace)).investigations.list()
        result = {
            "schema_version": 1,
            "investigations": [value.model_dump(mode="json") for value in values[:1_000]],
            "total": len(values),
            "returned": min(len(values), 1_000),
            "truncated": len(values) > 1_000,
        }
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@investigations_app.command("show")
def investigations_show(
    investigation_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show one investigation."""
    try:
        result = InvestigationService(_workspace(workspace)).investigations.read(investigation_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@hypotheses_app.command("record")
def hypotheses_record(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Record or revise a falsifiable hypothesis."""
    try:
        result = InvestigationService(_workspace(workspace)).record_hypothesis(
            _request(RecordHypothesisRequest, structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@hypotheses_app.command("show")
def hypotheses_show(
    hypothesis_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show the current hypothesis revision."""
    try:
        result = InvestigationService(_workspace(workspace)).hypotheses.read(hypothesis_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@findings_app.command("record")
def findings_record(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Record or revise a claim with validated evidence references."""
    try:
        result = FindingService(_workspace(workspace)).record(
            _request(RecordFindingRequest, structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@findings_app.command("list")
def findings_list(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """List bounded current finding projections."""
    try:
        values = FindingService(_workspace(workspace)).findings.list()
        result = {
            "schema_version": 1,
            "findings": [value.model_dump(mode="json") for value in values[:1_000]],
            "total": len(values),
            "returned": min(len(values), 1_000),
            "truncated": len(values) > 1_000,
        }
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@findings_app.command("show")
def findings_show(
    finding_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show the current finding revision."""
    try:
        result = FindingService(_workspace(workspace)).findings.read(finding_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@run_sets_app.command("freeze")
def run_sets_freeze(
    run_ids: Annotated[list[str], typer.Argument(help="Run IDs in stable order.")],
    corpus_commit_id: Annotated[
        str | None,
        typer.Option(
            "--corpus-commit",
            help="Freeze against this explicit corpus commit instead of current HEAD.",
        ),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Freeze a cohort against the current corpus snapshot."""
    try:
        result = RunSetService(_workspace(workspace)).freeze(
            FreezeRunIdsRequest(
                run_ids=tuple(run_ids),
                corpus_commit_id=corpus_commit_id,
            )
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@run_sets_app.command("show")
def run_sets_show(
    run_set_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show one immutable run set."""
    try:
        result = RunSetService(_workspace(workspace)).store.read(run_set_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("compare")
def analyze_compare(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Compare frozen cohorts using the declared paired estimand."""
    try:
        result = ComparisonService(_workspace(workspace)).compare(
            _adapt_request(TypeAdapter(CompareRunSetsRequest), structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("record-comparison")
def analyze_record_comparison(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Persist a comparison and its typed analysis provenance."""
    try:
        result = ComparisonService(_workspace(workspace)).record(
            _adapt_request(TypeAdapter(CompareRunSetsRequest), structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("compare-kernel-validation")
def analyze_compare_kernel_validation(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Compare exact correctness metrics from two frozen run cohorts."""
    try:
        result = KernelValidationComparisonService(_workspace(workspace)).compare(
            _adapt_request(TypeAdapter(KernelValidationCompareRequest), structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("record-kernel-validation-comparison")
def analyze_record_kernel_validation_comparison(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Persist a correctness comparison and its exact input provenance."""
    try:
        result = KernelValidationComparisonService(_workspace(workspace)).record(
            _adapt_request(TypeAdapter(KernelValidationCompareRequest), structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("record")
def analyze_record(
    structured_input: Annotated[str, typer.Argument(help="JSON request or @file.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Run and persist one versioned analysis recipe."""
    try:
        result = AnalysisMaterializationService(_workspace(workspace)).record(
            _adapt_request(_MATERIALIZE_ANALYSIS_REQUEST_ADAPTER, structured_input)
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("scaling")
def analyze_scaling(
    experiment_id: str,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Summarize per-trial medians and block completeness for an experiment."""
    try:
        result = RecipeService(_workspace(workspace)).scaling(experiment_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("hotspots")
def analyze_hotspots(
    input_id: Annotated[str, typer.Argument(help="Run or artifact identifier.")],
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded source-linked frame aggregates."""
    try:
        result = RecipeService(_workspace(workspace)).hotspots(input_id, limit=limit)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("memory")
def analyze_memory(
    input_id: Annotated[str, typer.Argument(help="Run or artifact identifier.")],
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Summarize explicit memory concepts and allocation frames."""
    try:
        result = RecipeService(_workspace(workspace)).memory(input_id, limit=limit)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("execution")
def analyze_execution(
    input_id: Annotated[str, typer.Argument(help="Run or artifact identifier.")],
    compare_to: Annotated[
        str | None,
        typer.Option("--compare-to", help="Second run or artifact to compare."),
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded coverage and semantic execution observations."""
    try:
        result = RecipeService(_workspace(workspace)).execution(
            input_id,
            comparison_input_id=compare_to,
            limit=limit,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("pytorch")
def analyze_pytorch(
    input_id: Annotated[str, typer.Argument(help="Run or artifact identifier.")],
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Summarize operators from a normalized torch.profiler trace."""
    try:
        result = RecipeService(_workspace(workspace)).pytorch(
            input_id,
            limit=limit,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("accelerator-launches")
def analyze_accelerator_launches(
    input_id: Annotated[str, typer.Argument(help="Run or artifact identifier.")],
    compare_to: Annotated[
        str | None,
        typer.Option("--compare-to", help="Optional comparison run or artifact."),
    ] = None,
    phase: Annotated[
        str | None,
        typer.Option("--phase", help="Exact normalized phase to select."),
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Summarize direct launches, graph launches, kernels, and accelerator idle gaps."""
    try:
        result = RecipeService(_workspace(workspace)).accelerator_launches(
            input_id,
            comparison_input_id=compare_to,
            phase=phase,
            limit=limit,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@analyze_app.command("failures")
def analyze_failures(
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Cluster the latest terminal states across the local run population."""
    try:
        result = RecipeService(_workspace(workspace)).failures(limit=limit)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@stacks_app.command("callers")
def stacks_callers(
    input_id: str,
    frame_id: str,
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 100,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded direct callers for one normalized frame."""
    try:
        result = DrilldownService(_workspace(workspace)).callers(
            input_id,
            frame_id,
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@stacks_app.command("callees")
def stacks_callees(
    input_id: str,
    frame_id: str,
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 100,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded direct callees for one normalized frame."""
    try:
        result = DrilldownService(_workspace(workspace)).callees(
            input_id,
            frame_id,
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@stacks_app.command("examples")
def stacks_examples(
    input_id: str,
    frame_id: str,
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 20,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded representative stacks containing one frame."""
    try:
        result = DrilldownService(_workspace(workspace)).examples(
            input_id,
            frame_id,
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@trace_app.command("window")
def trace_window(
    artifact_id: str,
    start_ns: Annotated[int, typer.Option("--start", min=0)],
    end_ns: Annotated[int, typer.Option("--end", min=1)],
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 100,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Query slices overlapping one bounded time interval."""
    try:

        async def run() -> BaseModel:
            return await TraceWindowService(_workspace(workspace)).get(
                artifact_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
                cursor=cursor,
                run_id=run_id,
            )

        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@trace_app.command("operation-window")
def operation_window(
    artifact_id: str,
    start_ns: Annotated[int, typer.Option("--start", min=0)],
    end_ns: Annotated[int, typer.Option("--end", min=1)],
    trace_id: Annotated[str | None, typer.Option("--trace-id")] = None,
    limit: Annotated[int | None, typer.Option(min=1, max=1_000)] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded OTLP spans overlapping a time window."""
    try:
        result = LifecycleEvidenceService(_workspace(workspace)).get_operation_window(
            artifact_id=artifact_id,
            start_ns=start_ns,
            end_ns=end_ns,
            trace_id=trace_id,
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@trace_app.command("transitions")
def operation_transitions(
    artifact_id: str,
    trace_id: Annotated[str | None, typer.Option("--trace-id")] = None,
    max_depth: Annotated[int, typer.Option("--max-depth", min=0, max=32)] = 8,
    limit: Annotated[int | None, typer.Option(min=1, max=1_000)] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return bounded parent/child transitions and coverage gaps."""
    try:
        result = LifecycleEvidenceService(_workspace(workspace)).get_operation_transitions(
            artifact_id=artifact_id,
            trace_id=trace_id,
            max_depth=max_depth,
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@trace_app.command("gaps")
def lifecycle_gaps(
    artifact_id: str,
    limit: Annotated[int | None, typer.Option(min=1, max=1_000)] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Return explicit OTLP timestamp, parent, and identity gaps."""
    try:
        result = LifecycleEvidenceService(_workspace(workspace)).get_lifecycle_gaps(
            artifact_id=artifact_id, limit=limit
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("otlp")
def extract_otlp(
    run_id: str,
    artifact_id: Annotated[str | None, typer.Option("--artifact-id")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Normalize a registered OTLP file artifact into bounded Parquet evidence."""
    try:
        result = OtlpTraceService(_workspace(workspace)).extract_otlp_trace(run_id, artifact_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@measurements_app.command("query")
def measurements_query(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    artifact_id: Annotated[str | None, typer.Option("--artifact-id")] = None,
    name_prefix: Annotated[str | None, typer.Option("--name-prefix")] = None,
    include_warmups: Annotated[bool, typer.Option("--include-warmups")] = False,
    limit: Annotated[int | None, typer.Option(min=1, max=1_000)] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Query measurements through reviewed filters and stable keyset cursors."""
    try:
        result = EvidenceQueryService(_workspace(workspace)).measurements(
            run_id=run_id,
            artifact_id=artifact_id,
            name_prefix=name_prefix,
            include_warmups=include_warmups,
            limit=limit,
            cursor=cursor,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@evidence_app.command("get")
def evidence_get(
    ref_type: Annotated[
        str,
        typer.Argument(help="Typed evidence kind."),
    ],
    ref_id: Annotated[str, typer.Argument(help="Typed evidence identifier.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Retrieve bounded evidence metadata without binary artifact content."""
    try:
        parsed_ref_type = EvidenceReferenceType(ref_type)
    except ValueError:
        _fail(
            DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unsupported evidence reference type {ref_type!r}.",
            )
        )
    try:
        result = EvidenceLookupService(_workspace(workspace)).get(
            parsed_ref_type,
            ref_id,
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@evidence_app.command("summarize")
def evidence_summarize(
    baseline_run_id: Annotated[str | None, typer.Option("--baseline-run")] = None,
    candidate_run_id: Annotated[str | None, typer.Option("--candidate-run")] = None,
    run_ids: Annotated[list[str] | None, typer.Option("--run")] = None,
    comparison_ids: Annotated[list[str] | None, typer.Option("--comparison")] = None,
    analysis_ids: Annotated[list[str] | None, typer.Option("--analysis")] = None,
    finding_ids: Annotated[list[str] | None, typer.Option("--finding")] = None,
    output_excerpts: Annotated[
        SummaryExcerptPolicy,
        typer.Option("--output-excerpts"),
    ] = SummaryExcerptPolicy.NONE,
    sensitive_context: Annotated[
        SummarySensitiveContextPolicy,
        typer.Option("--sensitive-context"),
    ] = SummarySensitiveContextPolicy.REDACT,
    output_format: Annotated[
        Literal["json", "markdown"],
        typer.Option("--format"),
    ] = "markdown",
    workspace: WorkspaceOption = None,
) -> None:
    """Render a bounded proof summary from immutable evidence references."""
    try:
        result = EvidenceSummaryService(_workspace(workspace)).summarize(
            EvidenceSummaryRequest(
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
                run_ids=tuple(run_ids or ()),
                comparison_ids=tuple(comparison_ids or ()),
                analysis_ids=tuple(analysis_ids or ()),
                finding_ids=tuple(finding_ids or ()),
                output_excerpts=output_excerpts,
                sensitive_context=sensitive_context,
            )
        )
    except DomainError as error:
        _fail(error)
    if output_format == "json":
        typer.echo(result.summary.model_dump_json(indent=2))
    else:
        typer.echo(result.markdown, nl=False)


@extract_app.command("pyperf")
def extract_pyperf(
    run_id: Annotated[str, typer.Argument(help="Import run containing pyperf JSON.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract pyperf workers, warmups, loops, and raw values."""
    try:
        result = PyPerfExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("benchmark-samples")
def extract_benchmark_samples(
    run_id: Annotated[
        str,
        typer.Argument(help="Import run containing flameox benchmark-samples v1 JSON."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract producer-neutral raw benchmark samples and timing semantics."""
    try:
        result = BenchmarkSamplesExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("node-cpu-prof")
def extract_node_cpu_prof(
    run_id: Annotated[str, typer.Argument(help="Run containing a V8 .cpuprofile artifact.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded evidence from a Node/V8 CPU profile."""
    try:
        result = V8CpuProfExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("node-heap-prof")
def extract_node_heap_prof(
    run_id: Annotated[str, typer.Argument(help="Run containing a V8 .heapprofile artifact.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded evidence from a Node/V8 sampling heap profile."""
    try:
        result = V8HeapProfExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("kernel-validation")
def extract_kernel_validation(
    run_id: Annotated[
        str,
        typer.Argument(help="Execution run with registered kernel-validation evidence."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded per-case numerical validation evidence."""
    try:
        result = KernelValidationExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("compute-sanitizer")
def extract_compute_sanitizer(
    run_id: Annotated[
        str,
        typer.Argument(help="Run containing an official Compute Sanitizer XML report."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded sanitizer findings through the isolated XML worker."""
    try:
        result = ComputeSanitizerExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("nvbench")
def extract_nvbench(
    run_id: Annotated[
        str,
        typer.Argument(help="Import run containing an NVBench JSON + jsonbin bundle."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract NVBench sample times and frequencies from a preserved bundle."""
    try:
        result = NvbenchExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("nsight-compute")
def extract_nsight_compute(
    run_id: Annotated[
        str,
        typer.Argument(help="Run containing an unchanged .ncu-rep or .ncu-repz report."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded metrics through NVIDIA's installed ncu_report interface."""
    try:
        result = NsightComputeExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("inference-trace")
def extract_inference_trace(
    run_id: Annotated[str, typer.Argument(help="Import run containing Mooncake JSONL.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded prompt-free request schedule evidence."""
    try:
        result = InferenceArtifactExtractor(_workspace(workspace)).extract_trace(run_id)
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@extract_app.command("inference-result")
def extract_inference_result(
    run_id: Annotated[str, typer.Argument(help="Import run containing provider result data.")],
    provider: Annotated[
        InferenceScenarioProvider,
        typer.Option("--provider", help="Maintained provider artifact schema."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract prompt-free AIPerf request rows or vLLM aggregate measurements."""
    try:
        extractor = InferenceArtifactExtractor(_workspace(workspace))
        result = (
            extractor.extract_aiperf_result(run_id)
            if provider is InferenceScenarioProvider.AIPERF
            else extractor.extract_sglang_result(run_id)
            if provider is InferenceScenarioProvider.SGLANG_BENCH
            else extractor.extract_vllm_result(run_id)
        )
    except DomainError as error:
        _fail(error, as_json=json_output)
    _emit(result, as_json=json_output)


@extract_app.command("python-startup")
def extract_python_startup(
    run_id: Annotated[
        str,
        typer.Argument(help="Run containing startup pyperf JSON and an import-time trace."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract startup samples, peak RSS, and package-grouped import costs."""
    try:
        result = PythonStartupExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("pytest")
def extract_pytest(
    run_id: Annotated[str, typer.Argument(help="Run containing pytest event JSONL.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract test phases, fixture cost, worker lifecycle, and failure latency."""
    try:
        result = PytestExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("coverage")
def extract_coverage(
    run_id: Annotated[
        str,
        typer.Argument(help="Import run containing coverage.py data."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract repository-relative line, branch, and context observations."""
    try:
        result = CoverageExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("memray")
def extract_memray(
    run_id: Annotated[str, typer.Argument(help="Import run containing Memray data.")],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract peak, retained-end, allocation, and frame evidence."""
    selected_workspace = _workspace(workspace)
    try:
        result = asyncio.run(
            MemrayExtractor(selected_workspace).extract(
                run_id,
                limits=memray_extraction_limits(selected_workspace),
            )
        )
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("perfetto")
def extract_perfetto(
    run_id: Annotated[
        str,
        typer.Argument(help="Import run containing a Perfetto-compatible trace."),
    ],
    artifact_id: Annotated[
        str | None,
        typer.Option("--artifact-id", help="Exact trace artifact for a multi-cycle run."),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract curated slice aggregates with a pinned local Trace Processor."""
    try:

        async def run() -> BaseModel:
            return await PerfettoExtractor(_workspace(workspace)).extract(
                run_id,
                artifact_id=artifact_id,
            )

        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("nsight-systems")
def extract_nsight_systems(
    run_id: Annotated[
        str,
        typer.Argument(help="Import run containing an official Nsight Systems SQLite export."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract a maintained, versioned subset of an Nsight Systems SQLite export."""
    try:

        async def run() -> BaseModel:
            return await NsightSystemsExtractor(_workspace(workspace)).extract(run_id)

        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("observations")
def extract_observations(
    run_id: Annotated[
        str,
        typer.Argument(help="Run containing flameox SDK observations."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract bounded semantic annotations emitted by flameox.sdk."""
    try:
        result = ObservationExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@mcp_app.command("serve")
def mcp_serve(
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Fixed project root exposed to MCP."),
    ] = Path("."),
    initialize: Annotated[
        bool,
        typer.Option("--init", help="Initialize .diagnostics before protocol startup."),
    ] = False,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Explicit workspace root; keeps the project root as the workload root.",
        ),
    ] = None,
) -> None:
    """Serve flameox over stdio; stdout is reserved for protocol messages."""
    run_server(project_root, initialize=initialize, workspace_root=workspace)


@mcp_app.command("inspect")
def mcp_inspect(
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Fixed project root exposed to MCP."),
    ] = Path("."),
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Explicit workspace root used by the inspected server."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """List the schemas and annotations exposed by the MCP adapter."""

    async def inspect_server() -> dict[str, Any]:
        async with Client(
            create_server(project_root, workspace_root=workspace),
            raise_exceptions=True,
        ) as client:
            tools = await client.list_tools()
            resources = await client.list_resource_templates()
            instructions = client.instructions
        return {
            "schema_version": 1,
            "instructions": instructions,
            "tools": [tool.model_dump(mode="json") for tool in tools.tools],
            "resource_templates": [
                resource.model_dump(mode="json") for resource in resources.resource_templates
            ],
        }

    _emit(_run_async(inspect_server), as_json=json_output)
