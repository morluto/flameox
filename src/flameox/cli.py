from __future__ import annotations

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
from pydantic import BaseModel

from flameox import __version__, setup_ui
from flameox.adapters import (
    AdapterRegistry,
    CoverageExtractor,
    MemrayExtractor,
    ObservationExtractor,
    PerfettoExtractor,
    PyPerfExtractor,
    PytestExtractor,
    PythonStartupExtractor,
    SetupClient,
)
from flameox.analysis import RecipeService
from flameox.application import (
    AnalysisMaterializationService,
    ArtifactService,
    CapabilityService,
    CaptureService,
    CompactionService,
    CompareRunSetsRequest,
    ComparisonService,
    CreateInvestigationRequest,
    DrilldownService,
    EvidenceLookupService,
    EvidenceQueryService,
    EvidenceSummaryRequest,
    EvidenceSummaryService,
    ExecutionPolicy,
    ExperimentService,
    FindingService,
    FreezeRunSetRequest,
    GarbageCollector,
    ImportArtifactRequest,
    ImportService,
    IntegrityService,
    InvestigationService,
    MaterializeAnalysisRequest,
    NativeViewerService,
    QuarantineService,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RecoveryService,
    RepairPlan,
    RepairService,
    RunSetService,
    SetupOperation,
    SetupService,
    WorkloadService,
    workspace_status,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode, Sensitivity
from flameox.mcp import create_server, run_server
from flameox.storage import RunStore, Workspace

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
stacks_app = typer.Typer(help="Inspect bounded call relationships and stacks.")
trace_app = typer.Typer(help="Inspect bounded temporal trace windows.")
adapters_app = typer.Typer(help="Discover and approve third-party adapter entry points.")
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
app.add_typer(stacks_app, name="stacks")
app.add_typer(trace_app, name="trace")
app.add_typer(adapters_app, name="adapters")

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
    try:
        if value.startswith("@"):
            return model.model_validate_json(Path(value[1:]).read_text())
        return model.model_validate_json(value)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Structured input is invalid: {exc}") from exc


def _fail(error: DomainError) -> NoReturn:
    typer.echo(f"{error.code.value}: {_terminal_text(error.message)}", err=True)
    for remediation in error.remediation:
        typer.echo(f"  {_terminal_text(remediation)}", err=True)
    exit_codes = {
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
            workspace_root=workspace,
        )
        result = workspace_status(initialized)
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
        _fail(error)
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
        _fail(error)
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
        _fail(error)
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
        _fail(error)
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
        result = IntegrityService(_workspace(workspace)).validate(full=full)
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


@app.command("repair")
def repair(
    plan_path: Annotated[
        Path | None,
        typer.Argument(
            help="Validated repair-plan JSON to apply; omit to preview a plan.",
        ),
    ] = None,
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Preview safe repairs, or apply one exact structured plan."""
    try:
        service = RepairService(_workspace(workspace))
        if plan_path is None:
            result: BaseModel = service.plan()
        else:
            try:
                plan = RepairPlan.model_validate_json(plan_path.read_text())
            except (OSError, ValueError) as exc:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Repair plan is unreadable or invalid: {plan_path}",
                ) from exc
            result = service.apply(plan)
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
                allow_external_path=True,
            )
        )
    except DomainError as error:
        _fail(error)
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
        return await service.execute(plan.plan_id)

    try:
        result = _run_async(run)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


def _parameter_overrides(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Invalid parameter overrides: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Parameter overrides must be a JSON object.",
        )
    return cast(dict[str, Any], parsed)


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
        return await service.run(plan.plan_id)

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


@capture_app.command("plan")
def capture_plan(
    adapter: Annotated[str, typer.Argument(help="Registered capture adapter.")],
    workload_name: Annotated[str, typer.Option("--workload")],
    parameters: Annotated[
        str,
        typer.Option("--parameters", help="JSON object of declared scalar overrides."),
    ] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Resolve a side-effect-free local capture plan."""

    async def plan() -> BaseModel:
        try:
            values = json.loads(parameters)
            if not isinstance(values, dict):
                raise ValueError("parameters must be a JSON object")
            return await CaptureService(_workspace(workspace)).plan(
                workload_name=workload_name,
                adapter=adapter,
                parameters=values,
                execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                f"Invalid parameter overrides: {exc}",
            ) from exc

    try:
        result = _run_async(plan)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@capture_app.command("run")
def capture_run(
    adapter: Annotated[str, typer.Argument(help="Registered capture adapter.")],
    workload_name: Annotated[str, typer.Option("--workload")],
    parameters: Annotated[str, typer.Option("--parameters")] = "{}",
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Plan and execute one named workload capture."""

    async def run() -> BaseModel:
        values = json.loads(parameters)
        if not isinstance(values, dict):
            raise ValueError("parameters must be a JSON object")
        service = CaptureService(_workspace(workspace))
        plan = await service.plan(
            workload_name=workload_name,
            adapter=adapter,
            parameters=values,
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        return await service.execute(plan.plan_id)

    try:
        result = _run_async(run)
    except (json.JSONDecodeError, ValueError) as exc:
        _fail(
            DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                f"Invalid parameter overrides: {exc}",
            )
        )
    except DomainError as error:
        _fail(error)
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
        result = CompactionService(_workspace(workspace)).compact()
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@catalog_app.command("status")
def catalog_status(
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show catalog version and freshness."""
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
    """Validate catalog freshness and corpus inventory schemas."""
    try:
        selected = _workspace(workspace)
        integrity = IntegrityService(selected).validate(full=False)
        result = {
            "schema_version": 1,
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
        catalog = Catalog(_workspace(workspace))
        with catalog.open_snapshot() as snapshot:
            rows = snapshot.execute(
                "SELECT run_id, created_at, run_type, capture_status "
                "FROM runs ORDER BY created_at DESC, run_id LIMIT ?",
                (limit,),
            ).fetchall()
            result = {
                "schema_version": 1,
                "corpus_commit_id": snapshot.commit.commit_id,
                "runs": [
                    {
                        "run_id": row[0],
                        "created_at": row[1].isoformat(),
                        "run_type": row[2],
                        "capture_status": row[3],
                    }
                    for row in rows
                ],
                "returned": len(rows),
                "limit": limit,
            }
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
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Freeze a cohort against the current corpus snapshot."""
    try:
        result = RunSetService(_workspace(workspace)).freeze(
            FreezeRunSetRequest(run_ids=tuple(run_ids))
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
            _request(CompareRunSetsRequest, structured_input)
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
            _request(CompareRunSetsRequest, structured_input)
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
            _request(MaterializeAnalysisRequest, structured_input)
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
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Query slices overlapping one bounded time interval."""
    try:

        async def run() -> BaseModel:
            return await PerfettoExtractor(_workspace(workspace)).trace_window(
                artifact_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
                cursor=cursor,
            )

        result = _run_async(run)
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
    allowed = {
        "analysis",
        "artifact",
        "comparison",
        "generation",
        "observation",
        "run",
        "run_set",
        "trial",
    }
    if ref_type not in allowed:
        _fail(
            DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unsupported evidence reference type {ref_type!r}.",
            )
        )
    try:
        result = EvidenceLookupService(_workspace(workspace)).get(
            cast(
                Literal[
                    "analysis",
                    "artifact",
                    "comparison",
                    "generation",
                    "observation",
                    "run",
                    "run_set",
                    "trial",
                ],
                ref_type,
            ),
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
        Literal["none", "internal"],
        typer.Option("--output-excerpts"),
    ] = "none",
    sensitive_context: Annotated[
        Literal["redact", "include"],
        typer.Option("--sensitive-context"),
    ] = "redact",
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


@extract_app.command("python-startup")
def extract_python_startup(
    run_id: Annotated[str, typer.Argument(help="Run containing Python startup JSON.")],
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
    try:
        result = MemrayExtractor(_workspace(workspace)).extract(run_id)
    except DomainError as error:
        _fail(error)
    _emit(result, as_json=json_output)


@extract_app.command("perfetto")
def extract_perfetto(
    run_id: Annotated[
        str,
        typer.Argument(help="Import run containing a Perfetto-compatible trace."),
    ],
    workspace: WorkspaceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Extract curated slice aggregates with a pinned local Trace Processor."""
    try:

        async def run() -> BaseModel:
            return await PerfettoExtractor(_workspace(workspace)).extract(run_id)

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
) -> None:
    """Serve flameox over stdio; stdout is reserved for protocol messages."""
    run_server(project_root, initialize=initialize)


@mcp_app.command("inspect")
def mcp_inspect(
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Fixed project root exposed to MCP."),
    ] = Path("."),
    json_output: JsonOption = False,
) -> None:
    """List the schemas and annotations exposed by the MCP adapter."""

    async def inspect_server() -> dict[str, Any]:
        async with Client(create_server(project_root), raise_exceptions=True) as client:
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
