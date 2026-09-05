"""Thin command-line mirrors of the stateless application capabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import anyio
import typer
from pydantic import ValidationError

from flameox import __version__
from flameox.mcp import create_server, run_server
from flameox.runtime_contracts import CaptureTarget, ExperimentDesign, PathSource, RuntimeFailure
from flameox.setup import (
    DEFAULT_PREPARATION_TIMEOUT_SECONDS,
    MAX_PREPARATION_TIMEOUT_SECONDS,
    SETUP_CLIENTS,
    ClientSetupPlan,
    CliVersionAdvisory,
    ProviderPreparation,
    SetupClient,
    SetupFailure,
    apply_client_setup,
    detect_setup_clients,
    external_provider_requirements,
    mcp_launcher,
    parse_setup_clients,
    path_cli_version_advisory,
    plan_client_setup,
    prepare_providers,
)
from flameox.stateless import AnalysisRuntime

app = typer.Typer(
    name="flameox",
    help="Collect and query bounded local runtime evidence.",
    no_args_is_help=True,
)
mcp_app = typer.Typer(help="Serve or inspect the local MCP adapter.")
evidence_app = typer.Typer(help="Query or show optional preserved evidence.")
app.add_typer(mcp_app, name="mcp")
app.add_typer(evidence_app, name="evidence")


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = None,
) -> None:
    """Run Flameox without creating a workspace or repository."""


def _runtime() -> AnalysisRuntime:
    return AnalysisRuntime()


def _json_object(value: str, *, option: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"invalid JSON: {error.msg}", param_hint=option) from error
    if not isinstance(parsed, dict):
        raise typer.BadParameter("value must decode to an object", param_hint=option)
    return cast(dict[str, Any], parsed)


def _write(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _cli_failure(error: RuntimeFailure | ValidationError) -> NoReturn:
    if isinstance(error, ValidationError):
        value = {"code": "INVALID_INPUT", "message": str(error), "details": {}}
    else:
        value = {"code": error.code, "message": error.message, "details": error.details}
    typer.echo(json.dumps(value, sort_keys=True), err=True)
    raise typer.Exit(code=1)


@app.command("setup")
def setup(
    provider: Annotated[
        list[str] | None,
        typer.Option("--provider", help="Include a managed or host profiler provider."),
    ] = None,
    client: Annotated[
        list[str] | None,
        typer.Option("--client", help="Configure one global MCP client; repeatable."),
    ] = None,
    all_clients: Annotated[bool, typer.Option("--all", help="Configure every client.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Apply explicit selections.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show resolved global changes without writing.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit structured output.")] = False,
    timeout_seconds: Annotated[
        int,
        typer.Option("--timeout-seconds", min=1, max=MAX_PREPARATION_TIMEOUT_SECONDS),
    ] = DEFAULT_PREPARATION_TIMEOUT_SECONDS,
) -> None:
    """Configure global MCP clients and optionally prepare provider extras."""
    selected = provider or []
    try:
        explicit_clients = parse_setup_clients(client or [])
        if all_clients and explicit_clients:
            raise SetupFailure("--all cannot be combined with --client.")
        interactive = _is_interactive()
        if yes and not (all_clients or explicit_clients):
            raise SetupFailure("--yes requires --client or --all; detection is not consent.")
        if dry_run and not (all_clients or explicit_clients) and (not interactive or json_output):
            raise SetupFailure("--dry-run requires --client or --all.")

        selected_clients = list(SETUP_CLIENTS) if all_clients else explicit_clients
        if not selected_clients:
            if not interactive or json_output:
                raise SetupFailure(
                    "No MCP client selected. Run setup in an interactive terminal, or pass "
                    "--client <name> --yes (for example, --client codex --yes)."
                )
            selected_clients = _select_setup_clients(detect_setup_clients())
            if not selected_clients:
                typer.echo("Flameox setup cancelled. No changes were made.")
                return
        elif not interactive and not (yes or dry_run):
            raise SetupFailure("Non-interactive setup requires --yes or --dry-run.")

        plans = plan_client_setup(selected_clients, selected)
        advisory = path_cli_version_advisory()
        if dry_run:
            value = _setup_value(plans, selected, preparation=None, advisory=advisory, dry_run=True)
            if json_output:
                _write(value)
            else:
                _write_setup_plan(plans)
                _write_external_guidance(value)
                _write_advisories(value)
            return

        preparation = prepare_providers(selected, timeout_seconds)
        results = apply_client_setup(plans)
    except SetupFailure as error:
        raise typer.BadParameter(str(error)) from error

    value = _setup_value(plans, selected, preparation=preparation, advisory=advisory, dry_run=False)
    value["clients"] = [
        {
            "id": result.client.value,
            "name": result.client.display_name,
            "path": str(result.path),
            "status": result.action,
        }
        for result in results
    ]
    value["client_registration_changed"] = any(
        result.action in {"created", "updated"} for result in results
    )
    restart_clients = [
        result.client.display_name for result in results if result.action != "already_current"
    ]
    value["restart_required"] = bool(restart_clients)
    value["next_action"] = (
        {
            "kind": "reconnect_mcp",
            "clients": restart_clients,
            "message": f"Restart or reconnect {', '.join(restart_clients)} to load Flameox.",
        }
        if restart_clients
        else None
    )
    if json_output:
        _write(value)
        return
    typer.echo("◆ Flameox")
    for result in results:
        label = {
            "created": "configured",
            "updated": "updated",
            "already_current": "already configured",
        }[result.action]
        typer.echo(f"  ✓ {result.client.display_name} {label}\n    {result.path}")
    if restart_clients:
        typer.echo(f"\nRestart or reconnect {', '.join(restart_clients)} to load Flameox.")
    _write_external_guidance(value)
    _write_advisories(value)


def _write_external_guidance(value: dict[str, object]) -> None:
    guidance = cast(list[str], value["external_guidance"])
    if guidance:
        typer.echo()
        typer.echo("\n".join(guidance))


def _write_advisories(value: dict[str, object]) -> None:
    advisories = cast(list[dict[str, str]], value["advisories"])
    if advisories:
        typer.echo()
        typer.echo("\n".join(f"Warning: {item['message']}" for item in advisories))


def _select_setup_clients(detected: list[SetupClient]) -> list[SetupClient]:
    import questionary

    typer.echo("◆ Flameox\n  Runtime evidence for coding agents\n")
    choices = [
        questionary.Choice(
            title=_setup_client_choice_title(item, item in detected),
            value=item.value,
            checked=item in detected,
        )
        for item in SETUP_CLIENTS
    ]
    selected = questionary.checkbox(
        "Which agents should use Flameox?",
        choices=choices,
        instruction="(space to select, enter to configure or update)",
        validate=lambda values: bool(values) or "Select at least one agent.",
    ).ask()
    return parse_setup_clients(selected or [])


def _setup_client_choice_title(client: SetupClient, detected: bool) -> str:
    path = _display_setup_path(client.active_config_path(Path.home()))
    marker = " — detected" if detected else ""
    return f"{client.display_name}{marker} · {path}"


def _display_setup_path(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _write_setup_plan(plans: list[ClientSetupPlan]) -> None:
    typer.echo("◆ Flameox dry-run\n  No changes were made.")
    for plan in plans:
        typer.echo(f"  ◇ {plan.client.display_name}: {plan.action}\n    {plan.path}")


def _setup_value(
    plans: list[ClientSetupPlan],
    providers: list[str],
    *,
    preparation: ProviderPreparation | None,
    advisory: CliVersionAdvisory | None,
    dry_run: bool,
) -> dict[str, object]:
    if preparation is None:
        launcher, launcher_args = mcp_launcher(providers)
        args = [*launcher_args, "mcp", "serve"]
        preparation_command: list[str] = []
        guidance = [item.guidance for item in external_provider_requirements(providers)]
    else:
        launcher = preparation.launcher_command
        args = preparation.launcher_args
        preparation_command = preparation.preparation_command
        guidance = [item.guidance for item in preparation.external_requirements]
    return {
        "command": launcher,
        "args": args,
        "resolved_version": __version__,
        "client_registration_changed": False,
        "repository_created": False,
        "providers": sorted(set(providers)),
        "preparation_command": preparation_command,
        "external_guidance": guidance,
        "advisories": (
            [
                {
                    "kind": "path_cli_version_mismatch",
                    "executable": advisory.executable,
                    "cli_version": advisory.cli_version,
                    "mcp_version": advisory.mcp_version,
                    "message": advisory.message,
                }
            ]
            if advisory is not None
            else []
        ),
        "dry_run": dry_run,
        "plan": [
            {
                "client": plan.client.value,
                "name": plan.client.display_name,
                "path": str(plan.path),
                "action": plan.action,
                "detected": plan.detected,
            }
            for plan in plans
        ],
    }


@app.command("analyze")
def analyze(
    capability_id: Annotated[str, typer.Argument()],
    sources: Annotated[list[Path], typer.Argument()],
    arguments: Annotated[str, typer.Option("--arguments")] = "{}",
    format_name: Annotated[str | None, typer.Option("--format")] = None,
    continuation: Annotated[str | None, typer.Option("--continuation")] = None,
    preserve: Annotated[bool, typer.Option("--preserve")] = False,
) -> None:
    """Analyze explicit artifacts and optionally preserve the result."""
    runtime = _runtime()
    try:
        result = runtime.analyze(
            capability_id,
            [PathSource(path=str(path.resolve()), format=format_name) for path in sources],
            _json_object(arguments, option="--arguments"),
            continuation=continuation,
        )
        if preserve:
            result["preserved"] = runtime.preserve_evidence(str(result["analysis_id"]))
        _write(result)
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@app.command("capture", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def capture(
    ctx: typer.Context,
    provider_id: Annotated[str, typer.Option("--provider")],
    capability_id: Annotated[str, typer.Option("--capability")] = "artifact.preview",
    cwd: Annotated[Path, typer.Option("--cwd")] = Path("."),
    capture_arguments: Annotated[str, typer.Option("--capture-arguments")] = "{}",
    analysis_arguments: Annotated[str, typer.Option("--analysis-arguments")] = "{}",
    experiment_json: Annotated[
        str | None,
        typer.Option(
            "--experiment",
            help=(
                "JSON experiment with cases, blocks, seed, metric, estimand, threshold, and "
                "optional oracle; the argv after -- is the default target."
            ),
        ),
    ] = None,
    preserve: Annotated[bool, typer.Option("--preserve")] = False,
) -> None:
    """Capture a typed argv target after `--` and immediately analyze its output."""
    argv = list(ctx.args)
    if argv and argv[0] == "--":
        argv.pop(0)
    if not argv:
        raise typer.BadParameter("capture requires an argv after --")
    try:
        experiment = (
            ExperimentDesign.model_validate(_json_object(experiment_json, option="--experiment"))
            if experiment_json is not None
            else None
        )
    except ValidationError as error:
        _cli_failure(error)
    runtime = _runtime()

    async def execute() -> dict[str, Any]:
        return await runtime.capture_and_analyze(
            CaptureTarget(
                argv=argv,
                cwd=str(cwd.resolve(strict=True)),
                provider_id=provider_id,
                capture_arguments=_json_object(capture_arguments, option="--capture-arguments"),
                analysis_arguments=_json_object(analysis_arguments, option="--analysis-arguments"),
            ),
            capability_id,
            mode="experiment" if experiment is not None else "single",
            experiment=experiment,
            preserve=preserve,
        )

    try:
        result = anyio.run(execute)
        if preserve and "preserved" not in result:
            result["preserved"] = runtime.preserve_evidence(str(result["analysis_id"]))
        _write(result)
        if result.get("analysis_failure") is not None or any(
            item["status"] != "succeeded" for item in result["capture"]["executions"]
        ):
            raise typer.Exit(code=1)
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@evidence_app.command("query")
def evidence_query(
    capability_id: Annotated[str | None, typer.Option("--capability")] = None,
    provider_id: Annotated[str | None, typer.Option("--provider")] = None,
    input_sha256: Annotated[str | None, typer.Option("--input-sha256")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Search immutable evidence manifests."""
    runtime = _runtime()
    try:
        _write(
            runtime.query_evidence(
                capability_id=capability_id,
                provider_id=provider_id,
                input_sha256=input_sha256,
                limit=limit,
                cursor=cursor,
            )
        )
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@evidence_app.command("location")
def evidence_location() -> None:
    """Show the selected local evidence directory without opening or modifying it."""
    runtime = _runtime()
    try:
        _write(
            {"directory": str(runtime.repository.root), "environment_variable": "FLAMEOX_DATA_DIR"}
        )
    finally:
        runtime.close()


@evidence_app.command("show")
def evidence_show(
    evidence_id: Annotated[str, typer.Argument()],
) -> None:
    """Read one canonical immutable evidence manifest."""
    runtime = _runtime()
    try:
        _write(runtime.read_evidence(evidence_id))
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@mcp_app.command("serve")
def mcp_serve() -> None:
    """Serve composable Flameox evidence tools over stdio."""
    run_server()


@mcp_app.command("inspect")
def mcp_inspect() -> None:
    """Print the exact MCP catalog without starting a transport."""

    async def inspect_server() -> dict[str, Any]:
        server = create_server()
        return {
            "tools": [item.model_dump(mode="json") for item in await server.list_tools()],
            "resources": [
                item.model_dump(mode="json") for item in await server.list_resource_templates()
            ],
        }

    _write(anyio.run(inspect_server))
