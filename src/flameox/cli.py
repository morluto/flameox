"""Thin command-line mirrors of the stateless application capabilities."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import anyio
import typer
from pydantic import ValidationError

from flameox import __version__
from flameox.mcp import create_server, run_server
from flameox.runtime_contracts import CaptureTarget, PathSource, RuntimeFailure
from flameox.setup import SetupFailure, prepare_providers
from flameox.stateless import AnalysisRuntime

app = typer.Typer(
    name="flameox",
    help="Collect and query bounded local runtime evidence.",
    no_args_is_help=True,
)
capabilities_app = typer.Typer(help="Discover and inspect runtime capabilities.")
mcp_app = typer.Typer(help="Serve or inspect the local MCP adapter.")
evidence_app = typer.Typer(help="Query or show optional preserved evidence.")
app.add_typer(capabilities_app, name="capabilities")
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


def _runtime(project_root: Path) -> AnalysisRuntime:
    return AnalysisRuntime(project_root)


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
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
    provider: Annotated[list[str] | None, typer.Option("--provider")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print version-bound MCP config and optionally install selected provider extras."""
    root = project_root.resolve(strict=True)
    selected = provider or []
    try:
        preparation = prepare_providers(selected)
    except SetupFailure as error:
        raise typer.BadParameter(str(error), param_hint="--provider") from error
    configured_providers = preparation.configured_managed_providers
    external_providers = [item.provider_id for item in preparation.external_requirements]
    guidance = [item.guidance for item in preparation.external_requirements]
    launcher, launcher_args = preparation.launcher_command, preparation.launcher_args
    args = [*launcher_args, "mcp", "serve", "--project-root", str(root)]
    value = {
        "command": launcher,
        "args": args,
        "resolved_version": __version__,
        "client_registration_changed": False,
        "project_root": str(root),
        "durable_repository": str(root / ".flameox"),
        "repository_created": False,
        "providers": sorted(set(configured_providers) | set(external_providers)),
        "install_command": preparation.install_command,
        "external_guidance": guidance,
    }
    if json_output:
        _write(value)
    else:
        typer.echo(
            "No MCP client registration was changed. Configure your client to run:\n"
            f"  {shlex.join([launcher, *args])}\n\n"
            + (
                "Installed the selected Python provider extras into the persistent Flameox "
                "uv tool environment.\n"
                if preparation.changed
                else "No Python provider packages were installed.\n"
            )
            + "\n".join(guidance)
        )


@capabilities_app.command("discover")
def capabilities_discover(
    intent: Annotated[str | None, typer.Option("--intent")] = None,
    source: Annotated[list[Path] | None, typer.Option("--source")] = None,
    include_unavailable: Annotated[bool, typer.Option("--include-unavailable")] = False,
    limit: Annotated[int, typer.Option("--limit", min=1, max=50)] = 10,
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Rank capabilities for an intent and optional explicit artifact paths."""
    runtime = _runtime(project_root)
    try:
        _write(
            runtime.discover_capabilities(
                intent,
                [PathSource(path=str(path.resolve())) for path in source or []],
                include_unavailable=include_unavailable,
                limit=limit,
            )
        )
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@capabilities_app.command("inspect")
def capabilities_inspect(
    capability_ids: Annotated[list[str], typer.Argument()],
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Inspect one to sixteen capability contracts."""
    runtime = _runtime(project_root)
    try:
        _write(runtime.inspect_capabilities(capability_ids))
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@app.command("analyze")
def analyze(
    capability_id: Annotated[str, typer.Argument()],
    sources: Annotated[list[Path], typer.Argument()],
    arguments: Annotated[str, typer.Option("--arguments")] = "{}",
    format_name: Annotated[str | None, typer.Option("--format")] = None,
    continuation: Annotated[str | None, typer.Option("--continuation")] = None,
    preserve: Annotated[bool, typer.Option("--preserve")] = False,
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Analyze explicit artifacts and optionally preserve the result."""
    runtime = _runtime(project_root)
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
    cwd: Annotated[str, typer.Option("--cwd")] = ".",
    capture_arguments: Annotated[str, typer.Option("--capture-arguments")] = "{}",
    analysis_arguments: Annotated[str, typer.Option("--analysis-arguments")] = "{}",
    preserve: Annotated[bool, typer.Option("--preserve")] = False,
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Capture a typed argv target after `--` and immediately analyze its output."""
    argv = list(ctx.args)
    if argv and argv[0] == "--":
        argv.pop(0)
    if not argv:
        raise typer.BadParameter("capture requires an argv after --")
    runtime = _runtime(project_root)

    async def execute() -> dict[str, Any]:
        return await runtime.capture_and_analyze(
            CaptureTarget(
                argv=argv,
                cwd=cwd,
                provider_id=provider_id,
                capture_arguments=_json_object(capture_arguments, option="--capture-arguments"),
                analysis_arguments=_json_object(analysis_arguments, option="--analysis-arguments"),
            ),
            capability_id,
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
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Search immutable evidence manifests."""
    runtime = _runtime(project_root)
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


@evidence_app.command("show")
def evidence_show(
    evidence_id: Annotated[str, typer.Argument()],
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Read one canonical immutable evidence manifest."""
    runtime = _runtime(project_root)
    try:
        _write(runtime.read_evidence(evidence_id))
    except (RuntimeFailure, ValidationError) as error:
        _cli_failure(error)
    finally:
        runtime.close()


@mcp_app.command("serve")
def mcp_serve(
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Serve Flameox over repo-local stdio with a fixed project root."""
    run_server(project_root)


@mcp_app.command("inspect")
def mcp_inspect(
    project_root: Annotated[Path, typer.Option("--project-root")] = Path("."),
) -> None:
    """Print the exact MCP catalog without starting a transport."""

    async def inspect_server() -> dict[str, Any]:
        server = create_server(project_root)
        return {
            "tools": [item.model_dump(mode="json") for item in await server.list_tools()],
            "resources": [
                item.model_dump(mode="json") for item in await server.list_resource_templates()
            ],
        }

    _write(anyio.run(inspect_server))
