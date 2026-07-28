from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import questionary
import typer
from questionary import Choice

from flameox.adapters.client_setup import ALL_SETUP_CLIENTS, SetupClient
from flameox.application.setup import SetupInspection, SetupPlan, SetupReport
from flameox.domain import DomainError, ErrorCode


def print_banner(inspection: SetupInspection) -> None:
    typer.echo()
    typer.secho("  flameox setup", bold=True)
    typer.echo("  Connect local coding agents to a versioned flameox MCP runtime.")
    typer.echo()
    if inspection.active_version is not None:
        clients = ", ".join(client.display_name for client in inspection.configured_clients)
        typer.echo(f"  Active runtime: {inspection.active_version}")
        typer.echo(f"  Connected: {clients or 'none'}")
        typer.echo()


def choose_action(inspection: SetupInspection, target_version: str) -> str:
    choices: list[Choice] = []
    if (
        inspection.active_version is not None
        and inspection.active_version != target_version
        and inspection.configured_clients
    ):
        choices.append(Choice(f"Update flameox to {target_version}", "update"))
    choices.extend(
        (
            Choice("Connect or update MCP clients", "configure"),
            Choice("Disconnect MCP clients", "remove"),
            Choice("Verify connected clients and the active runtime", "verify"),
        )
    )
    if len(inspection.installed_versions) > 1 and inspection.configured_clients:
        choices.append(Choice("Roll back to an installed version", "rollback"))
    choices.append(Choice("Exit", "exit"))
    answer = questionary.select("What would you like to do?", choices=choices).ask()
    return cast(str, _answer(answer))


def choose_clients(
    inspection: SetupInspection,
    *,
    remove: bool,
) -> tuple[SetupClient, ...]:
    selected = set(inspection.configured_clients if remove else ())
    detected = set(inspection.detected_clients)
    choices = [
        Choice(
            title=_client_label(client, detected=client in detected),
            value=client,
            checked=client in selected,
        )
        for client in ALL_SETUP_CLIENTS
    ]
    answer = questionary.checkbox(
        "Select MCP clients to disconnect:" if remove else "Select MCP clients to connect:",
        choices=choices,
    ).ask()
    values = _answer(answer)
    return tuple(client for client in ALL_SETUP_CLIENTS if client in values)


def choose_rollback_version(
    inspection: SetupInspection,
    current_version: str | None,
) -> str:
    choices = [version for version in inspection.installed_versions if version != current_version]
    answer = questionary.select("Select an installed runtime:", choices=choices).ask()
    return str(_answer(answer))


def print_plan(plan: SetupPlan) -> None:
    typer.echo()
    typer.secho("Planned changes", bold=True)
    if plan.runtime_executable is not None:
        typer.echo(f"  Runtime: {plan.runtime_action.value} {plan.version}")
        typer.echo(f"  Launcher: {plan.runtime_executable}")
    for client in plan.clients:
        detected = "" if client.detected else " (not detected)"
        if plan.operation.value == "verify":
            status = (
                "matches active runtime"
                if client.action.value == "already_current"
                else "does not match active runtime"
            )
            typer.echo(f"  {client.display_name}: {status} {client.path}{detected}")
        else:
            typer.echo(f"  {client.display_name}: {client.action.value} {client.path}{detected}")
    for warning in plan.warnings:
        typer.secho(f"  Warning: {warning}", fg=typer.colors.YELLOW)
    typer.echo()


def confirm_apply() -> bool:
    answer = questionary.confirm("Apply these changes?", default=False).ask()
    return bool(_answer(answer))


def print_report(report: SetupReport) -> None:
    typer.echo()
    if report.operation.value == "verify":
        typer.secho("flameox MCP runtime verified.", fg=typer.colors.GREEN)
        clients = ", ".join(client.display_name for client in report.unchanged_clients)
        if clients:
            typer.echo(f"Configured launchers verified: {clients}")
        else:
            typer.echo("No MCP clients are currently connected.")
        return
    changed = ", ".join(client.display_name for client in report.changed_clients)
    if changed:
        verb = "Disconnected" if report.operation.value == "remove" else "Updated"
        typer.secho(f"{verb}: {changed}", fg=typer.colors.GREEN)
    else:
        typer.secho("Everything is already current.", fg=typer.colors.GREEN)
    typer.echo("Restart connected clients so they reload their MCP configuration.")


def _client_label(client: SetupClient, *, detected: bool) -> str:
    return f"{client.display_name}{'  detected' if detected else ''}"


def _answer[T](value: T | None) -> T:
    if value is None:
        raise DomainError(ErrorCode.PROCESS_CANCELLED, "Setup cancelled.")
    return value


def selected_clients(flags: Iterable[tuple[SetupClient, bool]]) -> tuple[SetupClient, ...]:
    selected = {client for client, enabled in flags if enabled}
    return tuple(client for client in ALL_SETUP_CLIENTS if client in selected)
