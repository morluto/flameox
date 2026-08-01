import json
from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from flameox.cli import app


def test_help_lists_flameoxs_purpose() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "runtime evidence" in result.stdout


def test_capture_help_uses_named_workload_syntax() -> None:
    runner = CliRunner()

    plan = runner.invoke(app, ["capture", "plan", "--help"])
    run = runner.invoke(app, ["capture", "run", "--help"])

    assert plan.exit_code == 0, plan.output
    assert run.exit_code == 0, run.output
    plan_help = unstyle(plan.stdout)
    run_help = unstyle(run.stdout)
    assert "--workload" in plan_help
    assert "--workload" in run_help
    assert "argv" not in plan_help.lower()
    assert "argv" not in run_help.lower()


def test_workload_help_has_no_inert_approval_command() -> None:
    result = CliRunner().invoke(app, ["workload", "--help"])

    assert result.exit_code == 0, result.output
    assert "approve" not in result.stdout.lower()


def test_init_import_and_list_run_as_json(tmp_path: Path) -> None:
    runner = CliRunner()
    project = tmp_path / "project"
    project.mkdir()
    workspace = project / ".diagnostics"
    artifact = tmp_path / "benchmark.json"
    artifact.write_text('{"benchmarks": []}')

    initialized = runner.invoke(app, ["init", str(project), "--json"])
    rebuilt = runner.invoke(
        app,
        ["catalog", "rebuild", "--workspace", str(workspace), "--json"],
    )
    imported = runner.invoke(
        app,
        [
            "import",
            str(artifact),
            "--kind",
            "benchmark_samples",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    listed = runner.invoke(
        app,
        ["runs", "list", "--workspace", str(workspace), "--json"],
    )

    assert initialized.exit_code == 0, initialized.output
    assert rebuilt.exit_code == 0, rebuilt.output
    assert imported.exit_code == 0, imported.output
    assert listed.exit_code == 0, listed.output
    imported_payload = json.loads(imported.stdout)
    listed_payload = json.loads(listed.stdout)
    assert listed_payload["runs"][0]["run_id"] == imported_payload["run"]["run_id"]
    assert listed_payload["corpus_commit_id"] == imported_payload["corpus_commit_id"]


def test_global_workspace_project_json_and_quiet_options(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    workspace = project / ".diagnostics"
    runner = CliRunner()
    assert runner.invoke(app, ["init", str(project)]).exit_code == 0

    structured = runner.invoke(
        app,
        [
            "--workspace",
            str(workspace),
            "--project-root",
            str(project),
            "--json",
            "status",
        ],
    )
    quiet = runner.invoke(
        app,
        ["--workspace", str(workspace), "--quiet", "status"],
    )

    assert structured.exit_code == 0, structured.output
    assert json.loads(structured.stdout)["workspace_id"]
    assert quiet.exit_code == 0
    assert quiet.stdout == ""


def test_global_project_root_initializes_and_invalid_request_exits_two(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runner = CliRunner()

    initialized = runner.invoke(
        app,
        ["--project-root", str(project), "init"],
    )
    invalid = runner.invoke(
        app,
        [
            "--workspace",
            str(project / ".diagnostics"),
            "investigations",
            "create",
            "{invalid",
        ],
    )

    assert initialized.exit_code == 0, initialized.output
    assert (project / ".diagnostics" / "workspace.json").is_file()
    assert invalid.exit_code == 2
    assert "Structured input is invalid" in invalid.output
