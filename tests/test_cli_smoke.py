import json
from pathlib import Path

from typer.testing import CliRunner

from flamo.cli import app


def test_help_lists_flamos_purpose() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "runtime evidence" in result.stdout


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
