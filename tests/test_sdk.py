from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.cli import app


@pytest.mark.integration
def test_capture_analyzes_real_sdk_observations(tmp_path: Path) -> None:
    script = (
        "from flameox.sdk import observe, phase\n"
        "with phase('compile'):\n"
        "    observe('compiled', duration_ns=12)\n"
        "observe('finished')\n"
    )

    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "observations",
            "--capability",
            "failures.summary",
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            script,
        ],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)["blocks"][1]["rows"]
    assert [(row["name"], row["phase"]) for row in rows] == [
        ("flameox.phase.start", "compile"),
        ("compiled", "compile"),
        ("flameox.phase.end", "compile"),
        ("finished", None),
    ]
    assert rows[1]["values"] == {"duration_ns": 12}
