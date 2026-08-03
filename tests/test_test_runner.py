from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "test.py"


def test_list_reports_lanes_and_metadata_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "  golden" in result.stdout
    assert "Metadata commands:" in result.stdout
    assert "  capabilities validate managed setup metadata against extras" in result.stdout
