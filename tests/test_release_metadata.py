from __future__ import annotations

import json
from pathlib import Path

from flamo import __version__


def test_npm_bootstrap_matches_python_release_version() -> None:
    package = json.loads(
        (Path(__file__).parents[1] / "npm" / "package.json").read_text()
    )

    assert package["version"] == __version__
