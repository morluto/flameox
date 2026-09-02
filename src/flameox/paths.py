from __future__ import annotations

import os
import sys
from pathlib import Path


def default_data_directory() -> Path:
    """Return the user-owned Flameox data directory without creating it."""

    configured = os.environ.get("FLAMEOX_DATA_DIR")
    if configured:
        return Path(configured).expanduser().absolute()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "flameox"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "flameox"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "flameox"
