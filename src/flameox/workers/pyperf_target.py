"""Metadata-safe launcher for pyperf command targets with multiline argv."""

from __future__ import annotations

import base64
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    try:
        argv = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return 2
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return 2
    os.execvpe(argv[0], argv, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
