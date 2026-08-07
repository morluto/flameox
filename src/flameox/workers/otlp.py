from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from flameox.application.otlp import _OtlpRowLimitExceeded, _parse_otlp
from flameox.domain import DomainError, ErrorCode


def _write_response(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        request = json.loads(arguments.request.read_text())
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        parsed = _parse_otlp(
            Path(str(request["artifact_path"])),
            str(request["media_type"]),
            row_limit=int(request["row_limit"]),
        )
        response: dict[str, object] = {
            "ok": True,
            "resources": parsed.resources,
            "scopes": parsed.scopes,
            "spans": parsed.spans,
            "events": parsed.events,
            "links": parsed.links,
            "limitations": list(parsed.limitations),
        }
    except _OtlpRowLimitExceeded as exc:
        response = {
            "ok": True,
            "row_limit_exceeded": True,
            "counts": exc.counts,
            "limitations": list(exc.limitations),
        }
    except DomainError as exc:
        response = {"ok": False, "code": exc.code.value, "message": exc.message}
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        response = {
            "ok": False,
            "code": ErrorCode.ARTIFACT_PARSE_FAILED.value,
            "message": f"OTLP worker request is invalid: {exc}",
        }
    _write_response(arguments.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
