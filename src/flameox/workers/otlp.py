from __future__ import annotations

import json
from pathlib import Path

from flameox.application.otlp import _OtlpRowLimitExceeded, _parse_otlp
from flameox.domain import ErrorCode
from flameox.workers.protocol import run_worker


def _handle(request: dict[str, object], _request_path: Path) -> dict[str, object]:
    try:
        parsed = _parse_otlp(
            Path(str(request["artifact_path"])),
            str(request["media_type"]),
            row_limit=int(str(request["row_limit"])),
        )
        return {
            "ok": True,
            "resources": parsed.resources,
            "scopes": parsed.scopes,
            "spans": parsed.spans,
            "events": parsed.events,
            "links": parsed.links,
            "limitations": list(parsed.limitations),
        }
    except _OtlpRowLimitExceeded as exc:
        return {
            "ok": True,
            "row_limit_exceeded": True,
            "counts": exc.counts,
            "limitations": list(exc.limitations),
        }


def main() -> int:
    return run_worker(
        _handle,
        invalid_code=ErrorCode.ARTIFACT_PARSE_FAILED,
        invalid_message="OTLP worker request is invalid",
        caught=(KeyError, OSError, TypeError, ValueError, json.JSONDecodeError),
    )


if __name__ == "__main__":
    raise SystemExit(main())
