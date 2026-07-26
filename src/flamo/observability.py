from __future__ import annotations

import json
import re
import time
from pathlib import Path
from uuid import uuid4

import portalocker
from pydantic import Field

from flamo.domain.models import utc_now
from flamo.models import ContractModel

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]"),
    (r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", "[REDACTED_IPV6]"),
]


def sanitize(value: str) -> str:
    """Redact email addresses and IP addresses from a string value."""
    for pattern, replacement in _PII_PATTERNS:
        value = re.sub(pattern, replacement, value)
    return value


class OperationEvent(ContractModel):
    """Bounded operational metadata; artifact and process payloads are forbidden."""

    schema_version: int = 1
    timestamp: str
    operation_id: str
    operation: str = Field(min_length=1, max_length=128)
    phase: str = Field(min_length=1, max_length=256)
    run_id: str | None = Field(default=None, max_length=128)
    adapter: str | None = Field(default=None, max_length=128)
    elapsed_ms: float | None = Field(default=None, ge=0)
    lock_wait_ms: float | None = Field(default=None, ge=0)
    query_name: str | None = Field(default=None, max_length=128)
    query_duration_ms: float | None = Field(default=None, ge=0)
    rows_returned: int | None = Field(default=None, ge=0)
    bytes_returned: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=128)


class OperationLogger:
    def __init__(self, root: Path) -> None:
        self.path = root / "logs" / "operations.jsonl"
        self.lock_path = root / "logs" / "operations.lock"

    def new_id(self) -> str:
        return str(uuid4())

    def emit(
        self,
        *,
        operation_id: str,
        operation: str,
        phase: str,
        run_id: str | None = None,
        adapter: str | None = None,
        elapsed_ms: float | None = None,
        lock_wait_ms: float | None = None,
        query_name: str | None = None,
        query_duration_ms: float | None = None,
        rows_returned: int | None = None,
        bytes_returned: int | None = None,
        error_code: str | None = None,
    ) -> bool:
        event = OperationEvent(
            timestamp=utc_now().isoformat(),
            operation_id=operation_id,
            operation=sanitize(operation),
            phase=sanitize(phase),
            run_id=run_id,
            adapter=adapter,
            elapsed_ms=elapsed_ms,
            lock_wait_ms=lock_wait_ms,
            query_name=query_name,
            query_duration_ms=query_duration_ms,
            rows_returned=rows_returned,
            bytes_returned=bytes_returned,
            error_code=error_code,
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                event.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            with (
                portalocker.Lock(self.lock_path, mode="a", timeout=5),
                self.path.open("a", encoding="utf-8") as stream,
            ):
                stream.write(payload + "\n")
                stream.flush()
        except (OSError, portalocker.exceptions.LockException):
            return False
        return True


def elapsed_ms(started: float) -> float:
    return max(0.0, (time.monotonic() - started) * 1_000)
