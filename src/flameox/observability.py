from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from concurrent_log_handler import ConcurrentRotatingFileHandler
from pydantic import Field
from pythonjsonlogger.json import JsonFormatter

from flameox.domain.models import utc_now
from flameox.models import ContractModel

_HANDLER_LOCK = Lock()
_LOGGERS: dict[Path, logging.Logger] = {}
_FIELD_NAMES = (
    "timestamp operation_id operation phase run_id adapter elapsed_ms lock_wait_ms "
    "query_name query_duration_ms rows_returned bytes_returned error_code cleanup_status"
)


def _compact_json(value: object, **kwargs: Any) -> str:
    kwargs.pop("separators", None)
    return json.dumps(value, separators=(",", ":"), **kwargs)


class _ReportingRotatingFileHandler(ConcurrentRotatingFileHandler):
    """Turn logging's swallowed I/O failures into an observable emit result."""

    def handleError(self, record: logging.LogRecord) -> None:
        raise OSError(f"could not write operation event {record.getMessage()!r}")


class OperationEvent(ContractModel):
    """Closed diagnostic metadata; lifecycle authority remains in the control plane."""

    schema_version: int = 1
    timestamp: str
    operation_id: str
    operation: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    phase: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._: -]+$")
    run_id: str | None = Field(default=None, max_length=128)
    adapter: str | None = Field(default=None, max_length=128)
    elapsed_ms: float | None = Field(default=None, ge=0)
    lock_wait_ms: float | None = Field(default=None, ge=0)
    query_name: str | None = Field(default=None, max_length=128)
    query_duration_ms: float | None = Field(default=None, ge=0)
    rows_returned: int | None = Field(default=None, ge=0)
    bytes_returned: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=128)
    cleanup_status: Literal["pending", "complete", "incomplete"] | None = None


class OperationLogger:
    """Best-effort bounded diagnostics over standard logging; never recovery authority."""

    def __init__(self, root: Path) -> None:
        self.path = root / "logs" / "operations.jsonl"

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
        cleanup_status: Literal["pending", "complete", "incomplete"] | None = None,
    ) -> bool:
        event = OperationEvent(
            timestamp=utc_now().isoformat(),
            operation_id=operation_id,
            operation=operation,
            phase=phase,
            run_id=run_id,
            adapter=adapter,
            elapsed_ms=elapsed_ms,
            lock_wait_ms=lock_wait_ms,
            query_name=query_name,
            query_duration_ms=query_duration_ms,
            rows_returned=rows_returned,
            bytes_returned=bytes_returned,
            error_code=error_code,
            cleanup_status=cleanup_status,
        )
        try:
            logger = _logger_for(self.path)
            logger.info("operation", extra=event.model_dump(mode="json"))
        except (OSError, ValueError):
            return False
        return True


def _logger_for(path: Path) -> logging.Logger:
    normalized = path.absolute()
    with _HANDLER_LOCK:
        existing = _LOGGERS.get(normalized)
        if existing is not None:
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = _ReportingRotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            JsonFormatter(
                " ".join(f"%({name})s" for name in _FIELD_NAMES.split()),
                json_serializer=_compact_json,
            )
        )
        logger = logging.getLogger(f"flameox.operation.{len(_LOGGERS)}")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _LOGGERS[normalized] = logger
        return logger


def elapsed_ms(started: float) -> float:
    return max(0.0, (time.monotonic() - started) * 1_000)
