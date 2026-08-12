from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel

_MAX_REQUEST_BYTES = 4 * 1024 * 1024
type WorkerPayload = dict[str, object]
type WorkerHandler = Callable[[WorkerPayload, Path], WorkerPayload]


class WorkerSuccess(ContractModel):
    model_config = ConfigDict(extra="allow", frozen=True, validate_default=True)

    ok: Literal[True]
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    def payload(self) -> dict[str, JsonValue]:
        return {"ok": self.ok, **self.__pydantic_extra__}


class WorkerFailure(ContractModel):
    ok: Literal[False]
    code: ErrorCode
    message: str


type WorkerResponse = Annotated[WorkerSuccess | WorkerFailure, Field(discriminator="ok")]
WORKER_RESPONSE: TypeAdapter[WorkerResponse] = TypeAdapter(WorkerResponse)


def run_worker(
    handler: WorkerHandler,
    *,
    invalid_code: ErrorCode,
    invalid_message: str,
    caught: tuple[type[BaseException], ...],
) -> int:
    """Run the sole bounded child-side request/response protocol."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        raw = arguments.request.read_bytes()
        if len(raw) > _MAX_REQUEST_BYTES:
            raise ValueError("request exceeds the worker protocol limit")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        response = handler(request, arguments.request)
    except DomainError as exc:
        response = {"ok": False, "code": exc.code.value, "message": exc.message}
    except caught as exc:
        response = {
            "ok": False,
            "code": invalid_code.value,
            "message": f"{invalid_message}: {type(exc).__name__}: {exc}",
        }
    _write_response(arguments.response, response)
    return 0


def _write_response(path: Path, payload: WorkerPayload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
