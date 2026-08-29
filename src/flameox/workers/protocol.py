from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel

_MAX_REQUEST_BYTES = 4 * 1024 * 1024
WORKER_TRANSPORT: Literal["flameox.artifact-worker/v1"] = "flameox.artifact-worker/v1"


class WorkerOperationId(StrEnum):
    AIPERF_PARSE = "aiperf.parse"
    COMPUTE_SANITIZER_PARSE = "compute_sanitizer.parse"
    NSIGHT_COMPUTE_PARSE = "nsight_compute.parse"
    NSIGHT_SYSTEMS_PARSE = "nsight_systems.parse"
    NVML_OBSERVE = "nvml.observe"
    OTLP_PARSE = "otlp.parse"
    PERFETTO_QUERY = "perfetto.query"
    REDUCTION_EXECUTE = "reduction.execute"
    V8_PROFILE_PARSE = "v8_profile.parse"


class WorkerFailureKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INPUT_UNAVAILABLE = "input_unavailable"
    INPUT_FORMAT_UNSUPPORTED = "input_format_unsupported"
    INPUT_MALFORMED = "input_malformed"
    ROW_OR_OUTPUT_LIMIT = "row_or_output_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_INCOMPATIBLE = "provider_incompatible"
    WORKER_INTERNAL_ERROR = "worker_internal_error"
    OUTPUT_INVALID = "output_invalid"


_FAILURE_ERROR_CODES = {
    WorkerFailureKind.INVALID_REQUEST: ErrorCode.WORKSPACE_INVALID,
    WorkerFailureKind.INPUT_UNAVAILABLE: ErrorCode.ARTIFACT_PARSE_FAILED,
    WorkerFailureKind.INPUT_FORMAT_UNSUPPORTED: ErrorCode.ARTIFACT_PARSE_FAILED,
    WorkerFailureKind.INPUT_MALFORMED: ErrorCode.ARTIFACT_PARSE_FAILED,
    WorkerFailureKind.ROW_OR_OUTPUT_LIMIT: ErrorCode.QUERY_BUDGET_EXCEEDED,
    WorkerFailureKind.PROVIDER_UNAVAILABLE: ErrorCode.CAPABILITY_UNAVAILABLE,
    WorkerFailureKind.PROVIDER_INCOMPATIBLE: ErrorCode.ADAPTER_INCOMPATIBLE,
    WorkerFailureKind.WORKER_INTERNAL_ERROR: ErrorCode.INTERNAL_ERROR,
    WorkerFailureKind.OUTPUT_INVALID: ErrorCode.ARTIFACT_PARSE_FAILED,
}


class WorkerRequestEnvelope(ContractModel):
    transport: Literal["flameox.artifact-worker/v1"] = WORKER_TRANSPORT
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    operation: WorkerOperationId
    implementation: str = Field(min_length=1, max_length=200)
    payload: JsonValue


class TypedWorkerFailure(ContractModel):
    kind: WorkerFailureKind
    message: str = Field(min_length=1, max_length=500)


class WorkerSucceeded(ContractModel):
    transport: Literal["flameox.artifact-worker/v1"] = WORKER_TRANSPORT
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    operation: WorkerOperationId
    implementation: str = Field(min_length=1, max_length=200)
    kind: Literal["success"] = "success"
    payload: JsonValue


class WorkerFailed(ContractModel):
    transport: Literal["flameox.artifact-worker/v1"] = WORKER_TRANSPORT
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    operation: WorkerOperationId
    implementation: str = Field(min_length=1, max_length=200)
    kind: Literal["failure"] = "failure"
    failure: TypedWorkerFailure


type TypedWorkerResponse = Annotated[WorkerSucceeded | WorkerFailed, Field(discriminator="kind")]
TYPED_WORKER_RESPONSE: TypeAdapter[TypedWorkerResponse] = TypeAdapter(TypedWorkerResponse)


class WorkerOutputFile(ContractModel):
    role: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9._-]*$")
    relative_path: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    )
    media_type: str = Field(min_length=1, max_length=200)
    byte_length: int = Field(ge=0)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_is_safe_relative(self) -> WorkerOutputFile:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("worker output path must be a normalized relative path")
        return self


@dataclass(frozen=True, slots=True)
class WorkerDefinition[RequestT, ResponseT]:
    operation: WorkerOperationId
    module: str
    request: TypeAdapter[RequestT]
    response: TypeAdapter[ResponseT]
    name: str
    implementation: str
    timeout_seconds: float = 120


@dataclass(frozen=True, slots=True)
class WorkerContext:
    job_root: Path
    request_path: Path


@dataclass(frozen=True, slots=True)
class WorkerApplication[RequestT, ResponseT]:
    definition: WorkerDefinition[RequestT, ResponseT]
    handler: Callable[[RequestT, WorkerContext], ResponseT]
    invalid_failure: WorkerFailureKind
    invalid_message: str
    caught: tuple[type[BaseException], ...]


def worker_failure_error_code(kind: WorkerFailureKind) -> ErrorCode:
    return _FAILURE_ERROR_CODES[kind]


def run_typed_worker[RequestT, ResponseT](
    application: WorkerApplication[RequestT, ResponseT],
) -> int:
    """Execute one exact typed worker request with one shared transport contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        raw = arguments.request.read_bytes()
        if len(raw) > _MAX_REQUEST_BYTES:
            return 2
        envelope = WorkerRequestEnvelope.model_validate_json(raw)
    except (OSError, ValidationError, ValueError):
        return 2
    if envelope.operation is not application.definition.operation:
        return 2
    if envelope.implementation != application.definition.implementation:
        return 2
    try:
        request = application.definition.request.validate_python(envelope.payload)
        response = application.handler(
            request,
            WorkerContext(arguments.request.parent, arguments.request),
        )
        validated = application.definition.response.validate_python(response)
        response_payload = cast(
            JsonValue,
            application.definition.response.dump_python(validated, mode="json"),
        )
        outgoing: TypedWorkerResponse = WorkerSucceeded(
            request_id=envelope.request_id,
            operation=envelope.operation,
            implementation=envelope.implementation,
            payload=response_payload,
        )
        exit_code = 0
    except DomainError as exc:
        outgoing = WorkerFailed(
            request_id=envelope.request_id,
            operation=envelope.operation,
            implementation=envelope.implementation,
            failure=TypedWorkerFailure(
                kind=_failure_kind(exc.code),
                message=_bounded_message(exc.message),
            ),
        )
        exit_code = 1
    except Exception as exc:
        if not isinstance(exc, (ValidationError, *application.caught)):
            raise
        outgoing = WorkerFailed(
            request_id=envelope.request_id,
            operation=envelope.operation,
            implementation=envelope.implementation,
            failure=TypedWorkerFailure(
                kind=application.invalid_failure,
                message=_bounded_message(f"{application.invalid_message}: {type(exc).__name__}"),
            ),
        )
        exit_code = 1
    _write_typed_response(arguments.response, outgoing)
    return exit_code


def _failure_kind(code: ErrorCode) -> WorkerFailureKind:
    if code is ErrorCode.CAPABILITY_UNAVAILABLE:
        return WorkerFailureKind.PROVIDER_UNAVAILABLE
    if code in {
        ErrorCode.QUERY_BUDGET_EXCEEDED,
        ErrorCode.ARTIFACT_TOO_LARGE,
        ErrorCode.STORAGE_QUOTA_EXCEEDED,
    }:
        return WorkerFailureKind.ROW_OR_OUTPUT_LIMIT
    if code in {ErrorCode.INVALID_ARGUMENTS, ErrorCode.WORKSPACE_INVALID}:
        return WorkerFailureKind.INVALID_REQUEST
    if code is ErrorCode.ADAPTER_INCOMPATIBLE:
        return WorkerFailureKind.PROVIDER_INCOMPATIBLE
    if code is ErrorCode.INTERNAL_ERROR:
        return WorkerFailureKind.WORKER_INTERNAL_ERROR
    return WorkerFailureKind.INPUT_MALFORMED


def _bounded_message(value: str) -> str:
    return " ".join(value.split())[:500] or "Worker operation failed."


def _write_typed_response(path: Path, payload: TypedWorkerResponse) -> None:
    temporary = path.with_suffix(".tmp")
    encoded = TYPED_WORKER_RESPONSE.dump_json(payload)
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
