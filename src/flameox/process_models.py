from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, computed_field, model_validator

from flameox.models import ContractModel


class ProcessCancellationCause(StrEnum):
    TIMEOUT = "timeout"
    CALLER_CANCELLED = "caller_cancelled"
    OUTPUT_LIMIT = "output_limit"
    IO_FAILURE = "io_failure"
    STORAGE_RESERVE_EXCEEDED = "storage_reserve_exceeded"
    WRITABLE_LIMIT_EXCEEDED = "writable_limit_exceeded"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    PROCESS_ERROR = "process_error"


class ProcessTerminationKind(StrEnum):
    UNREPORTED = "unreported"
    EXITED = "exited"
    SIGNALLED = "signalled"


class UnreportedProcessTermination(ContractModel):
    kind: Literal[ProcessTerminationKind.UNREPORTED] = ProcessTerminationKind.UNREPORTED


class ExitedProcessTermination(ContractModel):
    kind: Literal[ProcessTerminationKind.EXITED] = ProcessTerminationKind.EXITED
    exit_code: Annotated[int, Field(ge=0)]


class SignalledProcessTermination(ContractModel):
    kind: Literal[ProcessTerminationKind.SIGNALLED] = ProcessTerminationKind.SIGNALLED
    signal: Annotated[int, Field(gt=0)]


type ProcessTermination = Annotated[
    UnreportedProcessTermination | ExitedProcessTermination | SignalledProcessTermination,
    Field(discriminator="kind"),
]


def process_termination_from_returncode(returncode: int | None) -> ProcessTermination:
    if returncode is None:
        return UnreportedProcessTermination()
    if returncode >= 0:
        return ExitedProcessTermination(exit_code=returncode)
    return SignalledProcessTermination(signal=-returncode)


def process_exit_code(termination: ProcessTermination) -> int | None:
    return termination.exit_code if isinstance(termination, ExitedProcessTermination) else None


type ResourcePolicyCancellationCause = Literal[
    ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED,
    ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
    ProcessCancellationCause.MEMORY_LIMIT_EXCEEDED,
]


class RuntimeResourceSummary(ContractModel):
    sampling_interval_ms: Annotated[int, Field(gt=0)]
    minimum_free_bytes: Annotated[int, Field(ge=0)] | None = None
    staging_growth_bytes: Annotated[int, Field(ge=0)] | None = None
    writable_root_growth_bytes: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    peak_rss_bytes: Annotated[int, Field(ge=0)] | None = None
    peak_rss_backend: str | None = None
    unavailable_metrics: tuple[str, ...] = ()
    policy_termination: ResourcePolicyCancellationCause | None = None


class ProcessResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    termination: ProcessTermination = Field(default_factory=UnreportedProcessTermination)
    wall_time_ns: Annotated[int, Field(ge=0)] | None = None
    peak_rss_bytes: Annotated[int, Field(ge=0)] | None = None
    cancellation_cause: ProcessCancellationCause | None = None
    cleanup_complete: bool | None = None
    resources: RuntimeResourceSummary | None = None
    stdout: str | None = None
    stderr: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timed_out(self) -> bool:
        return self.cancellation_cause is ProcessCancellationCause.TIMEOUT

    @model_validator(mode="after")
    def resource_summary_is_coherent(self) -> ProcessResult:
        if self.resources is not None and self.peak_rss_bytes != self.resources.peak_rss_bytes:
            raise ValueError("process and resource-summary peak RSS must agree")
        return self
