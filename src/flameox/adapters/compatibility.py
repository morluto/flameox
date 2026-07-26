from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

from flameox.domain import ArtifactRegistration, DomainError, ErrorCode


def require_supported_producer_major(
    registration: ArtifactRegistration,
    *,
    package: str,
    producer_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    """Reject identified producer majors newer than the installed public reader."""

    producer = (registration.producer or "").lower()
    producer_version = registration.producer_version
    if not any(token in producer for token in producer_tokens):
        return (
            f"The {package} artifact producer is unidentified; "
            "format compatibility could not be verified.",
        )
    if producer_version is None:
        return (
            f"The {package} artifact producer version is unavailable; "
            "format compatibility could not be verified.",
        )
    try:
        reader_version = Version(version(package))
        artifact_version = Version(producer_version)
    except (InvalidVersion, PackageNotFoundError) as exc:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"The {package} artifact producer version is not identifiable.",
            run_id=registration.run_id,
        ) from exc
    if artifact_version.major > reader_version.major:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"The {package} artifact was produced by an unsupported newer major version.",
            run_id=registration.run_id,
            details={
                "producer_version": str(artifact_version),
                "reader_version": str(reader_version),
            },
        )
    return ()
