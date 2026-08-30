from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from flameox.models import ContractModel


class PredicateClassification(StrEnum):
    INTERESTING = "interesting"
    NOT_INTERESTING = "not_interesting"
    UNRESOLVED = "unresolved"


class ReductionDisposition(StrEnum):
    SUCCEEDED = "succeeded"
    UNCHANGED = "unchanged"
    INCONCLUSIVE = "inconclusive"
    ORIGINAL_NOT_INTERESTING = "original_not_interesting"


class ReductionMinimality(StrEnum):
    NOT_CLAIMED = "not_claimed"


class PredicateObservation(ContractModel):
    repetition: int = Field(ge=0, le=19)
    classification: PredicateClassification
    exit_code: int | None = None
    failure_category: str | None = Field(default=None, max_length=100)
    duration_ms: float = Field(ge=0)


class ReductionAttemptReceipt(ContractModel):
    attempt_id: str = Field(pattern=r"^attempt-[0-9]{8}$")
    candidate_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    candidate_size_bytes: int = Field(ge=0)
    observations: tuple[PredicateObservation, ...]
    classification: PredicateClassification
    recorded_at: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def classification_matches_observations(self) -> ReductionAttemptReceipt:
        observed = {item.classification for item in self.observations}
        expected = (
            next(iter(observed))
            if len(observed) == 1 and PredicateClassification.UNRESOLVED not in observed
            else PredicateClassification.UNRESOLVED
        )
        if self.classification is not expected:
            raise ValueError("attempt classification does not match its observations")
        return self


def collapse_predicate_observations(
    values: tuple[PredicateClassification, ...],
) -> PredicateClassification:
    unique = set(values)
    if len(unique) != 1 or PredicateClassification.UNRESOLVED in unique:
        return PredicateClassification.UNRESOLVED
    return next(iter(unique))
