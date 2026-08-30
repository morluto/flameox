from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from flameox.models import ContractModel

_BoundedFilter = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, pattern=r"^[^\x00\r\n]+$"),
]


class NsightComputeOptions(ContractModel):
    """Bounded selections accepted by the managed Nsight Compute capture."""

    set: (
        Annotated[
            str,
            StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$"),
        ]
        | None
    ) = "basic"
    sections: (
        Annotated[
            tuple[
                Annotated[
                    str,
                    StringConstraints(
                        min_length=1,
                        max_length=100,
                        pattern=r"^[A-Za-z0-9_.-]+$",
                    ),
                ],
                ...,
            ],
            Field(min_length=1, max_length=32),
        ]
        | None
    ) = None
    kernel_name: _BoundedFilter | None = None
    launch_skip: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    launch_count: Annotated[int, Field(ge=1, le=1_000_000)] = 1
    replay_mode: Literal["kernel", "application", "range", "app-range"] = "kernel"

    @model_validator(mode="before")
    @classmethod
    def explicit_sections_replace_default_set(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("sections") is not None and "set" not in value:
            return {**value, "set": None}
        return value

    @model_validator(mode="after")
    def one_profile_selection(self) -> Self:
        if (self.set is None) == (self.sections is None):
            raise ValueError("exactly one of set or sections must be selected")
        if self.sections is not None and len(set(self.sections)) != len(self.sections):
            raise ValueError("sections must not contain duplicates")
        return self
