from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

BoundedFilter = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, pattern=r"^[^\x00\r\n]+$"),
]
