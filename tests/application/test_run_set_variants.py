from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.application import (
    ExcludedFreezeRunSetMember,
    FreezeRunIdsRequest,
    FreezeRunMembersRequest,
    FreezeRunSetRequest,
    IncludedFreezeRunSetMember,
    RunSetService,
)
from flameox.domain import digest_model
from flameox.domain.models import ExcludedRunSetMember, IncludedRunSetMember
from flameox.storage import Workspace
from tests.support.comparisons import imported_benchmark


def test_freeze_request_parser_returns_only_legal_membership_variants() -> None:
    adapter: TypeAdapter[FreezeRunSetRequest] = TypeAdapter(FreezeRunSetRequest)

    by_ids = adapter.validate_python({"run_ids": ["run-a"]})
    by_members = adapter.validate_python(
        {
            "members": [
                {"run_id": "run-a"},
                {"run_id": "run-b", "included": False, "reason": "timed out"},
            ]
        }
    )

    assert isinstance(by_ids, FreezeRunIdsRequest)
    assert isinstance(by_members, FreezeRunMembersRequest)
    assert isinstance(by_members.members[0], IncludedFreezeRunSetMember)
    assert isinstance(by_members.members[1], ExcludedFreezeRunSetMember)
    assert by_ids.model_dump(mode="json") == {
        "selection": {},
        "run_ids": ["run-a"],
        "members": [],
    }
    assert by_members.members[0].model_dump(mode="json") == {
        "run_id": "run-a",
        "trial_id": None,
        "included": True,
        "reason": None,
    }

    invalid: tuple[dict[str, object], ...] = (
        {},
        {"run_ids": ["run-a"], "members": [{"run_id": "run-b"}]},
        {"members": [{"run_id": "run-a", "included": False}]},
        {"members": [{"run_id": "run-a", "reason": "not excluded"}]},
    )
    for payload in invalid:
        with pytest.raises(ValidationError):
            adapter.validate_python(payload)


def test_frozen_members_round_trip_as_durable_variants_without_changing_digest(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    included_id = imported_benchmark(
        workspace,
        tmp_path / "included.json",
        (0.010, 0.011, 0.012),
    )
    excluded_id = imported_benchmark(
        workspace,
        tmp_path / "excluded.json",
        (0.020, 0.021, 0.022),
    )
    service = RunSetService(workspace)

    frozen = service.freeze(
        FreezeRunMembersRequest(
            members=(
                IncludedFreezeRunSetMember(run_id=included_id),
                ExcludedFreezeRunSetMember(run_id=excluded_id, reason="warmup drift"),
            )
        )
    )
    stored = service.store.read(frozen.run_set_id)

    assert isinstance(stored.members[0], IncludedRunSetMember)
    assert isinstance(stored.members[1], ExcludedRunSetMember)
    assert stored.members == frozen.members
    assert stored.membership_digest == digest_model(
        [member.model_dump(mode="json") for member in stored.members]
    )
