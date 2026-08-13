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
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import (
    MAX_RUN_SET_MEMBERS,
    ExcludedRunSetMember,
    IncludedRunSetMember,
)
from flameox.storage import RetentionIntentStore, Workspace
from tests.support.comparisons import imported_benchmark

pytestmark = [pytest.mark.integration, pytest.mark.serial]


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
        "corpus_commit_id": None,
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


def test_freeze_request_rejects_duplicate_unbounded_or_oversized_membership() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FreezeRunIdsRequest(run_ids=("run-a", "run-a"))
    with pytest.raises(ValidationError, match="only once"):
        FreezeRunMembersRequest(
            members=(
                IncludedFreezeRunSetMember(run_id="run-a", trial_id="trial-a"),
                IncludedFreezeRunSetMember(run_id="run-a", trial_id="trial-b"),
            )
        )
    with pytest.raises(ValidationError, match="at most"):
        FreezeRunIdsRequest(
            run_ids=tuple(f"run-{index}" for index in range(MAX_RUN_SET_MEMBERS + 1))
        )
    with pytest.raises(ValidationError, match="500 characters"):
        FreezeRunIdsRequest(
            run_ids=("run-a",),
            selection={"label": "x" * 501},
        )
    with pytest.raises(ValidationError, match="4 nested levels"):
        FreezeRunIdsRequest(
            run_ids=("run-a",),
            selection={"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
        )


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

    request = FreezeRunMembersRequest(
        members=(
            IncludedFreezeRunSetMember(run_id=included_id),
            ExcludedFreezeRunSetMember(run_id=excluded_id, reason="warmup drift"),
        )
    )
    frozen = service.freeze(request)
    repeated = service.freeze(request)
    stored = service.store.read(frozen.run_set_id)

    assert isinstance(stored.members[0], IncludedRunSetMember)
    assert isinstance(stored.members[1], ExcludedRunSetMember)
    assert stored.members == frozen.members
    assert repeated == frozen
    assert stored.membership_digest == digest_model(
        [member.model_dump(mode="json") for member in stored.members]
    )
    assert RetentionIntentStore(workspace).pending() == ()
    with Catalog(workspace).open_snapshot() as snapshot:
        count = snapshot.execute(
            "SELECT count(*) FROM run_sets WHERE run_set_id = ?",
            (frozen.run_set_id,),
        ).fetchone()
    assert count == (1,)


def test_freeze_quota_failure_leaves_no_large_control_record(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id = imported_benchmark(
        workspace,
        tmp_path / "run.json",
        (0.010, 0.011, 0.012),
    )
    workspace.paths.config.write_text(
        workspace.config.validated_copy(
            update={
                "storage": workspace.config.storage.validated_copy(
                    update={"max_workspace_bytes": 1}
                )
            }
        ).to_toml()
    )
    service = RunSetService(workspace)

    with pytest.raises(DomainError) as failure:
        service.freeze(FreezeRunIdsRequest(run_ids=(run_id,)))

    assert failure.value.code is ErrorCode.STORAGE_QUOTA_EXCEEDED
    assert service.store.control_plane.list_records(kind="run_sets") == ()
    assert len(RetentionIntentStore(workspace).pending()) == 1
