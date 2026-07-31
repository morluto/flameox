from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from flameox.application import (
    CompareRunSetsRequest,
    ComparisonService,
    FreezeRunSetMember,
    FreezeRunSetRequest,
    RunSetService,
)
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    RunSet,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.comparisons import (
    imported_benchmark,
)


@pytest.mark.anyio
async def test_async_comparison_cancellation_interrupts_duckdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate.json",
        (0.005, 0.0055, 0.006),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))
    request = CompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity="lower_is_better",
        practical_threshold=0.05,
    )
    query_started = threading.Event()
    original = ComparisonService._compatibility_mismatches

    def slow_compatibility(
        service: ComparisonService,
        snapshot: Snapshot,
        baseline_set: RunSet,
        candidate_set: RunSet,
    ) -> list[str]:
        query_started.set()
        snapshot.execute("SELECT sum(sin(i)) FROM range(100000000000) values(i)").fetchall()
        return original(service, snapshot, baseline_set, candidate_set)

    monkeypatch.setattr(ComparisonService, "_compatibility_mismatches", slow_compatibility)
    task = asyncio.create_task(ComparisonService(workspace).compare_async(request))
    assert await asyncio.to_thread(query_started.wait, 5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_trial_block_identity_makes_pairing_independent_of_member_order(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    runs: dict[str, str] = {}
    for name, values in {
        "baseline-1": (0.010, 0.010, 0.010),
        "baseline-2": (0.020, 0.020, 0.020),
        "candidate-1": (0.005, 0.005, 0.005),
        "candidate-2": (0.015, 0.015, 0.015),
    }.items():
        runs[name] = imported_benchmark(workspace, tmp_path / f"{name}.json", values)
    trials = [
        ("baseline-trial-1", runs["baseline-1"], "baseline", "block-1"),
        ("baseline-trial-2", runs["baseline-2"], "baseline", "block-2"),
        ("candidate-trial-1", runs["candidate-1"], "candidate", "block-1"),
        ("candidate-trial-2", runs["candidate-2"], "candidate", "block-2"),
    ]
    GenerationPublisher(workspace).publish_rows(
        {
            "trials": [
                {
                    "trial_id": trial_id,
                    "experiment_id": "experiment",
                    "variant_id": variant,
                    "run_id": run_id,
                    "combination_id": digest_model({"variant": variant, "block": block_id}),
                    "factors_json": json.dumps({"variant": variant}, sort_keys=True),
                    "block_id": block_id,
                    "order_in_block": 0,
                    "parameter_name": None,
                    "parameter_value_int": None,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "not_requested",
                    "failure_class": "none",
                }
                for trial_id, run_id, variant, block_id in trials
            ]
        },
        publisher="trial-fixture",
        publisher_version="1",
        input_run_ids=tuple(run_id for _, run_id, _, _ in trials),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(
        FreezeRunSetRequest(
            members=tuple(
                FreezeRunSetMember(run_id=run_id, trial_id=trial_id)
                for trial_id, run_id, variant, _ in trials
                if variant == "baseline"
            )
        )
    )
    candidates = [
        (trial_id, run_id) for trial_id, run_id, variant, _ in trials if variant == "candidate"
    ]
    candidate_forward = run_sets.freeze(
        FreezeRunSetRequest(
            members=tuple(
                FreezeRunSetMember(run_id=run_id, trial_id=trial_id)
                for trial_id, run_id in candidates
            )
        )
    )
    candidate_reversed = run_sets.freeze(
        FreezeRunSetRequest(
            members=tuple(
                FreezeRunSetMember(run_id=run_id, trial_id=trial_id)
                for trial_id, run_id in reversed(candidates)
            )
        )
    )
    service = ComparisonService(workspace)

    forward = service.compare(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate_forward.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.01,
        )
    )
    reversed_result = service.compare(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate_reversed.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.01,
        )
    )

    assert forward.comparison.complete_pair_n == 2
    assert forward.comparison.relative_change == reversed_result.comparison.relative_change
