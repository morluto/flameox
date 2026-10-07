from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import anyio
import pytest

from flameox.canonical import canonical_bytes
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentCase,
    ExperimentDesign,
    PathSource,
    RequestLimits,
    RuntimeFailure,
    WorkloadBudget,
)
from tests.support.processes import process_is_alive, wait_for_pid_file


@pytest.mark.integration
def test_typed_capability_never_falls_back_to_generic_rows(tmp_path: Path) -> None:
    pyperf_artifact = tmp_path / "benchmark.json"
    samples_artifact = tmp_path / "benchmark.samples.json"
    pyperf_artifact.write_text("{}")
    samples_artifact.write_text("{}")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "benchmark.summary",
                [
                    PathSource(path=str(pyperf_artifact), format="pyperf"),
                    PathSource(path=str(samples_artifact), format="samples"),
                ],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "UNSUPPORTED_FORMAT"


@pytest.mark.integration
def test_analysis_rejects_source_cardinality_before_resolving_paths(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "memory.hotspots",
                [
                    PathSource(path=str(tmp_path / "missing-a.bin"), format="memray"),
                    PathSource(path=str(tmp_path / "missing-b.bin"), format="memray"),
                ],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "INVALID_INPUT"
    assert failure.value.details == {
        "capability_id": "memory.hotspots",
        "minimum_sources": 1,
        "maximum_sources": 1,
        "actual_sources": 2,
    }


@pytest.mark.integration
def test_capture_rejects_declared_provider_capability_mismatch_before_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).touch()",
                        ],
                        cwd=str(tmp_path),
                        provider_id="pyperf",
                    ),
                    "failures.summary",
                )
        finally:
            runtime.close()
        assert failure.value.code == "UNSUPPORTED_FORMAT"
        assert failure.value.details["output_formats"] == ["pyperf"]
        assert failure.value.details["compatible_capture_providers"] == [
            "observations",
            "pytest",
        ]

    anyio.run(exercise)
    assert not marker.exists()


@pytest.mark.integration
def test_capture_rejects_experiments_that_exceed_analysis_source_limit(tmp_path: Path) -> None:
    marker = tmp_path / "executed"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                        cwd=str(tmp_path),
                        provider_id="direct",
                    ),
                    "artifact.preview",
                    experiment=ExperimentDesign(
                        cases=[ExperimentCase(name="a"), ExperimentCase(name="b")],
                        blocks=9,
                        seed=1,
                        metric="wall_time_ns",
                        estimand="median_difference",
                        practical_threshold=0,
                    ),
                )
        finally:
            runtime.close()
        assert failure.value.code == "LIMIT_EXCEEDED"

    anyio.run(exercise)
    assert not marker.exists()


@pytest.mark.integration
def test_capture_rejects_experiment_unsupported_by_single_source_analysis(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                        cwd=str(tmp_path),
                        provider_id="coverage",
                    ),
                    "coverage.summary",
                    experiment=ExperimentDesign(
                        cases=[ExperimentCase(name="a"), ExperimentCase(name="b")],
                        blocks=1,
                        seed=1,
                        metric="wall_time_ns",
                        estimand="median_difference",
                        practical_threshold=0,
                    ),
                )
        finally:
            runtime.close()

        assert failure.value.code == "INVALID_INPUT"
        assert failure.value.details == {
            "capability_id": "coverage.summary",
            "minimum_sources": 1,
            "maximum_sources": 1,
            "actual_sources": 2,
        }

    anyio.run(exercise)
    assert not marker.exists()


def test_special_file_sources_are_rejected_before_decoding(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("artifact.preview", [PathSource(path=str(fifo))], {})
    finally:
        runtime.close()

    assert failure.value.code == "INVALID_INPUT"


@pytest.mark.process
def test_experiment_environment_limit_applies_after_overrides_are_merged(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "pass"],
                        cwd=str(tmp_path),
                        provider_id="direct",
                        environment={f"TARGET_{index}": "value" for index in range(32)},
                    ),
                    "artifact.preview",
                    experiment=ExperimentDesign(
                        cases=[
                            ExperimentCase(
                                name="baseline",
                                environment={f"CASE_{index}": "value" for index in range(32)},
                            ),
                            ExperimentCase(name="candidate"),
                        ],
                        blocks=1,
                        seed=1,
                        metric="wall_time_ns",
                        estimand="median_difference",
                        practical_threshold=0,
                    ),
                )
        finally:
            runtime.close()

        assert failure.value.code == "INVALID_INPUT"

    anyio.run(exercise)


@pytest.mark.process
def test_capture_rejects_unbounded_durable_provenance_before_execution(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            before = list(runtime.scratch.iterdir())
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "x" * 8_000],
                        cwd=str(tmp_path),
                        provider_id="direct",
                    ),
                    "artifact.preview",
                    limits=RequestLimits(max_provenance_bytes=4 * 1024),
                )
            after = list(runtime.scratch.iterdir())
        finally:
            runtime.close()

        assert failure.value.code == "LIMIT_EXCEEDED"
        assert before == after == []

    anyio.run(exercise)


@pytest.mark.integration
def test_analysis_is_bounded_deterministic_and_does_not_change_input(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(4)]))
    before = artifact.read_bytes()
    request = [PathSource(path=str(artifact))]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        limits = RequestLimits(max_rows=2)
        first = runtime.analyze("artifact.preview", request, {}, limits=limits)
        second = runtime.analyze("artifact.preview", request, {}, limits=limits)
    finally:
        runtime.close()

    assert first == second
    assert first["analysis_id"] == second["analysis_id"]
    assert first["inputs"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert first["coverage"] == {"rows_returned": 2, "rows_observed": 3, "complete": False}
    assert first["continuation"]
    assert first["truncation"] == {"reason": "row_limit", "next_offset": 2}
    assert artifact.read_bytes() == before


@pytest.mark.integration
@pytest.mark.parametrize("offset", [-1, 100])
def test_analysis_rejects_a_tampered_continuation_offset(tmp_path: Path, offset: int) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(4)]))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        limits = RequestLimits(max_rows=2)
        first = runtime.analyze(
            "artifact.preview", [PathSource(path=str(artifact))], {}, limits=limits
        )
        decoded = json.loads(
            base64.urlsafe_b64decode(
                first["continuation"] + "=" * (-len(first["continuation"]) % 4)
            )
        )
        decoded["payload"]["offset"] = offset
        decoded["checksum"] = hashlib.sha256(canonical_bytes(decoded["payload"])).hexdigest()
        continuation = base64.urlsafe_b64encode(canonical_bytes(decoded)).decode().rstrip("=")

        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(artifact))],
                {},
                limits=limits,
                continuation=continuation,
            )
    finally:
        runtime.close()

    assert failure.value.code == "INVALID_INPUT"


@pytest.mark.integration
def test_continuation_pages_have_distinct_preservable_analysis_ids(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(4)]))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        first = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
        )
        second = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
            continuation=first["continuation"],
        )

        assert first["analysis_id"] != second["analysis_id"]
        assert [row["value"] for row in first["blocks"][1]["rows"]] == [0, 1]
        assert [row["value"] for row in second["blocks"][1]["rows"]] == [2, 3]
        assert runtime.preserve_evidence(first["analysis_id"])["evidence_id"]
        assert runtime.preserve_evidence(second["analysis_id"])["evidence_id"]
    finally:
        runtime.close()


@pytest.mark.integration
def test_session_analysis_cache_expires_least_recently_used_handles(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(65)]))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        first = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=1),
        )
        latest = first
        for _ in range(64):
            latest = runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(artifact))],
                {},
                limits=RequestLimits(max_rows=1),
                continuation=latest["continuation"],
            )

        assert runtime.preserve_evidence(latest["analysis_id"])["evidence_id"]
        with pytest.raises(RuntimeFailure, match="expired"):
            runtime.preserve_evidence(first["analysis_id"])
    finally:
        runtime.close()


@pytest.mark.integration
def test_explicit_inputs_fail_at_byte_and_file_bounds(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * 2048)
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "one.txt").write_text("one")
    (directory / "two.txt").write_text("two")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as byte_failure:
            runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(oversized))],
                {},
                limits=RequestLimits(max_input_bytes=1024),
            )
        assert byte_failure.value.code == "LIMIT_EXCEEDED"

        with pytest.raises(RuntimeFailure) as file_failure:
            runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(directory))],
                {},
                limits=RequestLimits(max_input_files=1),
            )
        assert file_failure.value.code == "LIMIT_EXCEEDED"
    finally:
        runtime.close()


@pytest.mark.integration
def test_input_limits_apply_across_all_explicit_sources(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"a" * 600)
    second.write_bytes(b"b" * 600)
    sources = [PathSource(path=str(first)), PathSource(path=str(second))]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as byte_failure:
            runtime.analyze(
                "artifact.preview",
                sources,
                {},
                limits=RequestLimits(max_input_bytes=1024),
            )
        assert byte_failure.value.code == "LIMIT_EXCEEDED"

        with pytest.raises(RuntimeFailure) as file_failure:
            runtime.analyze(
                "artifact.preview",
                sources,
                {},
                limits=RequestLimits(max_input_files=1),
            )
        assert file_failure.value.code == "LIMIT_EXCEEDED"
    finally:
        runtime.close()


@pytest.mark.integration
def test_digest_mismatch_fails_before_decoding(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid.json"
    artifact.write_text("not json")

    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    with pytest.raises(RuntimeFailure, match="SHA-256 mismatch") as failure:
        runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact), expected_sha256="0" * 64)],
            {},
        )
    assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
    runtime.close()


@pytest.mark.integration
def test_json_object_sequence_is_streamed_with_bounded_continuation(tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps({"metadata": {"ignored": True}, "results": list(range(20))}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        first = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=3),
        )
        second = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=3),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    first_rows = first["blocks"][1]["rows"]
    assert first_rows[0]["key"] == "metadata"
    assert first_rows[0]["value_type"] == "object"
    assert [row["value"] for row in first_rows[1:]] == [0, 1]
    assert [row["value"] for row in second["blocks"][1]["rows"]] == [2, 3, 4]


@pytest.mark.integration
def test_json_object_preview_includes_all_arrays_and_root_scalars(tmp_path: Path) -> None:
    artifact = tmp_path / "sections.json"
    artifact.write_text(json.dumps({"first": [1, 2], "label": "kept", "second": [{"value": 3}]}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=10),
        )
    finally:
        runtime.close()

    rows = result["blocks"][1]["rows"]
    assert rows == [
        {"section": "first", "value": 1, "input_sha256": result["inputs"][0]["sha256"]},
        {"section": "first", "value": 2, "input_sha256": result["inputs"][0]["sha256"]},
        {"key": "label", "value": "kept", "input_sha256": result["inputs"][0]["sha256"]},
        {
            "section": "second",
            "value": 3,
            "input_sha256": result["inputs"][0]["sha256"],
        },
    ]


@pytest.mark.integration
def test_preview_source_digest_wins_over_user_row_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "rows.json"
    artifact.write_text(json.dumps([{"input_sha256": "f" * 64, "value": 1}]))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"][0]["input_sha256"] == result["inputs"][0]["sha256"]


@pytest.mark.process
def test_direct_capture_reports_progress_and_preserves_native_output(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=10)
        )
        updates: list[tuple[int, int]] = []

        async def progress(current: int, total: int, _message: str) -> None:
            updates.append((current, total))

        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "print('captured')"],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                progress=progress,
            )
            assert result["blocks"][1]["rows"][0]["text"] == "captured"
            assert updates == [(0, 1), (1, 1)]
            provider_id = result["provider"]["id"]
            result["provider"]["id"] = "caller-mutated"
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert {item["role"] for item in manifest["body"]["artifacts"]} == {
                "capture-0001/stdout",
                "capture-0001/stderr",
            }
            layout = manifest["body"]["source_layout"]
            analyzed_index = layout["analysis_sources"][0]
            assert layout["sources"][analyzed_index]["role"] == ("capture-0001/stdout")
            bundle = (
                tmp_path
                / ".flameox"
                / "evidence"
                / "sha256"
                / preserved["evidence_id"][:2]
                / preserved["evidence_id"]
            )
            durable = json.loads((bundle / "data" / "analysis.json").read_text())
            assert durable["provider"]["id"] == provider_id
            execution = manifest["body"]["capture_request"]["executions"][0]
            assert execution["collector_executable_sha256"] == execution["executable_sha256"]
            assert execution["workload_executable_sha256"] == execution["executable_sha256"]
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_direct_capture_executes_and_preserves_empty_nonprogram_argument(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "import sys; print(repr(sys.argv[1]))", ""],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
            )
            assert result["blocks"][1]["rows"][0]["text"] == "''"
            ref = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(ref["evidence_id"])
            assert manifest["body"]["capture_request"]["executions"][0]["argv"][-1] == ""
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_timed_out_capture_returns_preservable_partial_evidence(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import time; print('before-timeout', flush=True); time.sleep(5)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                    budget=WorkloadBudget(timeout_seconds=0.2),
                ),
                "artifact.preview",
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "failed"
            assert execution["failure_code"] == "EXECUTION_TIMEOUT"
            assert execution["limit"]["kind"] == "timeout"
            assert execution["limit"]["configured"] == 0.2
            assert execution["limit"]["unit"] == "seconds"
            assert result["blocks"][1]["rows"][0]["text"] == "before-timeout"
            assert runtime.preserve_evidence(result["analysis_id"])["evidence_id"]
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_cancelled_capture_cleans_up_descendants(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        task = asyncio.create_task(
            runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        (
                            "import pathlib, subprocess, sys, time; "
                            "child=subprocess.Popen([sys.executable, '-c', 'import time; "
                            "time.sleep(30)']); "
                            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
                            "time.sleep(30)"
                        ),
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
            )
        )
        try:
            child_pid = await wait_for_pid_file(child_pid_file)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not process_is_alive(child_pid)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_experiment_runs_bounded_cases_and_semantic_oracle(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "print('base')"],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                experiment=ExperimentDesign(
                    cases=[
                        ExperimentCase(
                            name="baseline",
                            argv=[sys.executable, "-c", "print('baseline')"],
                        ),
                        ExperimentCase(
                            name="candidate",
                            argv=[sys.executable, "-c", "print('candidate')"],
                        ),
                    ],
                    blocks=1,
                    seed=7,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                    semantic_oracle=[
                        sys.executable,
                        "-c",
                        (
                            "import os, pathlib; "
                            "raise SystemExit(not pathlib.Path("
                            "os.environ['FLAMEOX_CAPTURE_STDOUT']).read_text().strip())"
                        ),
                    ],
                ),
            )
            executions = result["capture"]["executions"]
            assert {item["case"] for item in executions} == {"baseline", "candidate"}
            assert all(item["semantic_oracle"]["status"] == "passed" for item in executions)
            assert all(item["status"] == "succeeded" for item in executions)
            comparison = result["blocks"][-1]["rows"][0]
            assert comparison["baseline_case"] == "baseline"
            assert comparison["candidate_case"] == "candidate"
            assert comparison["metric"] == "wall_time_ns"
            assert comparison["estimand"] == "median_difference"
            assert comparison["paired_blocks"] == 1
            estimate = comparison["estimate"]
            assert result["blocks"][-2]["values"]["decision_basis"] == (
                "descriptive_point_estimate"
            )
            if estimate is None:
                expected = "inconclusive"
            elif abs(estimate) <= 0:
                expected = "within_threshold"
            elif estimate < 0:
                expected = "practically_improved"
            else:
                expected = "practically_regressed"
            assert comparison["point_estimate_classification"] == expected
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert manifest["body"]["limitations"] == result["limitations"]
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_experiment_skips_semantic_oracle_after_failed_capture(tmp_path: Path) -> None:
    marker = tmp_path / "oracle-ran"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "raise SystemExit(7)"],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
                    blocks=1,
                    seed=7,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                    semantic_oracle=[
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()",
                    ],
                ),
            )
            assert all(item["status"] == "failed" for item in result["capture"]["executions"])
            assert all(item["semantic_oracle"] is None for item in result["capture"]["executions"])
            assert not marker.exists()
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_capture_returns_all_execution_provenance(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass", "x" * 12_000],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name=f"case-{index}") for index in range(16)],
                    blocks=1,
                    seed=7,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                ),
            )
        finally:
            runtime.close()

        executions = result["capture"]["executions"]
        assert len(executions) == 16
        assert all(item["argv"][-1] == "x" * 12_000 for item in executions)
        assert result["capture"]["outcome"]["execution_count"] == 16

    anyio.run(exercise)
