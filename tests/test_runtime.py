from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentCase,
    ExperimentDesign,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


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
