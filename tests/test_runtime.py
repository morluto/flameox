from __future__ import annotations

import json
import os
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
            source = PathSource(
                path=result["inputs"][0]["path"], format=result["inputs"][0]["format"]
            )
            sibling = runtime.analyze(
                "artifact.preview", [source], {}, limits=RequestLimits(max_rows=1)
            )
            assert sibling["analysis_id"] != result["analysis_id"]
            preserved = runtime.preserve_evidence(result["analysis_id"])
            assert Path(source.path).is_file()
            runtime.preserve_evidence(sibling["analysis_id"])
            assert not Path(source.path).exists()
            assert not list(runtime.scratch.glob("capture-*"))
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
@pytest.mark.skipif(os.name == "nt", reason="Executable script fixture requires POSIX")
@pytest.mark.parametrize("mutation", ["oracle", "output"])
def test_experiment_rejects_changed_oracle_or_capture_bytes(tmp_path: Path, mutation: str) -> None:
    oracle = tmp_path / "oracle"
    oracle.write_text(f"#!{sys.executable}\nraise SystemExit(1)\n")
    oracle.chmod(0o755)
    argv = [sys.executable, "-c", "print('real')"]
    oracle_argv = [str(oracle)]
    if mutation == "oracle":
        argv = [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text(sys.argv[2]); print('real')",
            str(oracle),
            f"#!{sys.executable}\nraise SystemExit(0)\n",
        ]
    else:
        oracle_argv = [
            sys.executable,
            "-c",
            "import os; from pathlib import Path; "
            "Path(os.environ['FLAMEOX_CAPTURE_STDOUT']).write_text('forged\\n')",
        ]

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(argv=argv, cwd=str(tmp_path), provider_id="direct"),
                "artifact.preview",
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name="base"), ExperimentCase(name="candidate")],
                    blocks=1,
                    seed=1,
                    metric="wall_time_ns",
                    estimand="mean_difference",
                    practical_threshold=0,
                    semantic_oracle=oracle_argv,
                ),
                preserve=mutation == "oracle",
            )
            if mutation == "oracle":
                assert result["capture"]["outcome"]["failed_count"] == 2
                for execution in result["capture"]["executions"]:
                    assert (
                        execution["semantic_oracle"]["failure_code"] == "MISSING_OR_CHANGED_INPUT"
                    )
                    assert execution["semantic_oracle"]["returncode"] is None
                assert (
                    runtime.read_evidence(result["preserved"]["evidence_id"])["body"][
                        "evidence_kind"
                    ]
                    == "capture"
                )
            else:
                assert result["analysis_failure"]["code"] == "MISSING_OR_CHANGED_INPUT"
                assert result["blocks"][1]["rows"] == []
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
            assert result["blocks"][-2]["values"]["experiment_timing_scope"] == (
                "capture_process_wall_time"
            )
            assert result["blocks"][-2]["values"]["semantic_oracle_input_scope"] == (
                "capture_process_console"
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
            assert manifest["body"]["evidence_kind"] == "capture"
            assert (
                runtime.query_evidence(evidence_kind="capture")["evidence"][0]["evidence_id"]
                == preserved["evidence_id"]
            )
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.parametrize(
    "document", ["a,a\n1,2\n", "a,b\n1\n", "a,b\n1,2,3\n", "a\n" + "x" * 200_000]
)
def test_preview_rejects_lossy_or_oversized_csv(tmp_path: Path, document: str) -> None:
    artifact = tmp_path / "invalid.csv"
    artifact.write_text(document)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.parametrize("format_name", ["json", "jsonl", "csv", "parquet"])
def test_preview_preserves_native_fields_that_collide_with_provenance(
    tmp_path: Path, format_name: str
) -> None:
    import hashlib

    row = {"input_sha256": "native field", "value": "native value"}
    artifact = tmp_path / f"native.{format_name}"
    if format_name == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist([row]), artifact)
    elif format_name == "csv":
        artifact.write_text("input_sha256,value\nnative field,native value\n")
    else:
        artifact.write_text(json.dumps([row] if format_name == "json" else row))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        observed = result["blocks"][1]["rows"][0]
        assert observed["input_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert observed["value"] == row
    finally:
        runtime.close()


def test_preview_preserves_native_section_field(tmp_path: Path) -> None:
    artifact = tmp_path / "native.json"
    artifact.write_text(json.dumps({"actual": [{"section": "native", "answer": 42}]}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        row = result["blocks"][1]["rows"][0]
        assert row["section"] == "actual"
        assert row["value"] == {"section": "native", "answer": 42}
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "format_name,native,expected",
    [
        (format_name, native, expected)
        for format_name in ("json", "jsonl")
        for native, expected in (
            (str(2**53 + 42), str(2**53 + 42)),
            ("NaN", "NaN"),
            ("1e400", "Infinity"),
        )
    ]
    + [("jsonl", '"\\ud800"', None)],
)
def test_preview_native_values_stay_preservable_or_fail_with_typed_error(
    tmp_path: Path, format_name: str, native: str, expected: str | None
) -> None:
    artifact = tmp_path / f"native.{format_name}"
    document = '{"value":' + native + "}"
    artifact.write_text("[" + document + "]" if format_name == "json" else document + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        if expected is None or (format_name == "json" and native in {"NaN", "1e400"}):
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            assert failure.value.code == "DECODE_FAILURE"
        else:
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            assert result["blocks"][1]["rows"][0]["value"] == expected
            reference = runtime.preserve_evidence(result["analysis_id"])
            assert runtime.read_evidence(reference["evidence_id"])
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture")
def test_experiment_retains_completed_capture_when_later_executable_changes(tmp_path: Path) -> None:
    workload = tmp_path / "workload"
    workload.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        "print('observed')\nwith Path(__file__).open('a') as stream:\n"
        "    stream.write('# changed\\n')\n"
    )
    workload.chmod(0o755)

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(argv=[str(workload)], cwd=str(tmp_path), provider_id="direct"),
                "artifact.preview",
                preserve=True,
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
                    blocks=1,
                    seed=0,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                ),
            )
            executions = result["capture"]["executions"]
            assert executions[0]["status"] == "succeeded"
            assert executions[1]["status"] == "failed"
            assert executions[1]["failure_code"] == "MISSING_OR_CHANGED_INPUT"
            assert executions[1]["returncode"] is None
            assert result["blocks"][1]["rows"][0]["text"] == "observed"
            manifest = runtime.read_evidence(result["preserved"]["evidence_id"])
            assert manifest["body"]["capture_request"]["executions"] == executions
            assert len(manifest["body"]["artifacts"]) == 2
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_experiment_retains_capture_when_oracle_cannot_admit_more_files(tmp_path: Path) -> None:
    payload = {
        "schema_version": "flameox.benchmark-samples.v1",
        "producer": "bounded-workload",
        "benchmarks": [
            {
                "name": "work",
                "unit": "ns",
                "measurement_clock": "host_monotonic",
                "synchronization": "not_required",
                "samples": [1],
            }
        ],
    }
    code = (
        "import os,pathlib; p=pathlib.Path(os.environ['FLAMEOX_BENCHMARK_OUTPUT']); "
        f"p.write_text({json.dumps(payload)!r}); "
        "[(p.parent/str(i)).touch() for i in range(8187)]"
    )

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", code],
                    cwd=str(tmp_path),
                    provider_id="benchmark-samples",
                ),
                "benchmark.summary",
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name="a"), ExperimentCase(name="b")],
                    blocks=1,
                    seed=0,
                    metric="wall_time_ns",
                    estimand="mean_difference",
                    practical_threshold=0,
                    semantic_oracle=[sys.executable, "-c", "pass"],
                ),
                preserve=True,
            )
            first, second = result["capture"]["executions"]
            assert first["returncode"] == 0
            assert first["failure_code"] == "SEMANTIC_ORACLE_FAILED"
            assert first["semantic_oracle"]["failure_code"] == "LIMIT_EXCEEDED"
            assert first["semantic_oracle"]["returncode"] is None
            assert second["returncode"] is None
            assert second["failure_code"] == "LIMIT_EXCEEDED"
            assert result["capture"]["outcome"]["failed_count"] == 2
            manifest = runtime.read_evidence(result["preserved"]["evidence_id"])["body"]
            assert manifest["evidence_kind"] == "capture"
            assert any(item["format"] == "samples" for item in manifest["inputs"])
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_capture_reports_storage_reserve_and_observed_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    checks = 0

    def disk_usage(_path: object) -> SimpleNamespace:
        nonlocal checks
        checks += 1
        free = 128 * 1024 * 1024 if checks == 1 else 1024 * 1024
        return SimpleNamespace(total=256 * 1024 * 1024, used=0, free=free)

    monkeypatch.setattr("flameox.execution.shutil.disk_usage", disk_usage)

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "import time; print('partial'); time.sleep(10)"],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
            )
            limit = result["capture"]["executions"][0]["limit"]
            assert limit["kind"] == "storage_reserve_exceeded"
            assert limit["configured"] == 64 * 1024 * 1024
            assert limit["observed"] == 1024 * 1024
            assert limit["observation_available"] is True
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_stale_external_inputs_do_not_break_unrelated_analysis_eviction(tmp_path: Path) -> None:
    stale = tmp_path / "stale.txt"
    stale.write_text("old observation")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        runtime.analyze("artifact.preview", [PathSource(path=str(stale))], {})
        stale.unlink()
        stale.symlink_to(stale)
        for index in range(65):
            artifact = tmp_path / f"current-{index}.txt"
            artifact.write_text(f"observation {index}")
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            assert result["blocks"][1]["rows"][0]["text"] == f"observation {index}"
        preserved = runtime.preserve_evidence(result["analysis_id"])
        assert runtime.read_evidence(preserved["evidence_id"])
    finally:
        runtime.close()


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0, reason="Requires enforced POSIX directory permissions"
)
def test_unreadable_directory_members_prevent_analysis_and_preservation(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    hidden = bundle / "nested"
    hidden.mkdir(parents=True)
    (bundle / "visible.txt").write_text("visible")
    (hidden / "native.txt").write_text("native")
    store = tmp_path / "store"
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        source = PathSource(path=str(bundle))
        readable = runtime.analyze("artifact.preview", [source], {})
        assert readable["coverage"]["complete"] is True
        hidden.chmod(0)
        with pytest.raises(RuntimeFailure) as analysis_failure:
            runtime.analyze("artifact.preview", [source], {})
        assert analysis_failure.value.code == "INVALID_INPUT"
        with pytest.raises(RuntimeFailure) as preservation_failure:
            runtime.preserve_evidence(readable["analysis_id"])
        assert preservation_failure.value.code == "INVALID_INPUT"
        assert not list(store.rglob("*.json"))
    finally:
        hidden.chmod(0o700)
        runtime.close()


@pytest.mark.process
def test_live_scratch_file_churn_does_not_interrupt_other_analysis(tmp_path: Path) -> None:
    import subprocess
    import time

    artifact = tmp_path / "input.txt"
    artifact.write_text("stable input")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    active = runtime.scratch / "capture-active"
    active.mkdir()
    ready = tmp_path / "ready"
    workload = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pathlib import Path\n"
            f"p = Path({str(active / 'temporary')!r})\n"
            f"Path({str(ready)!r}).touch()\n"
            "while True:\n    p.write_bytes(b'x')\n    p.unlink()\n",
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        for _ in range(300):
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            assert result["coverage"]["complete"] is True
        assert workload.poll() is None
    finally:
        workload.terminate()
        workload.wait(timeout=5)
        runtime.close()


@pytest.mark.parametrize("provider", ["perfetto", "nsight-compute"])
@pytest.mark.parametrize("symlink", [False, True])
def test_projection_cache_and_continuations_bind_external_reader_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, symlink: bool
) -> None:
    from typing import Any

    from flameox.providers.contracts import ProviderAnalysis
    from flameox.providers.nsight_compute import NsightComputeProvider
    from flameox.providers.perfetto import PerfettoProvider

    reader = tmp_path / ("trace_processor_shell" if provider == "perfetto" else "ncu_report.py")
    reader.write_text("#!/bin/sh\n# first reader\nexit 0\n")
    reader.chmod(0o755)
    if symlink:
        canonical = reader.with_name("canonical-reader")
        reader.rename(canonical)
        try:
            reader.symlink_to(canonical)
        except OSError:
            pytest.skip("Creating a symlink requires platform privileges")
    monkeypatch.setenv("PATH", str(tmp_path))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    selected: PerfettoProvider | NsightComputeProvider
    if provider == "perfetto":
        monkeypatch.setenv("FLAMEOX_TRACE_PROCESSOR", str(reader))
        selected = runtime.perfetto
        capability, format_name = "trace.summary", "chrome-trace"
    else:
        runtime.nsight_compute.interface_path = reader
        selected = runtime.nsight_compute
        capability, format_name = "gpu.kernel_metrics", "nsight-compute"
    artifact = tmp_path / "input.json"
    artifact.write_text('{"traceEvents":[]}')
    sources = [PathSource(path=str(artifact), format=format_name)]
    calls = 0

    def project(*_args: Any, **_kwargs: Any) -> ProviderAnalysis:
        nonlocal calls
        calls += 1
        identity = selected.projection_identity()
        return ProviderAnalysis(
            provider_id=provider,
            provider_version=identity,
            blocks=[
                {
                    "type": "table",
                    "rows": [{"index": index, "reader": identity} for index in range(2)],
                }
            ],
            rows_observed=2,
            complete=True,
            limitations=[],
        )

    monkeypatch.setattr(selected, "analyze", project)
    try:
        first = runtime.analyze(capability, sources, {}, limits=RequestLimits(max_rows=1))
        assert runtime.analyze(capability, sources, {}, limits=RequestLimits(max_rows=1)) == first
        assert calls == 1
        reader.write_text("#!/bin/sh\n# replacement reader\nexit 0\n")
        with pytest.raises(RuntimeFailure) as continuation_failure:
            runtime.analyze(
                capability,
                sources,
                {},
                limits=RequestLimits(max_rows=1),
                continuation=first["continuation"],
            )
        assert continuation_failure.value.code == "INVALID_INPUT"
        second = runtime.analyze(capability, sources, {}, limits=RequestLimits(max_rows=1))
        assert calls == 2
        assert second["blocks"][0]["rows"][0]["reader"] != first["blocks"][0]["rows"][0]["reader"]
        reader.unlink()
        with pytest.raises(RuntimeFailure) as missing_failure:
            runtime.analyze(capability, sources, {}, limits=RequestLimits(max_rows=1))
        assert missing_failure.value.code == "UNAVAILABLE_CAPABILITY"
        assert calls == 2
    finally:
        runtime.close()
