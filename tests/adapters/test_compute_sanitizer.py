from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.compute_sanitizer import ComputeSanitizerExtractor
from flameox.adapters.options import bind_adapter_options, run_semantics
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    RunSemanticsProjection,
)
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


def _report(*records: str, extra: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<ComputeSanitizerOutput>" + "".join(records) + extra + "</ComputeSanitizerOutput>"
    )


def _precise_record(*, unknown: str = "") -> str:
    return f"""
    <record>
      <kind>Precise</kind><level>Error</level>
      <who><threadIdx><x>0</x><y>1</y><z>2</z></threadIdx>
        <blockIdx><x>3</x><y>4</y><z>5</z></blockIdx></who>
      <what><text>Invalid __global__ write of size 4 bytes: Access is out of bounds</text>
        <space>__global__</space><size>4</size><direction>write</direction>
        <error>out of bounds</error><address>0x1234</address></what>
      <where><func>write_values</func><path>/project/src/kernel.cu</path>
        <line>8</line><pc>0xe0</pc></where>
      <hostStack><saveLocation>kernel launch time</saveLocation>
        <frame><func>main</func><path>/project/src/main.cc</path><line>12</line>
          <module>/project/build/app</module><pc>0x1</pc></frame></hostStack>
      {unknown}
    </record>
    """


def _import(
    workspace: Workspace,
    source: Path,
    *,
    producer: str = "compute-sanitizer",
    producer_version: str | None = "2026.2.1",
) -> str:
    return (
        ImportService(workspace)
        .import_artifact(
            ImportArtifactRequest(
                path=source,
                kind=ArtifactKind.SANITIZER_REPORT,
                producer=producer,
                producer_version=producer_version,
            )
        )
        .run.run_id
    )


def test_compute_sanitizer_treats_wrong_producer_as_malformed_not_absent(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "report.xml"
    source.write_text(_report())
    run_id = _import(workspace, source, producer="coverage")

    with pytest.raises(DomainError) as error:
        ComputeSanitizerExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_compute_sanitizer_extracts_precise_memory_findings(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "report.xml"
    source.write_text(_report(_precise_record().replace("/project", str(tmp_path))))
    run_id = _import(workspace, source)

    result = ComputeSanitizerExtractor(workspace).extract(run_id)
    repeated = ComputeSanitizerExtractor(workspace).extract(run_id)

    assert result.status == "findings"
    assert result.finding_count == 1
    assert result.classifications == {"memory_access": 1}
    assert result.limitations == ()
    assert repeated.corpus_commit_id == result.corpus_commit_id
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, file, line_from, value_json FROM observations "
            "WHERE kind = 'sanitizer.finding'"
        ).fetchall()
    assert rows[0][:3] == ("memory_access", "src/kernel.cu", 8)
    assert '"thread":{"x":0,"y":1,"z":2}' in rows[0][3]


def test_compute_sanitizer_accepts_clean_empty_report(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "clean.xml"
    source.write_text('<?xml version="1.0"?><ComputeSanitizerOutput/>')

    result = ComputeSanitizerExtractor(workspace).extract(
        _import(
            workspace,
            source,
            producer_version="Version 2026.2.1.0 (build 38334959) (public-release)",
        )
    )

    assert result.status == "clean"
    assert result.finding_count == 0
    assert result.semantics.origin == "import"
    assert result.semantics.unavailable_fields == ("configuration", "scope")


@pytest.mark.parametrize(
    ("producer_version", "expected_limitation"),
    (
        (None, "producer version is unavailable"),
        ("2025.1.0", "outside the verified 2026 compatibility family"),
        ("2027.0", "outside the verified 2026 compatibility family"),
        ("unknown", "producer version is not identifiable"),
    ),
    ids=("missing", "legacy", "future", "unidentifiable"),
)
def test_compute_sanitizer_marks_unverified_clean_reports_inconclusive(
    tmp_path: Path,
    producer_version: str | None,
    expected_limitation: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "clean.xml"
    source.write_text('<?xml version="1.0"?><ComputeSanitizerOutput/>')

    run_id = _import(workspace, source, producer_version=producer_version)

    result = ComputeSanitizerExtractor(workspace).extract(run_id)

    assert result.status == "inconclusive"
    assert len(result.limitations) == 1
    assert expected_limitation in result.limitations[0]


def test_compute_sanitizer_reports_unknown_record_shapes(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "unknown.xml"
    source.write_text(_report(_precise_record(unknown="<futureField>value</futureField>")))

    result = ComputeSanitizerExtractor(workspace).extract(_import(workspace, source))

    assert result.finding_count == 1
    assert "Unknown Compute Sanitizer record element: futureField." in result.limitations


@pytest.mark.parametrize(
    "payload",
    (
        "<ComputeSanitizerOutput><record>",
        "<OtherOutput/>",
        '<!DOCTYPE x [<!ENTITY boom "x">]><ComputeSanitizerOutput>&boom;</ComputeSanitizerOutput>',
        "<!--" + ("padding" * 800) + "-->"
        '<!DOCTYPE x [<!ENTITY boom "x">]><ComputeSanitizerOutput>&boom;</ComputeSanitizerOutput>',
    ),
    ids=("truncated", "wrong-root", "entity", "entity-after-prefix"),
)
def test_compute_sanitizer_rejects_malformed_or_unsafe_xml(
    tmp_path: Path,
    payload: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "bad.xml"
    source.write_text(payload)
    run_id = _import(workspace, source)

    with pytest.raises(DomainError) as error:
        ComputeSanitizerExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_compute_sanitizer_classifies_known_record_kinds(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "known.xml"
    records = [
        _precise_record().replace(
            "Invalid __global__ write of size 4 bytes: Access is out of bounds",
            "Invalid asynchronous memory copy API access",
        )
    ]
    for kind, message in (
        ("Api", "CUDA API call failed"),
        ("Sanitizer", "Sanitizer internal error"),
        ("Race", "Write-after-write hazard"),
        ("Initcheck", "Uninitialized global read"),
        ("Synccheck", "Barrier divergence"),
    ):
        record = _precise_record().replace("<kind>Precise</kind>", f"<kind>{kind}</kind>")
        records.append(
            record.replace(
                "Invalid __global__ write of size 4 bytes: Access is out of bounds",
                message,
            )
        )
    source.write_text(_report(*records))

    result = ComputeSanitizerExtractor(workspace).extract(_import(workspace, source))

    assert result.classifications == {
        "memory_access": 1,
        "api_error": 1,
        "sanitizer_error": 1,
        "race": 1,
        "uninitialized_memory": 1,
        "synchronization": 1,
    }


def test_compute_sanitizer_truncates_records_at_publication_budget(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "storage": workspace.config.storage.model_copy(update={"max_rows_per_generation": 4})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    source = tmp_path / "many.xml"
    source.write_text(
        _report(
            _precise_record(),
            _precise_record(),
            _precise_record(),
            _precise_record(),
        )
    )

    result = ComputeSanitizerExtractor(workspace).extract(_import(workspace, source))

    assert result.finding_count == 3
    assert "Sanitizer records were truncated to 3 entries." in result.limitations


def test_compute_sanitizer_options_are_strict_and_bind_suppression_digest(
    tmp_path: Path,
) -> None:
    suppression = tmp_path / "tools" / "sanitizer.supp"
    suppression.parent.mkdir()
    suppression.write_text("# fixture\n")

    bound = bind_adapter_options(
        "compute-sanitizer",
        {
            "tool": "racecheck",
            "launch_skip": 2,
            "launch_count": 3,
            "target_processes": "all",
            "target_processes_filter": "regex:worker-[0-9]+",
            "kernel_name": "kernel_substring=attention",
            "demangle": "simple",
            "suppression_file": "tools/sanitizer.supp",
        },
        project_root=tmp_path,
    )

    assert bound["suppression_file"] == "tools/sanitizer.supp"
    assert str(bound["suppression_digest"]).startswith("sha256:")
    semantics = run_semantics(
        "compute-sanitizer",
        "2026.2.1",
        bound,
    )
    projected = RunSemanticsProjection.from_semantics(semantics)
    assert projected.mode == "racecheck"
    assert projected.process_scope == "all"
    assert projected.bounds == {"launch_count": 3, "launch_skip": 2}
    assert projected.filters == {
        "kernel_name": "kernel_substring=attention",
        "suppression_digest": bound["suppression_digest"],
        "target_processes_filter": "regex:worker-[0-9]+",
    }
    synccheck = RunSemanticsProjection.from_semantics(
        run_semantics(
            "compute-sanitizer",
            "2026.2.1",
            {**bound, "tool": "synccheck"},
        )
    )
    assert synccheck.semantic_id != projected.semantic_id
    assert synccheck.mode == "synccheck"
    invocation = build_capture_invocation(
        "compute-sanitizer",
        ("python", "kernel.py"),
        tmp_path / "output",
        executable="/opt/cuda/bin/compute-sanitizer",
        options=cast(dict[str, object], bound),
        project_root=tmp_path,
    )
    assert invocation.argv == (
        "/opt/cuda/bin/compute-sanitizer",
        "--tool",
        "racecheck",
        "--xml",
        "--save",
        str(tmp_path / "output" / "compute-sanitizer.xml"),
        "--error-exitcode",
        "86",
        "--launch-skip",
        "2",
        "--launch-count",
        "3",
        "--target-processes",
        "all",
        "--demangle",
        "simple",
        "--target-processes-filter",
        "regex:worker-[0-9]+",
        "--kernel-name",
        "kernel_substring=attention",
        "--suppressions",
        str(suppression),
        "python",
        "kernel.py",
    )
    suppression.write_text("# changed after planning\n")
    with pytest.raises(DomainError) as changed:
        build_capture_invocation(
            "compute-sanitizer",
            ("python", "kernel.py"),
            tmp_path / "output",
            executable="/opt/cuda/bin/compute-sanitizer",
            options=cast(dict[str, object], bound),
            project_root=tmp_path,
        )
    assert changed.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    with pytest.raises(DomainError) as error:
        bind_adapter_options(
            "compute-sanitizer",
            {"arbitrary_flag": "--destroy-on-device-error=kernel"},
            project_root=tmp_path,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_compute_sanitizer_rejects_oversized_suppression_during_planning(
    tmp_path: Path,
) -> None:
    suppression = tmp_path / "oversized.supp"
    suppression.write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(DomainError, match="exceeds the 1 MiB limit") as error:
        bind_adapter_options(
            "compute-sanitizer",
            {"suppression_file": suppression.name},
            project_root=tmp_path,
        )

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


def test_compute_sanitizer_rejects_suppression_escape_and_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.supp"
    outside.write_text("fixture")
    with pytest.raises(DomainError):
        bind_adapter_options(
            "compute-sanitizer",
            {"suppression_file": "../outside.supp"},
            project_root=tmp_path,
        )
    link = tmp_path / "linked.supp"
    link.symlink_to(outside)
    with pytest.raises(DomainError):
        bind_adapter_options(
            "compute-sanitizer",
            {"suppression_file": "linked.supp"},
            project_root=tmp_path,
        )
