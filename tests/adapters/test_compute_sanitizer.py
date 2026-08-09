from __future__ import annotations

import stat
from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.compute_sanitizer import ComputeSanitizerExtractor
from flameox.adapters.options import bind_adapter_options
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    ImportArtifactRequest,
    ImportService,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    CapabilityReport,
    CapabilityStatus,
    DomainError,
    ErrorCode,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment


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


def _import(workspace: Workspace, source: Path) -> str:
    return (
        ImportService(workspace)
        .import_artifact(
            ImportArtifactRequest(
                path=source,
                kind=ArtifactKind.SANITIZER_REPORT,
                producer="compute-sanitizer",
                producer_version="2026.2.1",
            )
        )
        .run.run_id
    )


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

    result = ComputeSanitizerExtractor(workspace).extract(_import(workspace, source))

    assert result.status == "clean"
    assert result.finding_count == 0


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


def test_compute_sanitizer_uses_record_kind_before_message_substrings(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "async.xml"
    source.write_text(
        _report(
            _precise_record().replace(
                "Invalid __global__ write of size 4 bytes: Access is out of bounds",
                "Invalid asynchronous memory copy API access",
            )
        )
    )

    result = ComputeSanitizerExtractor(workspace).extract(_import(workspace, source))

    assert result.classifications == {"memory_access": 1}


@pytest.mark.parametrize(
    ("kind", "message", "classification"),
    (
        ("Api", "CUDA API call failed", "api_error"),
        ("Sanitizer", "Sanitizer internal error", "sanitizer_error"),
        ("Race", "Write-after-write hazard", "race"),
        ("Initcheck", "Uninitialized global read", "uninitialized_memory"),
        ("Synccheck", "Barrier divergence", "synchronization"),
    ),
)
def test_compute_sanitizer_classifies_known_record_kinds(
    tmp_path: Path,
    kind: str,
    message: str,
    classification: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "known.xml"
    record = _precise_record().replace("<kind>Precise</kind>", f"<kind>{kind}</kind>")
    record = record.replace(
        "Invalid __global__ write of size 4 bytes: Access is out of bounds",
        message,
    )
    source.write_text(_report(record))

    result = ComputeSanitizerExtractor(workspace).extract(_import(workspace, source))

    assert result.classifications == {classification: 1}


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


@pytest.mark.anyio
async def test_compute_sanitizer_rejects_suppression_changed_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sanitizer = tmp_path / "fake-compute-sanitizer"
    marker = tmp_path / "sanitizer-ran"
    sanitizer.write_text(f"#!/bin/sh\ntouch {marker}\n")
    sanitizer.chmod(sanitizer.stat().st_mode | stat.S_IXUSR)
    suppression = tmp_path / "sanitizer.supp"
    suppression.write_text("# planned bytes\n")
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n"
    )
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    capability = CapabilityReport(
        adapter="compute-sanitizer",
        status=CapabilityStatus.AVAILABLE,
        executable=str(sanitizer),
        version="fixture",
        supported_modes=("memcheck", "racecheck", "initcheck", "synccheck"),
        supported_formats=("compute-sanitizer-xml",),
        permission_status="granted",
        probe_kind="active",
    )
    service = CaptureService(workspace)
    monkeypatch.setattr(service.capabilities, "get", lambda _adapter: capability)

    async def probe(_adapter: str, *, refresh: bool = False) -> CapabilityReport:
        assert refresh
        return capability

    monkeypatch.setattr(service.capabilities, "probe", probe)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        adapter_options={"suppression_file": "sanitizer.supp"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    suppression.write_text("# bytes changed after authorization\n")

    with pytest.raises(DomainError) as changed:
        await service.execute(plan.plan_id)
    assert changed.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert not marker.exists()


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
