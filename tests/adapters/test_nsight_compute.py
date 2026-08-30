from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flameox.adapters.nsight_compute import NsightComputeExtractor
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import ArtifactStore, RunStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process]

_INTERFACE_FIXTURES = Path(__file__).parent.parent / "fixtures" / "nsight_compute"


def _interface_fixture(path: Path, name: str) -> Path:
    path.mkdir()
    interface = path / "ncu_report.py"
    shutil.copyfile(_INTERFACE_FIXTURES / f"{name}.py.txt", interface)
    return interface


def _import_report(workspace: Workspace, path: Path) -> str:
    return (
        ImportService(workspace)
        .import_artifact(
            ImportArtifactRequest(
                path=path,
                kind=ArtifactKind.KERNEL_PROFILE,
                producer="nsight.compute",
                producer_version="2099.1",
            )
        )
        .run.run_id
    )


def test_extractor_uses_fake_official_interface_in_isolated_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "fake.ncu-rep"
    source.write_bytes(b"opaque NVIDIA report bytes")
    before = source.read_bytes()
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "official-interface", "basic")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    result = NsightComputeExtractor(workspace).extract(run_id)
    repeated = NsightComputeExtractor(workspace).extract(run_id)

    assert result.report_version == "2099.1"
    assert result.range_count == 1
    assert result.action_count == 1
    assert result.metric_count == 1
    assert result.observation_count >= 7
    assert result.truncated is False
    assert any("Unsupported metric value type" in item for item in result.limitations)
    assert any("metric_subtype failed with RuntimeError" in item for item in result.limitations)
    assert repeated.corpus_commit_id == result.corpus_commit_id
    assert repeated.report_interface_sha256 == result.report_interface_sha256
    interface.write_text(interface.read_text() + "\n# reader revision\n", encoding="utf-8")
    revised = NsightComputeExtractor(workspace).extract(run_id)
    assert revised.report_interface_sha256 != result.report_interface_sha256
    assert revised.schema_fingerprint != result.schema_fingerprint
    assert revised.corpus_commit_id != result.corpus_commit_id
    registration = next(item for item in RunStore(workspace).read(run_id).artifacts)
    assert (
        ArtifactStore(workspace).get(registration.artifact_id).payload_path.read_bytes() == before
    )
    with Catalog(workspace).open_snapshot() as snapshot:
        provenance_row = snapshot.execute(
            "SELECT evidence_level, value_json FROM observations WHERE kind = 'profile.extraction'"
        ).fetchone()
        rule_rows = snapshot.execute(
            "SELECT value_json FROM observations WHERE kind = 'nsight_compute.rule' "
            "ORDER BY observation_id"
        ).fetchall()
    assert provenance_row is not None
    assert provenance_row[0] == "derived"
    assert json.loads(str(provenance_row[1]))["section_ids"] == [
        "SpeedOfLight",
        "SpeedOfLight_RooflineChart",
    ]
    assert sorted(
        (json.loads(value) for (value,) in rule_rows),
        key=lambda fact: str(fact["rule_identifier"]),
    ) == [
        {
            "focus_metrics": [
                {
                    "info": "Inspect the roofline chart.",
                    "name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "severity": "HIGH",
                    "value": 84.56,
                }
            ],
            "location": {"action_index": 0, "action_name": "vector_add", "range_index": 0},
            "rule_identifier": "ExplicitRoofline",
            "rule_message": {
                "message": "Arithmetic intensity is below the available roofline.",
                "provider_type": "OPTIMIZATION",
                "title": "Roofline analysis",
            },
            "section_identifier": "SpeedOfLight",
            "speedup_estimation": {
                "estimated_speedup": 42.38,
                "meaning": "global_runtime_reduction",
                "provider_type": "GLOBAL",
            },
        },
        {
            "focus_metrics": [
                {
                    "info": "Review memory access efficiency.",
                    "name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "severity": "HIGH",
                    "value": 93.2,
                }
            ],
            "location": {"action_index": 0, "action_name": "vector_add", "range_index": 0},
            "rule_identifier": "MemoryPipelines",
            "rule_message": {
                "message": "Memory pipelines limit local hardware efficiency.",
                "provider_type": "WARNING",
                "title": "Memory throughput",
            },
            "section_identifier": "SpeedOfLight",
            "speedup_estimation": {
                "estimated_speedup": 80.0,
                "meaning": "local_hardware_efficiency_increase",
                "provider_type": "LOCAL",
            },
        },
    ]


def test_extractor_bounds_published_rows_by_worker_response_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"max_output_bytes": 96 * 1024}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    source = tmp_path / "bounded.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "bounded-interface", "basic")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )
    result = NsightComputeExtractor(workspace).extract(run_id)

    assert result.metric_count == 1
    assert result.observation_count == 1
    assert "Metrics were truncated to 1 entries." in result.limitations
    assert "Observations were bounded to 1 entries." in result.limitations


def test_uint_metrics_round_trip_exactly_with_provider_signedness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "uint-boundaries.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "uint-interface", "uint_boundaries")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    result = NsightComputeExtractor(workspace).extract(run_id)
    queried = EvidenceQueryService(workspace).measurements(run_id=run_id, limit=10)

    assert result.metric_count == 5
    values = {item.name: item.value for item in queried.measurements}
    uint32_max = values["uint32.max"]
    assert uint32_max is not None
    assert uint32_max.model_dump(mode="json") == {
        "kind": "unsigned_integer",
        "value": 2**32 - 1,
    }
    for name, expected in (
        ("uint64.signed_boundary", 2**63),
        ("uint64.signed_boundary_plus_one", 2**63 + 1),
        ("uint64.max", 2**64 - 1),
    ):
        value = values[name]
        assert value is not None
        assert value.model_dump(mode="json") == {
            "kind": "unsigned_integer",
            "value": expected,
        }
    fallback = values["fallback.signed"]
    assert fallback is not None
    assert fallback.model_dump(mode="json") == {
        "kind": "integer",
        "value": -42,
    }
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, value_uint, value_kind, dimensions['provider_value_kind'] "
            "FROM measurements ORDER BY name"
        ).fetchall()
    assert ("uint64.max", 2**64 - 1, "unsigned_integer", "uint64") in rows


def test_declared_uint64_outside_provider_range_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "invalid-uint64.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "invalid-uint-interface", "invalid_uint64")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    with pytest.raises(DomainError) as failure:
        NsightComputeExtractor(workspace).extract(run_id)

    assert failure.value.code is ErrorCode.ADAPTER_INCOMPATIBLE
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT count(*) FROM measurements WHERE run_id = ?",
            (run_id,),
        ).fetchone() == (0,)


def test_action_limit_reports_unvisited_later_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "storage": workspace.config.storage.validated_copy(
                update={"max_rows_per_generation": 5}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    source = tmp_path / "multiple-ranges.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "multi-range-interface", "multi_range")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    result = NsightComputeExtractor(workspace).extract(run_id)

    assert result.range_count == 2
    assert result.action_count == 2
    assert "Actions were bounded to 2 entries." in result.limitations


def test_source_files_preserve_mapping_and_sequence_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "source-files.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "source-files-interface", "source_files")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    result = NsightComputeExtractor(workspace).extract(run_id)

    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, value_json FROM observations "
            "WHERE kind = 'profile.source_files' ORDER BY name"
        ).fetchall()
    by_name = {name: json.loads(value) for name, value in rows}
    assert by_name["mapping"]["files"] == [
        {"id": str(index), "path": f"kernel-{index}.cu"} for index in range(8)
    ]
    assert by_name["sequence"]["files"] == [
        {"id": "0", "path": "first.cu"},
        {"id": "1", "path": "second.cu"},
    ]
    assert "Source files were bounded to 8 identifier/path pairs." in result.limitations


def test_source_markers_share_the_observation_budget_and_report_omissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "storage": workspace.config.storage.validated_copy(
                update={"max_rows_per_generation": 7}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    source = tmp_path / "source-markers.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "source-markers-interface", "source_markers")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    result = NsightComputeExtractor(workspace).extract(run_id)

    assert result.observation_count == 3
    assert (
        "Source marker source/SASS/PTX details were truncated by the observation budget."
        in result.limitations
    )
    assert "Source markers were truncated by the observation budget." in result.limitations


def test_corrupt_report_from_official_interface_is_bounded_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "corrupt.ncu-rep"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)
    interface = _interface_fixture(tmp_path / "corrupt-interface", "corrupt")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )

    with pytest.raises(DomainError) as failure:
        NsightComputeExtractor(workspace).extract(run_id)

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "RuntimeError" in failure.value.message


def test_rejects_non_native_report_extension(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "report.bin"
    source.write_bytes(b"opaque")
    run_id = _import_report(workspace, source)

    with pytest.raises(DomainError) as failure:
        NsightComputeExtractor(workspace).extract(run_id)

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
