# ruff: noqa: E501 - the fake official interface is embedded source, not product code
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.nsight_compute import NsightComputeExtractor, find_ncu_report_interface
from flameox.adapters.options import bind_adapter_options
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import ArtifactStore, RunStore, Workspace


def _fake_interface(path: Path, *, corrupt: bool = False) -> Path:
    path.mkdir()
    body = """
class Metric:
    ValueKind_UINT32 = 1
    ValueKind_UINT64 = 2
    ValueKind_FLOAT = 3
    ValueKind_DOUBLE = 4
    ValueKind_STRING = 5
    def __init__(self, name, kind, value, unit): self._name, self._kind, self._value, self._unit = name, kind, value, unit
    def has_value(self): return True
    def kind(self): return self._kind
    def as_uint64(self): return int(self._value)
    def as_double(self): return float(self._value)
    def as_string(self): return str(self._value)
    def value(self): return self._value
    def unit(self): return self._unit
    def description(self): return 'official metric description'
    def metric_type(self): return 2
    def metric_subtype(self):
        if self._name == 'unknown': raise RuntimeError('optional subtype unavailable')
        return None
    def rollup_operation(self): return 1

class Action:
    def __init__(self):
        self.metrics = {
            'sm__cycles_elapsed.avg': Metric('cycles', 2, 42, 'cycle'),
            'device__attribute_display_name': Metric('device', 5, 'Fake GPU', ''),
            'unsupported.metric': Metric('unknown', 99, object(), ''),
        }
    def name(self): return 'vector_add'
    def workload_type(self):
        def values():
            for index in range(100): yield index
            raise AssertionError('provider iterable was consumed past its bound')
        return values()
    def metric_names(self): return tuple(self.metrics)
    def metric_by_name(self, name): return self.metrics[name]
    def rule_results_as_dicts(self):
        return [
            {
                'rule_identifier': 'ExplicitRoofline',
                'section_identifier': 'SpeedOfLight_RooflineChart',
                'k' * 250: 'v' * 300,
            },
            {'deep': {'a': {'b': {'c': {'d': {'e': 'end'}}}}}},
        ]
    def source_files(self): return {'0': 'kernel.cu'}
    def source_markers(self): return [{'source_address': 1234, 'message': 'source marker'}]
    def source_info(self, address): return {'file': 'kernel.cu', 'line': 7, 'address': address}
    def sass_by_pc(self, address): return {'address': address, 'instruction': 'LDG.E'}
    def ptx_by_pc(self, address): return {'address': address, 'instruction': 'ld.global'}

class Range:
    def num_actions(self): return 1
    def action_by_idx(self, index): return Action()

class Report:
    def get_version(self): return '2099.1'
    def num_ranges(self): return 1
    def range_by_idx(self, index): return Range()

def load_report(path):
    return Report()
"""
    if corrupt:
        body = "def load_report(path):\n    raise RuntimeError('corrupt official report')\n"
    interface = path / "ncu_report.py"
    interface.write_text(body, encoding="utf-8")
    return interface


def _multi_range_interface(path: Path) -> Path:
    path.mkdir()
    interface = path / "ncu_report.py"
    interface.write_text(
        """
class Action:
    def name(self): return 'bounded_action'
    def workload_type(self): return None
    def metric_names(self): return ()
    def rule_results_as_dicts(self): return ()
    def source_files(self): return {}
    def source_markers(self): return ()

class Range:
    def __init__(self, count): self.count = count
    def name(self): return 'range'
    def num_actions(self): return self.count
    def action_by_idx(self, index): return Action()

class Report:
    def get_version(self): return '2099.1'
    def num_ranges(self): return 2
    def range_by_idx(self, index): return Range(2 if index == 0 else 1)

def load_report(path): return Report()
""",
        encoding="utf-8",
    )
    return interface


def _source_files_interface(path: Path) -> Path:
    path.mkdir()
    interface = path / "ncu_report.py"
    interface.write_text(
        """
class Action:
    def __init__(self, index): self.index = index
    def name(self): return 'mapping' if self.index == 0 else 'sequence'
    def workload_type(self): return None
    def metric_names(self): return ()
    def rule_results_as_dicts(self): return ()
    def source_files(self):
        if self.index == 0: return {str(index): f'kernel-{index}.cu' for index in range(9)}
        return ['first.cu', 'second.cu']
    def source_markers(self): return ()

class Range:
    def num_actions(self): return 2
    def action_by_idx(self, index): return Action(index)

class Report:
    def get_version(self): return '2099.1'
    def num_ranges(self): return 1
    def range_by_idx(self, index): return Range()

def load_report(path): return Report()
""",
        encoding="utf-8",
    )
    return interface


def _source_markers_interface(path: Path) -> Path:
    path.mkdir()
    interface = path / "ncu_report.py"
    interface.write_text(
        """
class Action:
    def name(self): return 'markers'
    def workload_type(self): return None
    def metric_names(self): return ()
    def rule_results_as_dicts(self): return ()
    def source_files(self): return {}
    def source_markers(self):
        return [
            {'source_address': 1, 'message': 'first'},
            {'source_address': 2, 'message': 'second'},
        ]
    def source_info(self, address): return {'line': address}
    def sass_by_pc(self, address): return {'sass': address}
    def ptx_by_pc(self, address): return {'ptx': address}

class Range:
    def num_actions(self): return 1
    def action_by_idx(self, index): return Action()

class Report:
    def get_version(self): return '2099.1'
    def num_ranges(self): return 1
    def range_by_idx(self, index): return Range()

def load_report(path): return Report()
""",
        encoding="utf-8",
    )
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


def test_strict_options_build_only_documented_bounded_arguments(tmp_path: Path) -> None:
    selected = bind_adapter_options(
        "nsight.compute",
        {
            "sections": ["LaunchStats", "SpeedOfLight"],
            "kernel_name": "vector_add",
            "launch_skip": 2,
            "launch_count": 3,
            "replay_mode": "application",
        },
        project_root=tmp_path,
    )
    invocation = build_capture_invocation(
        "nsight.compute",
        ("./workload",),
        tmp_path,
        executable="/opt/nvidia/ncu",
        options=cast(dict[str, object], selected),
        project_root=tmp_path,
    )

    assert invocation.artifact_kinds == (ArtifactKind.KERNEL_PROFILE,)
    assert invocation.argv == (
        "/opt/nvidia/ncu",
        "--export",
        str(tmp_path / "nsight-compute.ncu-rep"),
        "--force-overwrite",
        "--replay-mode",
        "application",
        "--launch-skip",
        "2",
        "--launch-count",
        "3",
        "--section",
        "LaunchStats",
        "--section",
        "SpeedOfLight",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "vector_add",
        "./workload",
    )

    with pytest.raises(DomainError) as unknown:
        bind_adapter_options(
            "nsight.compute",
            {"set": "basic", "arbitrary_flags": ["--import"]},
            project_root=tmp_path,
        )
    assert unknown.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    with pytest.raises(DomainError):
        bind_adapter_options(
            "nsight.compute",
            {"set": None, "sections": ["regex:.*"]},
            project_root=tmp_path,
        )


def test_interface_next_to_selected_executable_wins_over_newer_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_root = tmp_path / "2025.1"
    executable = selected_root / "bin" / "ncu"
    executable.parent.mkdir(parents=True)
    executable.touch()
    selected_interface = selected_root / "extras" / "python" / "ncu_report.py"
    selected_interface.parent.mkdir(parents=True)
    selected_interface.touch()
    installation_root = tmp_path / "installations"
    matching_interface = installation_root / "2025.1" / "extras" / "python" / "ncu_report.py"
    matching_interface.parent.mkdir(parents=True)
    matching_interface.touch()
    newer_interface = installation_root / "2099.1" / "extras" / "python" / "ncu_report.py"
    newer_interface.parent.mkdir(parents=True)
    newer_interface.touch()
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute._NCU_INSTALL_ROOTS",
        (installation_root,),
    )

    selected = find_ncu_report_interface(
        executable=executable,
        producer_version="Version 2025.1.0",
    )

    assert selected == selected_interface
    assert find_ncu_report_interface(producer_version="Version 2025.1.0") == matching_interface


def test_extractor_uses_fake_official_interface_in_isolated_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "fake.ncu-rep"
    source.write_bytes(b"opaque NVIDIA report bytes")
    before = source.read_bytes()
    run_id = _import_report(workspace, source)
    interface = _fake_interface(tmp_path / "official-interface")
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
    assert result.observation_count >= 5
    assert result.roofline_present is True
    assert any("Unsupported metric value type" in item for item in result.limitations)
    assert any("metric_subtype failed with RuntimeError" in item for item in result.limitations)
    assert any("collections were bounded" in item for item in result.limitations)
    assert any("values were bounded to 8 nodes" in item for item in result.limitations)
    assert any("values were bounded to depth 5" in item for item in result.limitations)
    assert any("keys were bounded" in item for item in result.limitations)
    assert any("strings were bounded" in item for item in result.limitations)
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
            "SELECT evidence_level FROM observations WHERE kind = 'profile.extraction'"
        ).fetchone()
        source_files_row = snapshot.execute(
            "SELECT value_json FROM observations WHERE kind = 'profile.source_files'"
        ).fetchone()
    assert provenance_row is not None
    assert source_files_row is not None
    assert provenance_row[0] == "derived"
    assert json.loads(source_files_row[0]) == {"files": [{"id": "0", "path": "kernel.cu"}]}


def test_extractor_clamps_rows_to_worker_response_budget(
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
    interface = _fake_interface(tmp_path / "bounded-interface")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )
    request: dict[str, object] = {}

    def run_sync(_worker: object, _module: str, payload: dict[str, object], **_: object) -> object:
        request.update(payload)
        return {
            "measurements": [],
            "observations": [],
            "metric_ids": [],
            "section_ids": [],
            "limitations": [],
            "report_version": "2099.1",
            "range_count": 0,
            "action_count": 0,
            "roofline_present": False,
        }

    monkeypatch.setattr("flameox.adapters.nsight_compute.ArtifactWorker.run_sync", run_sync)

    NsightComputeExtractor(workspace).extract(run_id)

    assert request["max_metrics"] == 1
    assert request["max_observations"] == 1


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
    interface = _multi_range_interface(tmp_path / "multi-range-interface")
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
    interface = _source_files_interface(tmp_path / "source-files-interface")
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
    interface = _source_markers_interface(tmp_path / "source-markers-interface")
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
    interface = _fake_interface(tmp_path / "corrupt-interface", corrupt=True)
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
