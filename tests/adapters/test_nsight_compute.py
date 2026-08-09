# ruff: noqa: E501 - the fake official interface is embedded source, not product code
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.nsight_compute import NsightComputeExtractor
from flameox.adapters.options import bind_adapter_options
from flameox.application import ImportArtifactRequest, ImportService
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
        return [{'rule_identifier': 'ExplicitRoofline', 'section_identifier': 'SpeedOfLight_RooflineChart', 'result_table': {'rows': [1, 2]}}]
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
    assert repeated.corpus_commit_id == result.corpus_commit_id
    registration = next(item for item in RunStore(workspace).read(run_id).artifacts)
    assert (
        ArtifactStore(workspace).get(registration.artifact_id).payload_path.read_bytes() == before
    )


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
