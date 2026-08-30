from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.action_graph import ActionId, ManualAction
from flameox.adapters.nsight_compute import NsightComputeExtractor
from flameox.analysis import RecipeService
from flameox.analysis.recipe_nsight_compute import NsightComputeRecipes
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.cli import app
from flameox.domain import ArtifactKind
from flameox.evidence import GenerationPublisher
from flameox.nsight_compute import NsightComputeProviderRuleFact, NsightComputeReportLocation
from flameox.storage import ArtifactStore, Workspace
from tests.support.analysis import run_row

pytestmark = [pytest.mark.integration, pytest.mark.process]

_INTERFACE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "nsight_compute" / "basic.py.txt"


def _fake_interface(path: Path) -> Path:
    path.mkdir()
    interface = path / "ncu_report.py"
    shutil.copyfile(_INTERFACE_FIXTURE, interface)
    return interface


def test_nsight_compute_projection_reads_typed_evidence_without_reopening_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    report = tmp_path / "profile.ncu-rep"
    report.write_bytes(b"immutable report bytes")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=report,
            kind=ArtifactKind.KERNEL_PROFILE,
            producer="nsight.compute",
            producer_version="2099.1",
        )
    )
    interface = _fake_interface(tmp_path / "official-interface")
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute.find_ncu_report_interface",
        lambda **_: interface,
    )
    NsightComputeExtractor(workspace).extract(imported.run.run_id)

    def fail_if_report_is_opened(*args: object, **kwargs: object) -> None:
        raise AssertionError("analysis must not reopen the native .ncu-rep")

    monkeypatch.setattr(ArtifactStore, "get", fail_if_report_is_opened)
    result = RecipeService(workspace).nsight_compute(imported.run.run_id, limit=1)

    assert result.total == 2
    assert result.returned == 1
    assert result.truncated is True
    finding = result.findings[0]
    assert finding.artifact_id == imported.artifact_id
    assert finding.rule.location.model_dump() == {
        "range_index": 0,
        "action_index": 0,
        "action_name": "vector_add",
    }
    assert finding.rule.rule_identifier == "MemoryPipelines"
    assert finding.rule.section_identifier == "SpeedOfLight"
    assert finding.rule.rule_message is not None
    assert finding.rule.rule_message.provider_type == "WARNING"
    assert finding.rule.speedup_estimation is not None
    assert finding.rule.speedup_estimation.model_dump() == {
        "estimated_speedup": 80.0,
        "meaning": "local_hardware_efficiency_increase",
        "provider_type": "LOCAL",
    }
    assert finding.rule.focus_metrics[0].model_dump() == {
        "name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "value": 93.2,
        "severity": "HIGH",
        "info": "Review memory access efficiency.",
    }
    assert result.target.status == "unqualified"
    assert result.coverage.model_dump() == {
        "section_count": 2,
        "global_runtime_reduction_findings": 1,
        "local_hardware_efficiency_findings": 1,
        "roofline_collected": True,
        "normalized_evidence_truncated": False,
    }
    assert result.recapture is not None
    assert result.recapture.selection.model_dump() == {
        "kernel_name": None,
        "sections": (),
        "replay_mode": "kernel",
    }
    assert isinstance(result.recapture.next_action, ManualAction)
    assert result.recapture.next_action.suggested_action is ActionId.PLAN_CAPTURE
    assert result.recapture.next_action.missing_arguments == ("workload_name", "parameters")

    cli = CliRunner().invoke(
        app,
        [
            "analyze",
            "nsight-compute",
            imported.run.run_id,
            "--limit",
            "1",
            "--workspace",
            str(workspace.paths.root),
            "--json",
        ],
    )

    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout)["findings"][0]["rule"]["speedup_estimation"] == {
        "estimated_speedup": 80.0,
        "meaning": "local_hardware_efficiency_increase",
        "provider_type": "LOCAL",
    }


def test_nsight_compute_analysis_uses_only_the_selected_report_artifact(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    report_artifact_id = "sha256:" + "a" * 64
    unrelated_artifact_id = "sha256:" + "b" * 64
    rule = NsightComputeProviderRuleFact(
        location=NsightComputeReportLocation(
            range_index=0,
            action_index=0,
            action_name="vector_add",
        ),
        rule_identifier="LaunchStats",
        section_identifier="LaunchStats",
    )
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("shared-run")],
            "artifact_registrations": [
                {
                    "registration_id": "ncu-registration",
                    "run_id": "shared-run",
                    "artifact_id": report_artifact_id,
                    "display_name": "profile.ncu-rep",
                    "kind": "kernel_profile",
                    "media_type": "application/octet-stream",
                    "byte_length": 1,
                    "sensitivity": "internal",
                    "role": "primary",
                    "producer": "nsight.compute",
                    "producer_version": "2099.1",
                    "registered_at": run_row("shared-run")["created_at"],
                },
                {
                    "registration_id": "other-registration",
                    "run_id": "shared-run",
                    "artifact_id": unrelated_artifact_id,
                    "display_name": "other-profile.bin",
                    "kind": "kernel_profile",
                    "media_type": "application/octet-stream",
                    "byte_length": 1,
                    "sensitivity": "internal",
                    "role": "supplementary",
                    "producer": "other",
                    "producer_version": "1",
                    "registered_at": run_row("shared-run")["created_at"],
                },
            ],
            "observations": [
                {
                    "observation_id": "ncu-action",
                    "run_id": "shared-run",
                    "artifact_id": report_artifact_id,
                    "kind": "profile.action",
                    "name": "vector_add",
                    "value_json": '{"range_index":0,"action_index":0}',
                    "file": None,
                    "line_from": None,
                    "line_to": None,
                    "context": None,
                    "evidence_level": "observed",
                },
                {
                    "observation_id": "ncu-extraction",
                    "run_id": "shared-run",
                    "artifact_id": report_artifact_id,
                    "kind": "profile.extraction",
                    "name": "ncu",
                    "value_json": (
                        '{"action_count":1,"section_ids":'
                        '["SpeedOfLight_RooflineChart"],"truncated":false}'
                    ),
                    "file": None,
                    "line_from": None,
                    "line_to": None,
                    "context": "extractor_provenance",
                    "evidence_level": "derived",
                },
                {
                    "observation_id": "ncu-rule",
                    "run_id": "shared-run",
                    "artifact_id": report_artifact_id,
                    "kind": "nsight_compute.rule",
                    "name": "LaunchStats",
                    "value_json": rule.model_dump_json(),
                    "file": None,
                    "line_from": None,
                    "line_to": None,
                    "context": None,
                    "evidence_level": "observed",
                },
                {
                    "observation_id": "other-action",
                    "run_id": "shared-run",
                    "artifact_id": unrelated_artifact_id,
                    "kind": "profile.action",
                    "name": "unrelated_action",
                    "value_json": '{"range_index":0,"action_index":0}',
                    "file": None,
                    "line_from": None,
                    "line_to": None,
                    "context": None,
                    "evidence_level": "observed",
                },
            ],
        },
        publisher="ncu-artifact-scope-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).nsight_compute("shared-run", limit=10)

    assert result.total == 1
    assert result.target.observed_action_total == 1
    assert [action.action_name for action in result.target.observed_actions] == ["vector_add"]
    assert result.coverage.section_count == 1
    assert result.coverage.roofline_collected is True


def test_nsight_compute_target_mismatch_is_indeterminate_for_truncated_action_evidence() -> None:
    target = NsightComputeRecipes._target_qualification(
        requested_kernel_name="intended_kernel",
        actions=(
            NsightComputeReportLocation(
                range_index=0,
                action_index=0,
                action_name="incidental_kernel",
            ),
        ),
        action_total=1,
        limit=10,
        action_evidence_truncated=True,
    )

    assert target.status == "indeterminate"
    assert "cannot establish a target mismatch" in target.reason
