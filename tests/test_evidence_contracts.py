from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.canonical import canonical_bytes
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    EvidenceSource,
    PathSource,
    RuntimeFailure,
)
from flameox.source_files import bundle_digest


def preserve_bundle(root: Path, name: str = "bundle") -> dict[str, Any]:
    bundle = root / name
    bundle.mkdir()
    (bundle / "foo").write_text("x" * 2048)
    runtime = AnalysisRuntime(evidence_directory=root / "store")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(bundle))], {})
        ref = runtime.preserve_evidence(result["analysis_id"])
        return runtime.read_evidence_agent_projection(ref["evidence_id"])
    finally:
        runtime.close()


def test_live_session_evidence_can_be_rescued_before_corrupt_store_restart(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "input.json"
    artifact.write_text('[{"value": 1}]')
    configured = tmp_path / "configured-store"
    configured.mkdir()
    (configured / "unexpected").write_text("corrupt")
    rescue = tmp_path / "rescue-store"
    runtime = AnalysisRuntime(evidence_directory=configured)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_CORRUPTION"

        rescued = runtime.rescue_evidence(result["analysis_id"], str(rescue))
        assert rescued["next_action"]["environment"] == {"FLAMEOX_DATA_DIR": str(rescue)}
        assert (configured / "unexpected").read_text() == "corrupt"
        rescued["next_action"]["environment"]["FLAMEOX_DATA_DIR"] = "mutated"
        rescued = runtime.rescue_evidence(result["analysis_id"], str(rescue))
        assert rescued["next_action"]["environment"] == {"FLAMEOX_DATA_DIR": str(rescue)}
    finally:
        runtime.close()

    reopened = AnalysisRuntime(evidence_directory=rescue)
    try:
        manifest = reopened.read_evidence(rescued["evidence_id"])
        assert manifest["body"]["capability_id"] == "artifact.preview"
    finally:
        reopened.close()


@pytest.mark.integration
@pytest.mark.process
@pytest.mark.parametrize("provider", ["direct", "node-cpu-profile"])
def test_preserved_capture_can_be_rescued_after_scratch_is_released(
    tmp_path: Path, provider: str
) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    capability_id = "artifact.preview" if provider == "direct" else "cpu.hotspots"

    async def capture() -> dict[str, Any]:
        return await runtime.capture_and_analyze(
            CaptureTarget(
                argv=[sys.executable, "-c", "print('retained evidence')"],
                cwd=str(tmp_path),
                provider_id=provider,
            ),
            capability_id,
        )

    try:
        result = anyio.run(capture)
        reference = runtime.preserve_evidence(result["analysis_id"])
        assert all(not Path(item["path"]).exists() for item in result["inputs"])
        rescued = runtime.rescue_evidence(result["analysis_id"], str(tmp_path / "rescue"))
        assert rescued["evidence_id"] == reference["evidence_id"]
        assert runtime.rescue_evidence(result["analysis_id"], str(tmp_path / "rescue")) == rescued
        assert runtime.preserve_evidence(result["analysis_id"]) == reference
    finally:
        runtime.close()
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "rescue")
    try:
        projection = reopened.read_evidence_agent_projection(rescued["evidence_id"])
        if provider == "direct":
            reanalysis = reopened.analyze(
                capability_id,
                [EvidenceSource.model_validate(item) for item in projection["analysis_sources"]],
                {},
            )
            assert reanalysis["blocks"][1]["rows"][0]["text"] == "retained evidence"
        else:
            assert projection["body"]["capture_request"]["executions"][0]["status"] == "failed"
    finally:
        reopened.close()


@pytest.mark.integration
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_rescue_reports_damaged_preserved_evidence_before_creating_destination(
    tmp_path: Path, damage: str
) -> None:
    artifact = tmp_path / "input.txt"
    artifact.write_text("native contents\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    destination = tmp_path / "rescue"
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        reference = runtime.preserve_evidence(result["analysis_id"])
        source = runtime.repository.select_source(
            reference["evidence_id"], selector=None, role=None
        )
        payload = source.members[0][1].path
        if damage == "missing":
            payload.unlink()
        else:
            payload.write_text("changed bytes")
        with pytest.raises(RuntimeFailure) as failure:
            runtime.rescue_evidence(result["analysis_id"], str(destination))
        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert not destination.exists()
    finally:
        runtime.close()


@pytest.mark.integration
def test_rescue_rejects_configured_or_nonempty_destination_without_losing_handle(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "input.json"
    artifact.write_text("[]")
    configured = tmp_path / "store"
    nonempty = tmp_path / "other"
    nonempty.mkdir()
    (nonempty / "file").write_text("owned")
    runtime = AnalysisRuntime(evidence_directory=configured)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        for destination in (configured, nonempty):
            with pytest.raises(RuntimeFailure) as failure:
                runtime.rescue_evidence(result["analysis_id"], str(destination))
            assert failure.value.code == "INVALID_INPUT"
        assert runtime.preserve_evidence(result["analysis_id"])["evidence_id"]
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "paths,valid",
    [
        (["a", "b"], True),
        (["a", "a/b"], False),
        (["a/b", "a"], False),
        ([r"\foo", "b"], False),
        (["C:foo", "b"], False),
        ([r"a\..\outside", "b"], False),
    ],
)
def test_manifest_layout_is_contained_and_materializable(
    tmp_path: Path, paths: list[str], valid: bool
) -> None:
    projection = preserve_bundle(tmp_path)
    evidence_id = projection["evidence_id"]
    store = tmp_path / "store"
    bundle = store / "evidence" / "sha256" / evidence_id[:2] / evidence_id
    manifest = json.loads((bundle / "manifest.json").read_text())
    body = manifest["body"]
    body["artifacts"].append({**body["artifacts"][0], "role": "second"})
    source = body["source_layout"]["sources"][0]
    source["relative_paths"] = paths
    source["artifact_indices"] = [0, 1]
    source["sha256"] = bundle_digest(
        (relative, member["sha256"])
        for relative, member in zip(paths, body["artifacts"], strict=True)
    )
    source["size_bytes"] *= 2
    body["analysis_request"]["inputs"][0]["sha256"] = source["sha256"]
    body["inputs"][0].update(sha256=source["sha256"], size_bytes=source["size_bytes"])
    changed_id = hashlib.sha256(canonical_bytes(body)).hexdigest()
    manifest["evidence_id"] = changed_id
    destination = bundle.parent.parent / changed_id[:2] / changed_id
    destination.parent.mkdir(exist_ok=True)
    bundle.rename(destination)
    (destination / "manifest.json").write_bytes(canonical_bytes(manifest))
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        if valid:
            selected = runtime.read_evidence_agent_projection(changed_id)
            result = runtime.analyze(
                "artifact.preview",
                [EvidenceSource.model_validate(selected["analysis_sources"][0])],
                {},
            )
            root = Path(result["inputs"][0]["path"])
            assert sorted(path.name for path in root.iterdir()) == paths
        else:
            with pytest.raises(RuntimeFailure) as failure:
                runtime.read_evidence_agent_projection(changed_id)
            assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_source_selection_checks_selected_payload_not_unrelated_analysis_data(
    tmp_path: Path,
) -> None:
    projection = preserve_bundle(tmp_path)
    evidence_id = projection["evidence_id"]
    store = tmp_path / "store"
    bundle = store / "evidence" / "sha256" / evidence_id[:2] / evidence_id
    (bundle / "data" / "analysis.json").write_text("corrupted")
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        result = runtime.analyze(
            "artifact.preview",
            [EvidenceSource.model_validate(projection["analysis_sources"][0])],
            {},
        )
        assert result["blocks"][1]["rows"][0]["path"] == "foo"
        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence_agent_projection(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_rescue_rejects_unresolvable_destination_before_publication(tmp_path: Path) -> None:
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preflight_rescue_destination(str(loop / "new"))
        assert failure.value.code == "INVALID_INPUT"
        assert str(tmp_path) not in failure.value.message
        assert not (tmp_path / "store").exists()
    finally:
        runtime.close()
