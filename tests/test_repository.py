from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flameox.canonical import canonical_bytes
from flameox.evidence_models import CaptureRequest
from flameox.repository import EvidenceRepository
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "invalid"),
    [("argv", [{}]), ("environment", {"NAME": 1})],
)
def test_repository_rejects_invalid_nested_capture_target(field: str, invalid: Any) -> None:
    target: dict[str, Any] = {
        "argv": ["python"],
        "cwd": "/workspace",
        "environment": {},
        "provider_id": "direct",
        "capture_arguments": {},
        "analysis_arguments": {},
    }
    request: dict[str, Any] = {
        "target": target,
        "mode": "single",
        "experiment": None,
        "executions": [],
    }
    target[field] = invalid

    with pytest.raises(ValidationError):
        CaptureRequest.model_validate(request)


@pytest.mark.integration
def test_missing_repository_metadata_does_not_hide_preserved_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        analysis = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(analysis["analysis_id"])
        (tmp_path / ".flameox" / "repository.json").unlink()

        with pytest.raises(RuntimeFailure) as query_failure:
            runtime.query_evidence()
        assert query_failure.value.code == "REPOSITORY_CORRUPTION"
        with pytest.raises(RuntimeFailure) as read_failure:
            runtime.read_evidence(preserved["evidence_id"])
        assert read_failure.value.code == "REPOSITORY_CORRUPTION"
        second = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
        )
        with pytest.raises(RuntimeFailure) as preserve_failure:
            runtime.preserve_evidence(second["analysis_id"])
        assert preserve_failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_concurrent_identical_publication_reuses_complete_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtimes = [AnalysisRuntime(evidence_directory=tmp_path / ".flameox") for _ in range(8)]
    analyses = [
        runtime.analyze(
            "artifact.preview", [PathSource(path=str(artifact))], {}, limits=RequestLimits()
        )
        for runtime in runtimes
    ]
    # Freeze the episode timestamp so every publication has the same content identity.
    episode = runtimes[0].analyses[analyses[0]["analysis_id"]].manifest_body["episode"]
    for runtime, analysis in zip(runtimes[1:], analyses[1:], strict=True):
        runtime.analyses[analysis["analysis_id"]].manifest_body["episode"] = episode

    try:
        with ThreadPoolExecutor(max_workers=len(runtimes)) as executor:
            results = list(
                executor.map(
                    lambda pair: pair[0].preserve_evidence(pair[1]["analysis_id"]),
                    zip(runtimes, analyses, strict=True),
                )
            )
        assert len({result["evidence_id"] for result in results}) == 1
        assert (
            runtimes[0].read_evidence(results[0]["evidence_id"])["evidence_id"]
            == results[0]["evidence_id"]
        )
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.integration
def test_interrupted_evidence_publication_never_exposes_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    original = EvidenceRepository._publish_directory

    def interrupt_evidence(repository: EvidenceRepository, stage: Path, destination: Path) -> None:
        if "evidence" in destination.parts:
            raise OSError("injected evidence publication interruption")
        original(repository, stage, destination)

    monkeypatch.setattr(EvidenceRepository, "_publish_directory", interrupt_evidence)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_IO_FAILURE"
        assert runtime.query_evidence()["evidence"] == []
        assert not list((tmp_path / ".flameox" / "evidence").rglob("manifest.json"))
    finally:
        runtime.close()


@pytest.mark.integration
def test_interrupted_artifact_publication_never_exposes_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    original = EvidenceRepository._publish_directory

    def interrupt_artifact(repository: EvidenceRepository, stage: Path, destination: Path) -> None:
        if "artifacts" in destination.parts:
            raise OSError("injected artifact publication interruption")
        original(repository, stage, destination)

    monkeypatch.setattr(EvidenceRepository, "_publish_directory", interrupt_artifact)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_IO_FAILURE"
        artifact_root = tmp_path / ".flameox" / "artifacts" / "sha256"
        assert not list(artifact_root.glob("*/*"))
        assert runtime.query_evidence()["evidence"] == []
    finally:
        runtime.close()


@pytest.mark.integration
def test_abandoned_staging_cleanup_removes_only_proven_dead_owner(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        runtime.repository.initialize()
        dead = tmp_path / ".flameox" / ".staging" / "999999999-dead" / "publication"
        unknown = tmp_path / ".flameox" / ".staging" / "unknown" / "publication"
        dead.mkdir(parents=True)
        unknown.mkdir(parents=True)

        runtime.repository.cleanup_abandoned_staging()

        assert not dead.parent.exists()
        assert unknown.parent.exists()
    finally:
        runtime.close()


@pytest.mark.integration
def test_unsupported_repository_format_is_not_read(tmp_path: Path) -> None:
    repository = tmp_path / ".flameox"
    repository.mkdir()
    (repository / "repository.json").write_text(
        json.dumps({"format_version": "999", "created_at": "2026-08-31T00:00:00+00:00"})
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()
        assert failure.value.code == "UNSUPPORTED_REPOSITORY_FORMAT"
    finally:
        runtime.close()


@pytest.mark.integration
def test_preservation_rejects_symlinked_repository_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".flameox").symlink_to(outside, target_is_directory=True)
    artifact = project / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=project / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})

        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])

        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert not (outside / "repository.json").exists()
    finally:
        runtime.close()


@pytest.mark.unit
def test_unpreserved_operations_create_no_durable_state(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()

    assert not (tmp_path / ".flameox").exists()
    assert not (tmp_path / ".diagnostics").exists()
    assert not list(tmp_path.rglob("*.sqlite*"))
    assert not list(tmp_path.rglob("*.duckdb"))


@pytest.mark.integration
def test_preservation_rejects_input_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        artifact.write_text('[{"changed":true}]')
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
    finally:
        runtime.close()


@pytest.mark.integration
def test_corrupt_manifest_and_missing_data_are_not_returned(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        (bundle / "data" / "analysis.json").unlink()

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_symlinked_evidence_data(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        outside = tmp_path / "outside-data"
        (bundle / "data").rename(outside)
        (bundle / "data").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_query_rejects_symlinked_inventory_prefix(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        runtime.repository.initialize()
        outside = tmp_path / "outside-inventory"
        outside.mkdir()
        evidence_root = tmp_path / ".flameox" / "evidence" / "sha256"
        (evidence_root / "aa").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_extra_metadata_and_missing_native_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        repository_metadata = tmp_path / ".flameox" / "repository.json"
        metadata = json.loads(repository_metadata.read_text())
        repository_metadata.write_text(json.dumps({**metadata, "mutable_head": "forbidden"}))

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"

        repository_metadata.write_text(json.dumps(metadata))
        manifest = runtime.read_evidence(evidence_id)
        digest = manifest["body"]["artifacts"][0]["sha256"]
        payload = tmp_path / ".flameox" / "artifacts" / "sha256" / digest[:2] / digest / "payload"
        payload.unlink()

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_self_consistent_manifest_with_invalid_body_shape(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["body"]["inputs"] = ["not-an-input"]
        malformed_id = hashlib.sha256(canonical_bytes(manifest["body"])).hexdigest()
        manifest["evidence_id"] = malformed_id
        malformed_bundle = bundle.parent.parent / malformed_id[:2] / malformed_id
        malformed_bundle.parent.mkdir(exist_ok=True)
        bundle.rename(malformed_bundle)
        (malformed_bundle / "manifest.json").write_bytes(canonical_bytes(manifest))

        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_invalid_nested_analysis_request_before_projection(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["body"]["analysis_request"]["inputs"] = ["not-an-input"]
        malformed_id = hashlib.sha256(canonical_bytes(manifest["body"])).hexdigest()
        manifest["evidence_id"] = malformed_id
        malformed_bundle = bundle.parent.parent / malformed_id[:2] / malformed_id
        malformed_bundle.parent.mkdir()
        bundle.rename(malformed_bundle)
        (malformed_bundle / "manifest.json").write_bytes(canonical_bytes(manifest))

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence_agent_projection(malformed_id)

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repeated_preservation_revalidates_bundle_and_returns_defensive_reference(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        first = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = first["evidence_id"]
        first["evidence_id"] = "0" * 64

        assert runtime.preserve_evidence(result["analysis_id"])["evidence_id"] == evidence_id

        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        (bundle / "data" / "analysis.json").write_text("corrupt")
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_query_pagination_is_deterministic_and_inventory_bound(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        for index in range(3):
            artifact = tmp_path / f"samples-{index}.json"
            artifact.write_text(json.dumps([{"value": index}]))
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            runtime.preserve_evidence(result["analysis_id"])

        first = runtime.query_evidence(limit=2)
        second = runtime.query_evidence(limit=2, cursor=first["continuation"])
        ids = [item["evidence_id"] for item in first["evidence"] + second["evidence"]]
        assert ids == sorted(ids)
        assert len(ids) == 3
        assert second["continuation"] is None

        decoded_cursor = json.loads(
            base64.urlsafe_b64decode(
                first["continuation"] + "=" * (-len(first["continuation"]) % 4)
            )
        )
        decoded_cursor["offset"] = 100
        beyond_inventory = (
            base64.urlsafe_b64encode(canonical_bytes(decoded_cursor)).decode().rstrip("=")
        )
        with pytest.raises(RuntimeFailure) as invalid_offset:
            runtime.query_evidence(limit=2, cursor=beyond_inventory)
        assert invalid_offset.value.code == "INVALID_INPUT"

        decoded_cursor["offset"] = "1"
        non_integer_offset = base64.urlsafe_b64encode(canonical_bytes(decoded_cursor)).decode()
        with pytest.raises(RuntimeFailure) as invalid_type:
            runtime.query_evidence(limit=2, cursor=non_integer_offset)
        assert invalid_type.value.code == "INVALID_INPUT"

        filtered = runtime.query_evidence(limit=1, capability_id="artifact.preview")
        with pytest.raises(RuntimeFailure) as changed_filters:
            runtime.query_evidence(
                limit=1,
                capability_id="cpu.hotspots",
                cursor=filtered["continuation"],
            )
        assert changed_filters.value.code == "INVALID_INPUT"

        with pytest.raises(RuntimeFailure) as malformed_digest:
            runtime.query_evidence(input_sha256="not-a-digest")
        assert malformed_digest.value.code == "INVALID_INPUT"

        earliest = runtime.read_evidence(ids[0])
        input_digest = earliest["body"]["inputs"][0]["sha256"]
        filtered = runtime.query_evidence(input_sha256=input_digest, limit=1)
        assert [item["evidence_id"] for item in filtered["evidence"]] == [ids[0]]
        assert filtered["continuation"] is None
    finally:
        runtime.close()


@pytest.mark.integration
def test_preservation_does_not_mutate_project_git_configuration(tmp_path: Path) -> None:
    git_info = tmp_path / ".git" / "info"
    git_info.mkdir(parents=True)
    exclude = git_info / "exclude"
    exclude.write_text("existing-pattern\n")
    tracked_ignore = tmp_path / ".gitignore"
    tracked_ignore.write_text("tracked-pattern\n")
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        runtime.preserve_evidence(result["analysis_id"])
        runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    assert exclude.read_text().splitlines() == ["existing-pattern"]
    assert tracked_ignore.read_text() == "tracked-pattern\n"
