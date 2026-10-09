from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from flameox.canonical import canonical_bytes
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


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


def test_interrupted_evidence_publication_never_exposes_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    original = os.rename

    def interrupt_evidence(stage: Path, destination: Path) -> None:
        if "evidence" in Path(os.fsdecode(destination)).parts:
            raise OSError(errno.EIO, "injected evidence publication interruption")
        original(stage, destination)

    monkeypatch.setattr(os, "rename", interrupt_evidence)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_IO_FAILURE"
        assert runtime.query_evidence()["evidence"] == []
        assert not list((tmp_path / ".flameox" / "evidence").rglob("manifest.json"))
    finally:
        runtime.close()


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
