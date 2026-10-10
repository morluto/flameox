from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
        analysis = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(analysis["analysis_id"])
        (tmp_path / ".flameox" / "repository.json").unlink()

        with pytest.raises(RuntimeFailure) as query_failure:
            runtime.query_evidence()
        assert query_failure.value.code == "REPOSITORY_CORRUPTION"
        with pytest.raises(RuntimeFailure) as read_failure:
            runtime.read_evidence(preserved["evidence_id"])
        assert read_failure.value.code == "REPOSITORY_CORRUPTION"
        second = runtime.analyze(
            "preview_artifact",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
        )
        with pytest.raises(RuntimeFailure) as preserve_failure:
            runtime.preserve_evidence(second["analysis_id"])
        assert preserve_failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.parametrize("directory_first", [False, True])
def test_source_kind_changed_after_decoding_cannot_return_successful_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_first: bool
) -> None:
    source = tmp_path / "native"
    if directory_first:
        source.mkdir()
    else:
        source.write_bytes(b"")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    original = runtime._iter_rows

    def replaced(path: Path, format_name: str) -> Iterator[dict[str, Any]]:
        yield from original(path, format_name)
        if directory_first:
            path.rmdir()
            path.write_bytes(b"")
        else:
            path.unlink()
            path.mkdir()

    monkeypatch.setattr(runtime, "_iter_rows", replaced)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("preview_artifact", [PathSource(path=str(source), format="text")], {})
        assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
    finally:
        runtime.close()


def test_interrupted_evidence_publication_never_exposes_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
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
        runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
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
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
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
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        (bundle / "data" / "analysis.json").unlink()

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.parametrize("filename", ["repository.json", "manifest.json", "artifact.json"])
def test_repository_rejects_recursively_nested_metadata(tmp_path: Path, filename: str) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    store = tmp_path / "store"
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        metadata = next(store.rglob(filename))
        native = b"[" * 10_000 + b"0" + b"]" * 10_000
        metadata.write_bytes(native)
        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(preserved["evidence_id"])
        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert metadata.read_bytes() == native
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "member", ["root", "prefix", "bundle", "manifest", "data", "artifact", "payload"]
)
def test_repository_rejects_symlinked_evidence_data(tmp_path: Path, member: str) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        store = tmp_path / ".flameox"
        artifact_bundle = next((store / "artifacts" / "sha256").glob("*/*"))
        selected = {
            "root": store,
            "prefix": bundle.parent,
            "bundle": bundle,
            "manifest": bundle / "manifest.json",
            "data": bundle / "data",
            "artifact": artifact_bundle,
            "payload": artifact_bundle / "payload",
        }[member]
        outside = tmp_path / "outside-data"
        is_directory = selected.is_dir()
        selected.rename(outside)
        selected.symlink_to(outside, target_is_directory=is_directory)

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.parametrize("nested", [False, True])
def test_first_preservation_creates_missing_data_directory_ancestors(
    tmp_path: Path, nested: bool
) -> None:
    store = tmp_path / "missing" / "parent" / "store" if nested else tmp_path / "store"
    artifact = tmp_path / "native.txt"
    artifact.write_text("native evidence")
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        assert not store.exists()
        preserved = runtime.preserve_evidence(result["analysis_id"])
        assert (
            runtime.read_evidence(preserved["evidence_id"])["body"]["evidence_kind"] == "analysis"
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "relative", ["artifacts", "artifacts/sha256", "evidence", "evidence/sha256", ".staging"]
)
def test_initialization_rejects_symlinked_layout_before_writing(
    tmp_path: Path, relative: str
) -> None:
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    selected = store / relative
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.symlink_to(outside, target_is_directory=True)
    artifact = tmp_path / "native.txt"
    artifact.write_text("native")
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert list(outside.iterdir()) == []
        assert not (store / "repository.json").exists()
    finally:
        runtime.close()


@pytest.mark.parametrize("owner", ["999999999999999999999-owner", "²-owner"])
def test_cleanup_retains_staging_with_unverifiable_process_identity(
    tmp_path: Path, owner: str
) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        runtime.repository.initialize()
        path = runtime.repository.root / ".staging" / owner
        path.mkdir()
        (path / "keep").write_text("unverifiable owner")
        runtime.repository.cleanup_abandoned_staging()
        assert (path / "keep").read_text() == "unverifiable owner"
    finally:
        runtime.close()


def test_repository_rejects_self_consistent_manifest_with_invalid_body_shape(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
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
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
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
        assert runtime.query_evidence()["inventory_status"] == "absent"
        with pytest.raises(RuntimeFailure) as absent_inventory:
            runtime.query_evidence(cursor="invalid-cursor")
        assert absent_inventory.value.code == "INVALID_INPUT"
        assert not (tmp_path / ".flameox").exists()

        for index in range(3):
            artifact = tmp_path / f"samples-{index}.json"
            artifact.write_text(json.dumps([{"value": index}]))
            result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
            runtime.preserve_evidence(result["analysis_id"])

        first = runtime.query_evidence(limit=2)
        second = runtime.query_evidence(limit=2, cursor=first["continuation"])
        ids = [item["evidence_id"] for item in first["evidence"] + second["evidence"]]
        assert ids == sorted(ids)
        assert len(ids) == 3
        assert second["continuation"] is None

        recursive_cursor = base64.urlsafe_b64encode(b"[" * 10_000 + b"0" + b"]" * 10_000).decode()
        with pytest.raises(RuntimeFailure) as invalid_cursor:
            runtime.query_evidence(cursor=recursive_cursor)
        assert invalid_cursor.value.code == "INVALID_INPUT"

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

        filtered = runtime.query_evidence(limit=1, operation="preview_artifact")
        with pytest.raises(RuntimeFailure) as changed_filters:
            runtime.query_evidence(
                limit=1,
                operation="rank_cpu_hotspots",
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
        result = runtime.analyze("preview_artifact", [PathSource(path=str(artifact))], {})
        runtime.preserve_evidence(result["analysis_id"])
        runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    assert exclude.read_text().splitlines() == ["existing-pattern"]
    assert tracked_ignore.read_text() == "tracked-pattern\n"


@pytest.mark.parametrize("directory_first", [False, True])
def test_empty_source_kind_is_retained_through_analysis_and_preservation(
    tmp_path: Path, directory_first: bool
) -> None:
    source = tmp_path / "native"
    if directory_first:
        source.mkdir()
    else:
        source.write_bytes(b"")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        inputs = [PathSource(path=str(source), format="text")]
        first = runtime.analyze("preview_artifact", inputs, {})
        assert first["coverage"]["complete"] is True
        preserved = runtime.preserve_evidence(first["analysis_id"])
        manifest = runtime.read_evidence(preserved["evidence_id"])
        assert manifest["body"]["source_layout"]["sources"][0]["is_directory"] is directory_first
        pending = runtime.analyze("preview_artifact", inputs, {}, limits=RequestLimits(max_rows=1))
        if directory_first:
            source.rmdir()
            source.write_bytes(b"")
        else:
            source.unlink()
            source.mkdir()
        # Reusing this path and digest must not reuse the first source's kind.
        second = runtime.analyze("preview_artifact", inputs, {})
        assert second["analysis_id"] != first["analysis_id"]
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(pending["analysis_id"])
        assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
        assert (
            runtime.preserve_evidence(second["analysis_id"])["evidence_id"]
            != preserved["evidence_id"]
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("document", ["repository", "artifact", "manifest"])
def test_obsolete_repository_formats_are_rejected_without_migration(
    tmp_path: Path, document: str
) -> None:
    source = tmp_path / "native.txt"
    source.write_text("preserved observation\n")
    store = tmp_path / "store"
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        analysis = runtime.analyze("preview_artifact", [PathSource(path=str(source))], {})
        reference = runtime.preserve_evidence(analysis["analysis_id"])
        manifest = runtime.read_evidence(reference["evidence_id"])
        assert manifest["format_version"] == "4"
        if document == "repository":
            path = store / "repository.json"
        elif document == "artifact":
            path = next((store / "artifacts").rglob("artifact.json"))
        else:
            path = next((store / "evidence").rglob("manifest.json"))
        payload = json.loads(path.read_bytes())
        payload["format_version"] = "3"
        path.write_bytes(canonical_bytes(payload))
        original = path.read_bytes()
        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(reference["evidence_id"])
        assert failure.value.code == "UNSUPPORTED_REPOSITORY_FORMAT"
        assert path.read_bytes() == original
    finally:
        runtime.close()
