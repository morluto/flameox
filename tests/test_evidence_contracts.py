from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.canonical import canonical_bytes
from flameox.repository import EvidenceRepository, RepositoryError
from flameox.runtime_contracts import (
    CaptureTarget,
    EvidenceSource,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)
from flameox.source_files import bundle_digest
from flameox.stateless import AnalysisRuntime


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


@pytest.mark.integration
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
        assert result["analysis_id"] in runtime.analyses
        assert (configured / "unexpected").read_text() == "corrupt"
        rescued["next_action"]["environment"]["FLAMEOX_DATA_DIR"] = "mutated"
        runtime.rescues.clear()
        rescued = runtime.rescue_evidence(result["analysis_id"], str(rescue))
        assert rescued["next_action"]["environment"] == {"FLAMEOX_DATA_DIR": str(rescue)}
        runtime.analyses.pop(result["analysis_id"])
        assert runtime.rescue_evidence(result["analysis_id"], str(rescue)) == rescued
    finally:
        runtime.close()

    reopened = AnalysisRuntime(evidence_directory=rescue)
    try:
        manifest = reopened.read_evidence(rescued["evidence_id"])
        assert manifest["body"]["capability_id"] == "artifact.preview"
    finally:
        reopened.close()


@pytest.mark.unit
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
            assert result["analysis_id"] in runtime.analyses
    finally:
        runtime.close()


@pytest.mark.unit
def test_rescue_repository_failures_identify_the_alternate_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "input.json"
    artifact.write_text("[]")
    configured = tmp_path / "store"
    rescue = tmp_path / "rescue"
    runtime = AnalysisRuntime(evidence_directory=configured)
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})

    def fail(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RepositoryError("REPOSITORY_CORRUPTION", "alternate is corrupt")

    monkeypatch.setattr(EvidenceRepository, "preserve", fail)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.rescue_evidence(result["analysis_id"], str(rescue))
        assert failure.value.details["configuration_source"] == "rescue_destination"
        assert "local_diagnostic" not in failure.value.details
        assert (
            failure.value.details["store_identifier"]
            == hashlib.sha256(str(rescue.absolute()).encode()).hexdigest()
        )
    finally:
        runtime.close()


@pytest.mark.unit
def test_rescue_publication_stays_anchored_if_parent_path_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "input.json"
    artifact.write_text("[]")
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    replacement_destination = parent / "rescue"
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "configured")
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    original_preserve = EvidenceRepository.preserve

    def replace_parent(repository: EvidenceRepository, **kwargs: Any) -> dict[str, Any]:
        parent.rename(moved_parent)
        parent.symlink_to(moved_parent, target_is_directory=True)
        return original_preserve(repository, **kwargs)

    monkeypatch.setattr(EvidenceRepository, "preserve", replace_parent)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.rescue_evidence(result["analysis_id"], str(replacement_destination))
        assert failure.value.code == "REPOSITORY_IO_FAILURE"
        assert parent.is_symlink()
        assert (moved_parent / "rescue" / "repository.json").is_file()
    finally:
        runtime.close()


@pytest.mark.unit
def test_rescue_preflight_requires_a_new_destination(tmp_path: Path) -> None:
    rescue = tmp_path / "rescue"
    rescue.mkdir()
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "configured")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preflight_rescue_destination(str(rescue))
        assert failure.value.code == "INVALID_INPUT"
        assert "new path" in failure.value.message
    finally:
        runtime.close()


@pytest.mark.unit
def test_rescue_rejects_a_physical_alias_of_the_configured_store(tmp_path: Path) -> None:
    physical_store = tmp_path / "physical-store"
    physical_store.mkdir()
    configured_alias = tmp_path / "configured-alias"
    configured_alias.symlink_to(physical_store, target_is_directory=True)
    runtime = AnalysisRuntime(evidence_directory=configured_alias)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preflight_rescue_destination(str(physical_store / "rescue"))
        assert failure.value.code == "INVALID_INPUT"
        assert not (physical_store / "rescue").exists()
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize("bound", ["input", "scratch"])
def test_evidence_limits_reject_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    projection = preserve_bundle(tmp_path)
    if bound == "scratch":
        monkeypatch.setattr("flameox.stateless.MAX_SESSION_SCRATCH_BYTES", 1024)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "artifact.preview",
                [EvidenceSource.model_validate(projection["analysis_sources"][0])],
                {},
                limits=RequestLimits(max_input_bytes=1024) if bound == "input" else None,
            )
        assert failure.value.code == "LIMIT_EXCEEDED"
        assert not [path for path in runtime.scratch.rglob("*") if path.is_file()]
    finally:
        runtime.close()


@pytest.mark.integration
def test_materialized_evidence_participates_in_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = preserve_bundle(tmp_path, "one")
    second = preserve_bundle(tmp_path, "two")
    monkeypatch.setattr("flameox.stateless.MAX_SESSION_SCRATCH_BYTES", 2048)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        old = runtime.analyze(
            "artifact.preview", [EvidenceSource.model_validate(first["analysis_sources"][0])], {}
        )
        runtime.analyze(
            "artifact.preview", [EvidenceSource.model_validate(second["analysis_sources"][0])], {}
        )
        assert (
            sum(path.stat().st_size for path in runtime.scratch.rglob("*") if path.is_file())
            <= 2048
        )
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(old["analysis_id"])
        assert failure.value.code == "EXPIRED_SESSION_ANALYSIS"
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize("bound", ["input_bytes", "input_files", "scratch_bytes", "scratch_files"])
def test_multisource_admission_is_aggregate_and_allocates_nothing_on_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    projections = [preserve_bundle(tmp_path, name) for name in ("one", "two")]
    limits = RequestLimits()
    if bound == "input_bytes":
        limits = RequestLimits(max_input_bytes=3000)
    elif bound == "input_files":
        limits = RequestLimits(max_input_files=1)
    elif bound == "scratch_bytes":
        monkeypatch.setattr("flameox.stateless.MAX_SESSION_SCRATCH_BYTES", 3000)
    else:
        monkeypatch.setattr("flameox.stateless.MAX_SESSION_SCRATCH_FILES", 1)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "artifact.preview",
                [
                    EvidenceSource.model_validate(item["analysis_sources"][0])
                    for item in projections
                ],
                {},
                limits=limits,
            )
        assert failure.value.code == "LIMIT_EXCEEDED"
        assert not [path for path in runtime.scratch.rglob("*") if path.is_file()]
    finally:
        runtime.close()


@pytest.mark.integration
def test_cache_handle_eviction_keeps_shared_active_evidence_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projection = preserve_bundle(tmp_path)
    monkeypatch.setattr("flameox.stateless.MAX_SESSION_ANALYSES", 1)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    source = EvidenceSource.model_validate(projection["analysis_sources"][0])
    try:
        first = runtime.analyze("artifact.preview", [source], {})
        second = runtime.analyze("artifact.preview", [source], {}, limits=RequestLimits(max_rows=2))
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(first["analysis_id"])
        assert failure.value.code == "EXPIRED_SESSION_ANALYSIS"
        ref = runtime.preserve_evidence(second["analysis_id"])
        assert runtime.read_evidence_agent_projection(ref["evidence_id"])["analysis_sources"]
    finally:
        runtime.close()


@pytest.mark.integration
def test_preserving_child_bundle_keeps_cached_ancestor_input(tmp_path: Path) -> None:
    projection = preserve_bundle(tmp_path)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        child = runtime.analyze(
            "artifact.preview",
            [EvidenceSource.model_validate(projection["analysis_sources"][0])],
            {},
        )
        root = Path(child["inputs"][0]["path"])
        ancestor = runtime.analyze("artifact.preview", [PathSource(path=str(root.parent))], {})
        runtime.preserve_evidence(child["analysis_id"])
        assert (root / "foo").read_text() == "x" * 2048
        assert runtime.preserve_evidence(ancestor["analysis_id"])["evidence_id"]
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
def test_acquiring_second_input_cannot_evict_the_first_active_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = preserve_bundle(tmp_path, "one")
    second = preserve_bundle(tmp_path, "two")
    monkeypatch.setattr("flameox.stateless.MAX_SESSION_SCRATCH_BYTES", 4096)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        source = EvidenceSource.model_validate(first["analysis_sources"][0])
        runtime.analyze("artifact.preview", [source], {})
        result = runtime.analyze(
            "artifact.preview",
            [source, EvidenceSource.model_validate(second["analysis_sources"][0])],
            {},
        )
        assert len(result["blocks"][1]["rows"]) == 2
        assert runtime.preserve_evidence(result["analysis_id"])["artifact_count"] == 2
    finally:
        runtime.close()


@pytest.mark.integration
def test_active_member_protects_its_containing_materialized_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = preserve_bundle(tmp_path, "one")
    second = preserve_bundle(tmp_path, "two")
    monkeypatch.setattr("flameox.stateless.MAX_SESSION_SCRATCH_BYTES", 2048)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "artifact.preview", [EvidenceSource.model_validate(first["analysis_sources"][0])], {}
        )
        member = Path(result["inputs"][0]["path"]) / "foo"
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "artifact.preview",
                [
                    PathSource(path=str(member)),
                    EvidenceSource.model_validate(second["analysis_sources"][0]),
                ],
                {},
            )
        assert failure.value.code == "LIMIT_EXCEEDED"
        assert member.read_text() == "x" * 2048
    finally:
        runtime.close()


@pytest.mark.integration
def test_reused_materialization_is_verified_before_analysis_cache_lookup(tmp_path: Path) -> None:
    projection = preserve_bundle(tmp_path)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    source = EvidenceSource.model_validate(projection["analysis_sources"][0])
    try:
        result = runtime.analyze("artifact.preview", [source], {})
        (Path(result["inputs"][0]["path"]) / "foo").write_text("changed")
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("artifact.preview", [source], {})
        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert "recovery" in failure.value.details
    finally:
        runtime.close()


@pytest.mark.integration
def test_failed_analysis_releases_new_empty_materializations(tmp_path: Path) -> None:
    publisher = AnalysisRuntime(evidence_directory=tmp_path / "store")
    sources = []
    try:
        for index in range(3):
            empty = tmp_path / str(index)
            empty.mkdir()
            result = publisher.analyze("artifact.preview", [PathSource(path=str(empty))], {})
            ref = publisher.preserve_evidence(result["analysis_id"])
            projection = publisher.read_evidence_agent_projection(ref["evidence_id"])
            sources.append(EvidenceSource.model_validate(projection["analysis_sources"][0]))
    finally:
        publisher.close()
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        for source in sources:
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze("cpu.hotspots", [source], {})
            assert failure.value.code == "UNSUPPORTED_FORMAT"
            assert not list((runtime.scratch / "evidence-sources").iterdir())
    finally:
        runtime.close()


@pytest.mark.integration
def test_bundle_and_member_compose_through_repeated_preservation(tmp_path: Path) -> None:
    projection = preserve_bundle(tmp_path)
    sources = [
        EvidenceSource.model_validate(projection["analysis_sources"][0]),
        EvidenceSource.model_validate(projection["body"]["artifacts"][0]["source"]),
    ]
    for _ in range(2):
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = runtime.analyze("artifact.preview", sources, {})
            ref = runtime.preserve_evidence(result["analysis_id"])
            projection = runtime.read_evidence_agent_projection(ref["evidence_id"])
            assert len(projection["analysis_sources"]) == 2
            sources = [
                EvidenceSource.model_validate(item) for item in projection["analysis_sources"]
            ]
        finally:
            runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize("target", ["repository", "manifest", "artifact"])
@pytest.mark.parametrize("payload", [[], None, "wrong shape", 1])
def test_nonobject_repository_json_has_typed_recovery(
    tmp_path: Path, target: str, payload: Any
) -> None:
    projection = preserve_bundle(tmp_path)
    evidence_id = projection["evidence_id"]
    store = tmp_path / "store"
    if target == "repository":
        path = store / "repository.json"
    elif target == "manifest":
        path = store / "evidence" / "sha256" / evidence_id[:2] / evidence_id / "manifest.json"
    else:
        digest = projection["body"]["artifacts"][0]["sha256"]
        path = store / "artifacts" / "sha256" / digest[:2] / digest / "artifact.json"
    path.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence_agent_projection(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert "recovery" in failure.value.details
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize("target", ["repository", "manifest", "artifact"])
def test_format_one_is_rejected_without_mutating_old_documents(tmp_path: Path, target: str) -> None:
    projection = preserve_bundle(tmp_path)
    evidence_id = projection["evidence_id"]
    store = tmp_path / "store"
    if target == "repository":
        path = store / "repository.json"
    elif target == "manifest":
        path = store / "evidence" / "sha256" / evidence_id[:2] / evidence_id / "manifest.json"
    else:
        digest = projection["body"]["artifacts"][0]["sha256"]
        path = store / "artifacts" / "sha256" / digest[:2] / digest / "artifact.json"
    document = json.loads(path.read_text())
    document["format_version"] = "1"
    original = json.dumps(document).encode()
    path.write_bytes(original)
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence_agent_projection(evidence_id)
        assert failure.value.code == "UNSUPPORTED_REPOSITORY_FORMAT"
        assert "recovery" in failure.value.details
        assert path.read_bytes() == original
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize("missing", ["source_layout", "relative_paths", "analysis_mapping"])
def test_current_format_requires_explicit_source_membership(tmp_path: Path, missing: str) -> None:
    projection = preserve_bundle(tmp_path)
    evidence_id = projection["evidence_id"]
    store = tmp_path / "store"
    bundle = store / "evidence" / "sha256" / evidence_id[:2] / evidence_id
    manifest = json.loads((bundle / "manifest.json").read_text())
    if missing == "source_layout":
        del manifest["body"]["source_layout"]
    elif missing == "relative_paths":
        del manifest["body"]["source_layout"]["sources"][0]["relative_paths"]
    else:
        manifest["body"]["source_layout"]["analysis_sources"] = []
    changed_id = hashlib.sha256(canonical_bytes(manifest["body"])).hexdigest()
    manifest["evidence_id"] = changed_id
    destination = bundle.parent.parent / changed_id[:2] / changed_id
    destination.parent.mkdir(exist_ok=True)
    bundle.rename(destination)
    (destination / "manifest.json").write_bytes(canonical_bytes(manifest))
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
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


@pytest.mark.integration
@pytest.mark.parametrize("value", [42, "hello", True, None])
def test_json_scalar_root_is_not_empty_evidence(tmp_path: Path, value: Any) -> None:
    artifact = tmp_path / "scalar.json"
    artifact.write_text(json.dumps(value))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        assert result["blocks"][1]["rows"][0]["value"] == value
        assert result["coverage"]["complete"] is True
    finally:
        runtime.close()


@pytest.mark.integration
def test_json_projection_keeps_literal_keys_separate_from_nested_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "keys.json"
    artifact.write_text('{"a.b":[1],"a":{"b":[2]}}')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        rows = result["blocks"][1]["rows"]
        assert [row["value"] for row in rows if row.get("section") == "a.b"] == [1]
        assert any(row.get("key") == "a" and row.get("value_type") == "object" for row in rows)
    finally:
        runtime.close()


@pytest.mark.process
def test_failed_analyses_obey_session_cache_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.stateless.MAX_SESSION_ANALYSES", 2)

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        handles = []
        try:
            for _ in range(3):
                result = await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[
                            sys.executable,
                            "-c",
                            "import os,pathlib; "
                            "pathlib.Path(os.environ['FLAMEOX_BENCHMARK_OUTPUT']).write_text('{')",
                        ],
                        cwd=str(tmp_path),
                        provider_id="benchmark-samples",
                    ),
                    "benchmark.summary",
                    preserve=True,
                )
                handles.append(result["analysis_id"])
                assert result["analysis_failure"]["code"] == "DECODE_FAILURE"
            with pytest.raises(RuntimeFailure) as failure:
                runtime.preserve_evidence(handles[0])
            assert failure.value.code == "EXPIRED_SESSION_ANALYSIS"
            assert runtime.preserve_evidence(handles[-1])["evidence_id"]
        finally:
            runtime.close()

    anyio.run(exercise)
