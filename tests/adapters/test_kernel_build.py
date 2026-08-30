from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.adapters.kernel_build import (
    KernelBuildManifest,
    kernel_build_json_schema,
)
from flameox.application.kernel_builds import KernelBuildImportService
from flameox.application.pipelines import (
    ArtifactPipeline,
    ArtifactPipelineService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import RunStore, Workspace

pytestmark = pytest.mark.unit


def _artifact(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": "application/json" if path.endswith(".json") else "text/plain",
    }


def _manifest(*, native_groups: list[dict[str, object]] | None = None) -> KernelBuildManifest:
    return KernelBuildManifest.model_validate(
        {
            "producer": "triton",
            "native_groups": native_groups
            or [
                {
                    "path": "triton-dumps/group-a",
                    "artifacts": [_artifact("triton-dumps/group-a/kernel.ttir", b"ttir")],
                }
            ],
        }
    )


def _write_groups(root: Path, *, changed_group: str | None = None) -> KernelBuildManifest:
    """Write an external import fixture, including provider metadata evidence."""

    groups: list[dict[str, object]] = []
    for group_name in ("source-hash-a", "source-hash-b"):
        artifacts: list[dict[str, object]] = []
        for extension in (
            "ttir",
            "ttgir",
            "llir",
            "ptx",
            "cubin",
            "sass",
            "json",
            "metadata",
        ):
            path = f"triton-dumps/{group_name}/kernel.{extension}"
            payload = f"{group_name}:{extension}".encode()
            if group_name == changed_group and extension == "ptx":
                payload = b"changed-ptx"
            artifact_path = root / path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(payload)
            artifacts.append(_artifact(path, payload))
        groups.append({"path": f"triton-dumps/{group_name}", "artifacts": artifacts})
    manifest = _manifest(native_groups=groups)
    (root / "kernel-build.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return manifest


@pytest.mark.parametrize(
    ("group_path", "artifact_path"),
    [
        ("../secret", "../secret/kernel.ttir"),
        ("/absolute", "/absolute/kernel.ttir"),
        ("triton-dumps//group", "triton-dumps//group/kernel.ttir"),
        ("triton-dumps/group", "triton-dumps/group/../kernel.ttir"),
        ("triton-dumps/group", "triton-dumps/group//kernel.ttir"),
        ("triton-dumps/group", "../secret/kernel.ttir"),
        ("triton-dumps/group", "triton-dumps/other/kernel.ttir"),
    ],
)
def test_kernel_build_manifest_rejects_paths_outside_the_declared_group(
    group_path: str,
    artifact_path: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=r"contained relative|below their native directory|normalized",
    ):
        _manifest(
            native_groups=[
                {
                    "path": group_path,
                    "artifacts": [_artifact(artifact_path, b"ttir")],
                }
            ]
        )


def test_kernel_build_manifest_is_a_versionless_group_provenance_document() -> None:
    manifest = _manifest()
    payload = manifest.model_dump(mode="json")

    assert "stages" not in payload
    assert set(payload) == {"producer", "native_groups", "attachments"}


def test_kernel_build_generated_schema_uses_the_canonical_unversioned_name() -> None:
    schema_root = Path(__file__).resolve().parents[2] / "src/flameox/schemas"
    canonical = schema_root / "kernel-build.schema.json"

    assert json.loads(canonical.read_text(encoding="utf-8")) == kernel_build_json_schema()


def _pipelines_by_group(
    workspace: Workspace,
    *,
    run_id: str,
) -> dict[str, ArtifactPipeline]:
    registrations = {
        registration.artifact_id: registration.display_name
        for registration in RunStore(workspace).read(run_id).artifacts
    }
    pipelines: dict[str, ArtifactPipeline] = {}
    for pipeline in ArtifactPipelineService(workspace).pipelines.list():
        if pipeline.run_id != run_id:
            continue
        first_artifact_id = pipeline.stages[0].artifact_id
        assert first_artifact_id is not None
        pipelines[registrations[first_artifact_id].rsplit("/", 1)[0]] = pipeline
    return pipelines


def test_kernel_build_import_preserves_two_groups_and_registers_sibling_lineages(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source"
    source.mkdir()
    manifest = _write_groups(source)

    result = KernelBuildImportService(workspace).import_manifest(
        source / "kernel-build.json",
        allow_external_path=True,
    )
    pipelines = _pipelines_by_group(workspace, run_id=result.run.run_id)

    assert len(result.pipeline_ids) == 2
    assert set(pipelines) == {"triton-dumps/source-hash-a", "triton-dumps/source-hash-b"}
    assert [registration.display_name for registration in result.run.artifacts[1:]] == [
        artifact.path for artifact in manifest.artifacts
    ]
    for group_path, pipeline in pipelines.items():
        assert pipeline.identity_quality == "imported_unverified"
        assert {
            "workload_identity",
            "workload_definition_id",
            "environment_id",
            "producer_version",
        }.isdisjoint(pipeline.model_dump())
        assert [stage.name for stage in pipeline.stages] == [
            "ttir",
            "ttgir",
            "llir",
            "ptx",
            "cubin",
            "sass",
        ]
        assert [stage.predecessor for stage in pipeline.stages] == [
            None,
            "ttir",
            "ttgir",
            "llir",
            "ptx",
            "cubin",
        ]
        assert all(
            registration.display_name.startswith(f"{group_path}/")
            for registration in result.run.artifacts
            if registration.artifact_id in {stage.artifact_id for stage in pipeline.stages}
        )
    metadata_ids = {
        registration.artifact_id
        for registration in result.run.artifacts
        if registration.display_name
        in {
            "triton-dumps/source-hash-a/kernel.json",
            "triton-dumps/source-hash-a/kernel.metadata",
        }
    }
    assert len(metadata_ids) == 2
    assert all(
        registration.role == "compiler_output"
        for registration in result.run.artifacts
        if registration.artifact_id in metadata_ids
    )
    assert all(
        metadata_ids.isdisjoint({stage.artifact_id for stage in pipeline.stages})
        for pipeline in pipelines.values()
    )


def test_kernel_build_import_preserves_a_metadata_only_group_without_a_stage(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source"
    source.mkdir()
    artifacts = [
        _artifact("triton-dumps/source-hash-metadata/kernel.json", b"{}"),
        _artifact("triton-dumps/source-hash-metadata/kernel.metadata", b"metadata"),
    ]
    for artifact in artifacts:
        path = source / str(artifact["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"metadata")
    manifest = _manifest(
        native_groups=[
            {
                "path": "triton-dumps/source-hash-metadata",
                "artifacts": artifacts,
            }
        ]
    )
    (source / "kernel-build.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    result = KernelBuildImportService(workspace).import_manifest(
        source / "kernel-build.json",
        allow_external_path=True,
    )
    pipelines = [
        pipeline
        for pipeline in ArtifactPipelineService(workspace).pipelines.list()
        if pipeline.run_id == result.run.run_id
    ]

    assert len(result.pipeline_ids) == 1
    assert pipelines[0].stages == ()
    assert all(
        registration.role == "compiler_output"
        for registration in result.run.artifacts
        if registration.display_name in {artifact["path"] for artifact in artifacts}
    )


def test_kernel_build_comparison_selects_one_sibling_pipeline_at_a_time(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    _write_groups(baseline_root)
    _write_groups(candidate_root, changed_group="source-hash-b")

    importer = KernelBuildImportService(workspace)
    baseline = importer.import_manifest(
        baseline_root / "kernel-build.json",
        allow_external_path=True,
    )
    candidate = importer.import_manifest(
        candidate_root / "kernel-build.json",
        allow_external_path=True,
    )
    baseline_pipelines = _pipelines_by_group(workspace, run_id=baseline.run.run_id)
    candidate_pipelines = _pipelines_by_group(workspace, run_id=candidate.run.run_id)

    comparison = ArtifactPipelineService(workspace).compare(
        baseline_pipelines["triton-dumps/source-hash-a"].pipeline_id,
        candidate_pipelines["triton-dumps/source-hash-a"].pipeline_id,
    )

    group_b_artifact_ids = {
        stage.artifact_id
        for pipeline in (
            baseline_pipelines["triton-dumps/source-hash-b"],
            candidate_pipelines["triton-dumps/source-hash-b"],
        )
        for stage in pipeline.stages
        if stage.artifact_id is not None
    }
    assert comparison.compatibility == "unknown"
    assert {stage.stage_name for stage in comparison.stages} == {
        "ttir",
        "ttgir",
        "llir",
        "ptx",
        "cubin",
        "sass",
    }
    assert not group_b_artifact_ids.intersection(comparison.input_artifact_ids)


def test_kernel_build_import_rejects_symlinked_manifest_before_parsing(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-external.json"
    external.write_text(_manifest().model_dump_json())
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.symlink_to(external)
    try:
        with pytest.raises(DomainError) as error:
            KernelBuildImportService(workspace).import_manifest(manifest_path)
        assert error.value.code is ErrorCode.EXECUTION_REFUSED
    finally:
        external.unlink()


def test_malformed_kernel_build_manifest_leaves_no_anonymous_state(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.write_text('{"unique-invalid-manifest": true}')

    with pytest.raises(DomainError) as error:
        KernelBuildImportService(workspace).import_manifest(manifest_path, allow_external_path=True)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert tuple(workspace.paths.artifacts.rglob("artifact.json")) == ()
    assert RunStore(workspace).list() == ()
    assert tuple(workspace.paths.staging.iterdir()) == ()


def test_kernel_build_import_applies_manifest_document_limit(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "kernel-build.json"
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    with manifest_path.open("ab") as stream:
        stream.truncate(1024 * 1024 + 1)

    with pytest.raises(DomainError) as error:
        KernelBuildImportService(workspace).import_manifest(manifest_path, allow_external_path=True)

    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE


def test_kernel_build_bundle_rejects_digest_mismatch_before_registration(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "triton-dumps/group-a/kernel.ttir"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("actualxx", encoding="utf-8")
    manifest = _manifest(
        native_groups=[
            {
                "path": "triton-dumps/group-a",
                "artifacts": [_artifact("triton-dumps/group-a/kernel.ttir", b"expected")],
            }
        ]
    )
    manifest_path = source / "kernel-build.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(DomainError, match="sha256 mismatch"):
        KernelBuildImportService(workspace).import_manifest(manifest_path, allow_external_path=True)

    assert tuple(workspace.paths.artifacts.rglob("artifact.json")) == ()


def test_kernel_build_import_rejects_unsupported_group_artifacts_without_importing(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "triton-dumps/group-a/kernel.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("unsupported", encoding="utf-8")
    manifest = _manifest(
        native_groups=[
            {
                "path": "triton-dumps/group-a",
                "artifacts": [_artifact("triton-dumps/group-a/kernel.txt", b"unsupported")],
            }
        ]
    )
    manifest_path = source / "kernel-build.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(DomainError, match="unsupported compiler artifact"):
        KernelBuildImportService(workspace).import_manifest(manifest_path, allow_external_path=True)

    assert RunStore(workspace).list() == ()
