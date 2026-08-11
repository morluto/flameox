from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.adapters import KernelBuildManifestV1, kernel_build_json_schema
from flameox.application import (
    ArtifactPipelineService,
    KernelBuildImportService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


def _manifest(**updates: object) -> KernelBuildManifestV1:
    payload: dict[str, object] = {
        "producer": "triton",
        "producer_version": "3.7.1",
        "workload_identity": "vector-add",
        "device_identity": "NVIDIA sm_86",
        "outcome": "succeeded",
        "cache_status": "miss",
        "source_environment": {
            "TRITON_DUMP_DIR": "compiler-dumps",
            "TRITON_KERNEL_DUMP": "1",
        },
        "stages": [
            {
                "name": "ttir",
                "ordinal": 0,
                "status": "available",
                "format": "mlir",
                "format_schema": "triton-ttir",
                "artifact": {
                    "path": "kernel/vector_add.ttir",
                    "byte_length": 4,
                    "sha256": "a" * 64,
                    "media_type": "text/plain",
                },
            },
            {
                "name": "ptx",
                "ordinal": 1,
                "predecessor": "ttir",
                "status": "cached",
                "format": "ptx",
                "format_schema": "nvidia-ptx",
                "artifact": {
                    "path": "kernel/vector_add.ptx",
                    "byte_length": 4,
                    "sha256": "b" * 64,
                    "media_type": "text/plain",
                },
            },
        ],
    }
    payload.update(updates)
    return KernelBuildManifestV1.model_validate(payload)


@pytest.mark.parametrize("path", ["../secret", "/absolute", "kernel/../secret", "."])
def test_kernel_build_rejects_uncontained_artifact_paths(path: str) -> None:
    stages = _manifest().model_dump(mode="json")["stages"]
    stages[0]["artifact"]["path"] = path

    with pytest.raises(ValidationError, match=r"contained relative|normalized"):
        _manifest(stages=stages)


def test_kernel_build_rejects_duplicate_artifacts_and_out_of_order_stages() -> None:
    duplicate = _manifest().model_dump(mode="json")["stages"]
    duplicate[1]["artifact"]["path"] = duplicate[0]["artifact"]["path"]
    with pytest.raises(ValidationError, match="artifact paths must be unique"):
        _manifest(stages=duplicate)

    out_of_order = _manifest().model_dump(mode="json")["stages"]
    out_of_order[0]["ordinal"], out_of_order[1]["ordinal"] = 1, 0
    with pytest.raises(ValidationError, match="declared in ordinal order"):
        _manifest(stages=out_of_order)


def test_failed_build_requires_failed_stage_and_success_rejects_missing_output() -> None:
    stages = _manifest().model_dump(mode="json")["stages"]
    stages[1]["status"] = "unavailable"
    stages[1]["artifact"] = None

    with pytest.raises(ValidationError, match="successful build"):
        _manifest(stages=stages)
    with pytest.raises(ValidationError, match="requires a failed stage"):
        _manifest(outcome="failed")


def test_kernel_build_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _manifest(unverified_hint=True)


def test_kernel_build_generated_schema_does_not_drift() -> None:
    schema_path = (
        Path(__file__).resolve().parents[2] / "src/flameox/schemas/kernel-build-v1.schema.json"
    )

    assert json.loads(schema_path.read_text(encoding="utf-8")) == kernel_build_json_schema()


def test_kernel_build_bundle_import_registers_existing_pipeline(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact_path = tmp_path / "kernel.ttir"
    artifact_path.write_text("ttir", encoding="utf-8")
    (tmp_path / "unrelated.ptx").write_text("must not be imported")
    manifest = _manifest(
        diagnostics=["diagnostic text remains native manifest data"],
        limitations=[f"limitation-{index}" for index in range(20)],
        stages=[
            {
                "name": "ttir",
                "ordinal": 0,
                "status": "available",
                "format": "mlir",
                "format_schema": "triton-ttir",
                "artifact": {
                    "path": artifact_path.name,
                    "byte_length": 4,
                    "sha256": hashlib.sha256(b"ttir").hexdigest(),
                    "media_type": "text/plain",
                },
            }
        ],
    )
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = KernelBuildImportService(workspace).import_manifest(manifest_path)

    assert len(result.run.artifacts) == 2
    assert result.run.artifacts[0].role == "kernel_build_manifest"
    assert result.run.artifacts[1].role == "compiler_stage"
    assert result.pipeline.workload_identity == "vector-add"
    assert result.pipeline.device_identity == "NVIDIA sm_86"
    assert result.pipeline.run_id == result.run.run_id
    assert result.pipeline.stages[0].artifact_id == result.run.artifacts[1].artifact_id
    assert len(result.pipeline.limitations) == 20
    assert "diagnostics are preserved" in result.pipeline.limitations[0]
    assert result.pipeline.limitations[-1].startswith("Additional limitations")


def test_imported_pipeline_comparison_distinguishes_mismatch_from_missing_version(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    pipeline_ids: dict[str, str] = {}
    for directory, workload, device in (
        ("baseline", "vector-add", "sm_86"),
        ("same-identity", "vector-add", "sm_86"),
        ("different-identity", "matrix-multiply", "sm_90"),
    ):
        root = tmp_path / directory
        root.mkdir()
        artifact = root / "kernel.ttir"
        artifact.write_text("ttir")
        manifest = _manifest(
            producer_version="unknown",
            workload_identity=workload,
            device_identity=device,
            stages=[
                {
                    "name": "ttir",
                    "ordinal": 0,
                    "status": "available",
                    "format": "mlir",
                    "format_schema": "triton-ttir",
                    "artifact": {
                        "path": artifact.name,
                        "byte_length": 4,
                        "sha256": hashlib.sha256(b"ttir").hexdigest(),
                    },
                }
            ],
        )
        manifest_path = root / "kernel-build.json"
        manifest_path.write_text(manifest.model_dump_json())
        pipeline_ids[directory] = (
            KernelBuildImportService(workspace).import_manifest(manifest_path).pipeline.pipeline_id
        )

    unknown = ArtifactPipelineService(workspace).compare(
        pipeline_ids["baseline"], pipeline_ids["same-identity"]
    )
    incompatible = ArtifactPipelineService(workspace).compare(
        pipeline_ids["baseline"], pipeline_ids["different-identity"]
    )

    assert unknown.compatibility == "unknown"
    assert any("producer_version" in item for item in unknown.limitations)
    assert incompatible.compatibility == "incompatible"
    assert incompatible.identity_mismatches == ("workload_identity", "device_identity")


def test_kernel_build_import_uses_bounded_declared_role(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact = tmp_path / "kernel.ttir"
    artifact.write_text("ttir")
    stage = _manifest().stages[0].model_dump(mode="json")
    stage["name"] = "x" * 100
    stage["artifact"] = {
        "path": artifact.name,
        "byte_length": 4,
        "sha256": hashlib.sha256(b"ttir").hexdigest(),
        "role": "compiler_stage",
    }
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.write_text(_manifest(stages=[stage]).model_dump_json())

    result = KernelBuildImportService(workspace).import_manifest(manifest_path)

    assert result.run.artifacts[1].role == "compiler_stage"


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
        KernelBuildImportService(workspace).import_manifest(manifest_path)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert tuple(workspace.paths.artifacts.rglob("artifact.json")) == ()
    assert tuple(workspace.paths.runs.iterdir()) == ()
    assert tuple(workspace.paths.staging.iterdir()) == ()


def test_kernel_build_import_applies_manifest_document_limit(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.write_text(_manifest().model_dump_json())
    with manifest_path.open("ab") as manifest_file:
        manifest_file.truncate(1024 * 1024 + 1)

    with pytest.raises(DomainError) as error:
        KernelBuildImportService(workspace).import_manifest(manifest_path)
    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE


def test_kernel_build_bundle_rejects_digest_mismatch_before_registration(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "kernel.ttir").write_text("fail", encoding="utf-8")
    manifest = _manifest(
        stages=[
            {
                "name": "ttir",
                "ordinal": 0,
                "status": "available",
                "format": "mlir",
                "format_schema": "triton-ttir",
                "artifact": {
                    "path": "kernel.ttir",
                    "byte_length": 4,
                    "sha256": hashlib.sha256(b"ttir").hexdigest(),
                },
            }
        ]
    )
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(DomainError, match="sha256 mismatch"):
        KernelBuildImportService(workspace).import_manifest(manifest_path)


def test_kernel_build_import_preserves_distinct_paths_with_same_basename(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    stages: list[dict[str, object]] = []
    for ordinal, directory in enumerate(("kernel-a", "kernel-b")):
        artifact_path = tmp_path / directory / "kernel.ptx"
        artifact_path.parent.mkdir()
        payload = f"ptx-{ordinal}".encode()
        artifact_path.write_bytes(payload)
        stages.append(
            {
                "name": f"ptx-{ordinal}",
                "ordinal": ordinal,
                "status": "available",
                "format": "ptx",
                "format_schema": "nvidia-ptx",
                "artifact": {
                    "path": f"{directory}/kernel.ptx",
                    "byte_length": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "media_type": "text/plain",
                },
            }
        )
    manifest_path = tmp_path / "kernel-build.json"
    manifest_path.write_text(_manifest(stages=stages).model_dump_json())

    result = KernelBuildImportService(workspace).import_manifest(manifest_path)

    assert [item.display_name for item in result.run.artifacts[1:]] == [
        "kernel-a/kernel.ptx",
        "kernel-b/kernel.ptx",
    ]
