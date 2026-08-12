"""Tests for NVBench JSON + jsonbin bundle import and extraction."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.nvbench import NvbenchExtractor
from flameox.adapters.options import bind_adapter_options
from flameox.application import NvbenchImportService
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.storage import GenerationManifest, RunStore, Workspace


def _float32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _nvbench_json(
    *,
    sidecar_filename: str = "out.json-bin/0.bin",
    sample_count: int = 3,
    benchmark_name: str = "cub.bench.scan",
    state_name: str = "[T=I32 Elements=2^16]",
    device_id: int = 0,
    json_version: tuple[int, int, int] = (1, 0, 0),
    nvbench_version: tuple[int, int, int] = (0, 1, 0),
) -> str:
    return json.dumps(
        {
            "meta": {
                "argv": ["bench", "-d", "0", "--jsonbin", "out.json"],
                "version": {
                    "json": {
                        "major": json_version[0],
                        "minor": json_version[1],
                        "patch": json_version[2],
                        "string": ".".join(str(v) for v in json_version),
                    },
                    "nvbench": {
                        "major": nvbench_version[0],
                        "minor": nvbench_version[1],
                        "patch": nvbench_version[2],
                        "string": ".".join(str(v) for v in nvbench_version),
                    },
                },
            },
            "devices": [
                {"id": device_id, "name": "NVIDIA GeForce RTX 4090"},
            ],
            "benchmarks": [
                {
                    "name": benchmark_name,
                    "index": 0,
                    "axes": [],
                    "states": [
                        {
                            "name": state_name,
                            "device": device_id,
                            "type_config_index": 0,
                            "is_skipped": False,
                            "summaries": [
                                {
                                    "tag": "nv/json/bin:sample_times",
                                    "name": "Sample Times File",
                                    "description": (
                                        "Binary file containing sample times "
                                        "as little-endian float32."
                                    ),
                                    "hint": "file/sample_times",
                                    "hide": "Not needed in table.",
                                    "data": [
                                        {
                                            "name": "filename",
                                            "type": "string",
                                            "value": sidecar_filename,
                                        },
                                        {
                                            "name": "size",
                                            "type": "int64",
                                            # NVBench json_printer.cu serializes
                                            # int64 values as decimal strings.
                                            "value": str(sample_count),
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
    )


def test_nvbench_extracts_float32_sample_times_without_lossy_conversion(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    samples = [0.004571, 0.004580, 0.004562]
    json_path = tmp_path / "nvbench.json"
    sidecar_path = tmp_path / "out.json-bin" / "0.bin"
    sidecar_path.parent.mkdir()
    document = json.loads(_nvbench_json(sample_count=len(samples)))
    document["benchmarks"][0]["states"].append(
        {
            "name": "[T=F32 Elements=2^20]",
            "device": 0,
            "is_skipped": True,
            "skip_reason": "No CUDA device",
            "summaries": [],
        }
    )
    json_path.write_text(json.dumps(document))
    sidecar_path.write_bytes(_float32_bytes(samples))

    run_id = NvbenchImportService(workspace).import_json(json_path).run.run_id

    result = NvbenchExtractor(workspace).extract(run_id)
    repeated = NvbenchExtractor(workspace).extract(run_id)

    assert result.measurement_count == 3
    assert result.benchmark_count == 1
    assert result.producer_version == "0.1.0"
    assert result.limitations == ()
    assert repeated.corpus_commit_id == result.corpus_commit_id
    run = RunStore(workspace).read(run_id)
    expected_input_ids = {
        registration.artifact_id
        for registration in run.artifacts
        if registration.role in {"primary", "nvbench_sidecar"}
    }
    head = workspace.corpus.read_head()
    generations = [
        GenerationManifest.model_validate_json(
            (workspace.paths.root / relative_path).read_text(encoding="utf-8")
        )
        for relative_path in head.generation_manifests
    ]
    generation = next(
        manifest for manifest in generations if manifest.publisher == NvbenchExtractor.name
    )
    assert set(generation.input_artifact_ids) == expected_input_ids
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, value_int, value_float, unit, is_warmup, "
            "dimensions FROM measurements WHERE run_id = ? "
            "ORDER BY value_index",
            (run_id,),
        ).fetchall()
    assert len(rows) == 3
    for row in rows:
        name, value_int, value_float, unit, is_warmup, dimensions = row
        assert name == "nvbench.cub.bench.scan.sample_times"
        assert value_int is None
        assert value_float is not None
        assert unit == "seconds"
        assert is_warmup is False
        assert dimensions["nvbench_benchmark"] == "cub.bench.scan"
    assert rows[0][2] == pytest.approx(0.004571, abs=1e-6)


def test_nvbench_rejects_nonfinite_binary_samples(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "nvbench.json"
    sidecar_path = tmp_path / "out.json-bin" / "0.bin"
    sidecar_path.parent.mkdir()
    json_path.write_text(_nvbench_json(sample_count=1))
    sidecar_path.write_bytes(_float32_bytes([float("nan")]))
    run_id = NvbenchImportService(workspace).import_json(json_path).run.run_id

    with pytest.raises(DomainError, match="non-finite"):
        NvbenchExtractor(workspace).extract(run_id)


def test_nvbench_measurement_identity_includes_consumed_sidecar(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "nvbench.json"
    sidecar_path = tmp_path / "out.json-bin" / "0.bin"
    sidecar_path.parent.mkdir()
    json_path.write_text(_nvbench_json(sample_count=1))
    sidecar_path.write_bytes(_float32_bytes([0.001]))
    first_run = NvbenchImportService(workspace).import_json(json_path).run.run_id
    first = NvbenchExtractor(workspace).extract(first_run)

    sidecar_path.write_bytes(_float32_bytes([0.002]))
    second_run = NvbenchImportService(workspace).import_json(json_path).run.run_id
    second = NvbenchExtractor(workspace).extract(second_run)

    with Catalog(workspace).open_snapshot() as snapshot:
        measurement_ids = snapshot.execute(
            "SELECT run_id, measurement_id FROM measurements "
            "WHERE run_id IN (?, ?) ORDER BY run_id",
            (first_run, second_run),
        ).fetchall()
    assert first.corpus_commit_id != second.corpus_commit_id
    assert len({measurement_id for _, measurement_id in measurement_ids}) == 2


@pytest.mark.parametrize("invalid_size", ["-5", 3], ids=["negative", "numeric-json"])
def test_nvbench_rejects_sidecar_sizes_outside_documented_encoding(
    tmp_path: Path,
    invalid_size: object,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "nvbench.json"
    document = json.loads(_nvbench_json(sample_count=3))
    document["benchmarks"][0]["states"][0]["summaries"][0]["data"][1]["value"] = invalid_size
    json_path.write_text(json.dumps(document))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_nvbench_rejects_declared_samples_over_row_budget_before_decoding(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "nvbench.json"
    sidecar_path = tmp_path / "out.json-bin" / "0.bin"
    sidecar_path.parent.mkdir()
    document = json.loads(_nvbench_json(sample_count=2))
    summaries = document["benchmarks"][0]["states"][0]["summaries"]
    summaries.append(
        {**summaries[0], "tag": "nv/json/bin:sample_freqs", "hint": "file/sample_freqs"}
    )
    summaries[1]["data"] = [dict(datum) for datum in summaries[1]["data"]]
    summaries[1]["data"][0]["value"] = "out.json-bin/1.bin"
    json_path.write_text(json.dumps(document))
    # Import verifies the byte length, while extraction must reject the declared
    # row allocation before decoding these non-finite values.
    sidecar_path.write_bytes(_float32_bytes([float("nan"), float("nan")]))
    second_sidecar = tmp_path / "out.json-bin" / "1.bin"
    second_sidecar.parent.mkdir(parents=True, exist_ok=True)
    second_sidecar.write_bytes(_float32_bytes([float("nan"), float("nan")]))
    run_id = NvbenchImportService(workspace).import_json(json_path).run.run_id
    config = workspace.config.model_copy(
        update={
            "storage": workspace.config.storage.model_copy(update={"max_rows_per_generation": 3})
        }
    )
    workspace.paths.config.write_text(config.to_toml())

    with pytest.raises(DomainError) as error:
        NvbenchExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED
    assert error.value.details == {"rows": 4, "max_rows": 3}


def test_nvbench_rejects_empty_document_without_schema_identity(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "nvbench.json"
    json_path.write_text('{"meta": {}, "benchmarks": []}')

    with pytest.raises(DomainError, match="schema major missing"):
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)


def test_nvbench_import_service_imports_only_declared_sidecars(tmp_path: Path) -> None:
    """Only sidecars referenced by the JSON are imported, not arbitrary siblings."""
    workspace = Workspace.initialize(tmp_path)
    samples = [0.01, 0.02, 0.03]
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / "0.bin"
    # An undeclared sibling that must NOT be imported
    undeclared_path = sidecar_dir / "1.bin"
    json_path.write_text(_nvbench_json(sample_count=len(samples)))
    sidecar_path.write_bytes(_float32_bytes(samples))
    undeclared_path.write_bytes(_float32_bytes([0.99, 0.88]))

    result = NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert result.sidecar_count == 1
    assert len(result.sidecar_artifact_ids) == 1
    # The undeclared file must not appear in the run
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT display_name FROM artifact_registrations "
            "WHERE run_id = ? ORDER BY display_name",
            (result.run.run_id,),
        ).fetchall()
    display_names = [row[0] for row in rows]
    assert "out.json-bin/0.bin" in display_names
    assert "out.json-bin/1.bin" not in display_names


def test_nvbench_import_rejects_bundles_over_public_member_limit(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    document = json.loads(_nvbench_json(sample_count=1))
    summary = document["benchmarks"][0]["states"][0]["summaries"][0]
    document["benchmarks"][0]["states"][0]["summaries"] = [
        {
            **summary,
            "data": [
                {
                    "name": "filename",
                    "type": "string",
                    "value": f"out.json-bin/{index}.bin",
                },
                {"name": "size", "type": "int64", "value": "1"},
            ],
        }
        for index in range(100)
    ]
    json_path.write_text(json.dumps(document))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED
    assert "exceeding the 99-sidecar limit" in error.value.message


def test_nvbench_import_service_binds_expected_byte_length(tmp_path: Path) -> None:
    """Sidecar expected_byte_length = count * 4 is verified after import."""
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / "0.bin"
    # JSON declares 3 samples (12 bytes) but we write 4 samples (16 bytes)
    json_path.write_text(_nvbench_json(sample_count=3))
    sidecar_path.write_bytes(_float32_bytes([0.01, 0.02, 0.03, 0.04]))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)
    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert "byte length mismatch" in error.value.message


def test_nvbench_rejects_unverified_json_schema_major_before_decoding(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_path = tmp_path / "0.bin"
    json_path.write_text(_nvbench_json(sidecar_filename="0.bin", json_version=(2, 0, 0)))
    sidecar_path.write_bytes(_float32_bytes([0.01, 0.02, 0.03]))

    with pytest.raises(DomainError, match="only major 1 is verified"):
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)


def test_nvbench_import_service_rejects_malformed_json(tmp_path: Path) -> None:
    """A unique non-JSON primary leaves no anonymous artifact or run."""
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    json_path.write_text("{not valid json")

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)
    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert tuple(workspace.paths.artifacts.rglob("artifact.json")) == ()
    assert RunStore(workspace).list() == ()
    assert tuple(workspace.paths.staging.iterdir()) == ()


def test_nvbench_import_service_rejects_conflicting_sidecar_sizes(
    tmp_path: Path,
) -> None:
    """Duplicate sidecar filenames with different sizes are rejected."""
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / "0.bin"
    # Build a document with two summaries referencing the same sidecar
    # filename but declaring different sizes.  Both use the documented
    # file/sample_freqs hint.
    doc = json.loads(_nvbench_json(sample_count=3))
    state = doc["benchmarks"][0]["states"][0]
    state["summaries"].append(
        {
            "tag": "sample_freqs",
            "hint": "file/sample_freqs",
            "data": [
                {"name": "filename", "type": "string", "value": "out.json-bin/0.bin"},
                {"name": "size", "type": "int64", "value": "5"},
            ],
        }
    )
    json_path.write_text(json.dumps(doc))
    sidecar_path.write_bytes(_float32_bytes([0.01, 0.02, 0.03]))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)
    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "conflicting sizes" in error.value.message


def test_nvbench_import_service_collapses_duplicate_sidecar_same_size(
    tmp_path: Path,
) -> None:
    """Duplicate sidecar filenames with identical sizes are collapsed."""
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / "0.bin"
    # Build a document with two summaries referencing the same sidecar
    # filename and the same size.  The second summary repeats the same
    # file/sample_times hint (NVBench may emit duplicate references).
    doc = json.loads(_nvbench_json(sample_count=3))
    state = doc["benchmarks"][0]["states"][0]
    state["summaries"].append(
        {
            "tag": "sample_times_dup",
            "hint": "file/sample_times",
            "data": [
                {"name": "filename", "type": "string", "value": "out.json-bin/0.bin"},
                {"name": "size", "type": "int64", "value": "3"},
            ],
        }
    )
    json_path.write_text(json.dumps(doc))
    sidecar_path.write_bytes(_float32_bytes([0.01, 0.02, 0.03]))

    result = NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)
    assert result.sidecar_count == 1
    assert NvbenchExtractor(workspace).extract(result.run.run_id).measurement_count == 3


def test_nvbench_rejects_reused_sidecar_with_conflicting_hint(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    (sidecar_dir / "0.bin").write_bytes(_float32_bytes([0.01, 0.02, 0.03]))
    doc = json.loads(_nvbench_json(sample_count=3))
    doc["benchmarks"][0]["states"][0]["summaries"].append(
        {
            "tag": "sample_freqs",
            "hint": "file/sample_freqs",
            "data": [
                {"name": "filename", "type": "string", "value": "out.json-bin/0.bin"},
                {"name": "size", "type": "int64", "value": "3"},
            ],
        }
    )
    json_path.write_text(json.dumps(doc))

    with pytest.raises(DomainError, match="conflicting sizes or sample hints"):
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)


def test_nvbench_import_service_rejects_unknown_file_hint(
    tmp_path: Path,
) -> None:
    """An unknown file/ hint must be rejected, not silently ignored."""
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / "0.bin"
    doc = json.loads(_nvbench_json(sample_count=3))
    state = doc["benchmarks"][0]["states"][0]
    state["summaries"].append(
        {
            "tag": "unknown_encoding",
            "hint": "file/unknown",
            "data": [
                {"name": "filename", "type": "string", "value": "out.json-bin/0.bin"},
                {"name": "size", "type": "int64", "value": "3"},
            ],
        }
    )
    json_path.write_text(json.dumps(doc))
    sidecar_path.write_bytes(_float32_bytes([0.01, 0.02, 0.03]))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)
    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "Unknown NVBench file hint" in error.value.message


@pytest.mark.parametrize(
    ("missing_name", "remaining_datum"),
    [
        ("filename", {"name": "size", "type": "int64", "value": "3"}),
        (
            "size",
            {
                "name": "filename",
                "type": "string",
                "value": "out.json-bin/0.bin",
            },
        ),
    ],
)
def test_nvbench_import_rejects_incomplete_sidecar_declarations(
    tmp_path: Path,
    missing_name: str,
    remaining_datum: dict[str, str],
) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    document = json.loads(_nvbench_json(sample_count=3))
    document["benchmarks"][0]["states"][0]["summaries"].append(
        {
            "tag": f"sample_times_no_{missing_name}",
            "hint": "file/sample_times",
            "data": [remaining_datum],
        }
    )
    json_path.write_text(json.dumps(document))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "missing" in error.value.message
    assert missing_name in error.value.message


def test_nvbench_import_service_rejects_missing_sidecar(tmp_path: Path) -> None:
    """A declared sidecar that does not exist on disk is rejected."""
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    json_path.write_text(_nvbench_json(sample_count=3))
    # No sidecar directory or file created

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED
    assert "out.json-bin/0.bin" in error.value.message


def test_nvbench_import_service_rejects_sidecar_traversal(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    json_path.write_text(_nvbench_json(sidecar_filename="../outside.bin", sample_count=1))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_nvbench_import_service_rejects_symlinked_sidecar(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(_float32_bytes([0.01]))
    (sidecar_dir / "0.bin").symlink_to(target)
    json_path.write_text(_nvbench_json(sample_count=1))

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path, allow_external_path=True)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


def test_nvbench_import_rejects_symlinked_primary_before_parsing(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-external.json"
    external.write_text(_nvbench_json())
    primary = tmp_path / "out.json"
    primary.symlink_to(external)
    try:
        with pytest.raises(DomainError) as error:
            NvbenchImportService(workspace).import_json(primary)
        assert error.value.code is ErrorCode.EXECUTION_REFUSED
    finally:
        external.unlink()


def test_nvbench_import_applies_provider_document_limit(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    json_path = tmp_path / "out.json"
    json_path.write_text(_nvbench_json())
    with json_path.open("ab") as document:
        document.truncate(16 * 1024 * 1024 + 1)

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(json_path)
    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE


def test_nvbench_import_service_verifies_primary_sha256(tmp_path: Path) -> None:
    """An expected_sha256 mismatch on the primary JSON is rejected."""
    workspace = Workspace.initialize(tmp_path)
    samples = [0.01, 0.02]
    json_path = tmp_path / "out.json"
    sidecar_dir = tmp_path / "out.json-bin"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / "0.bin"
    json_path.write_text(_nvbench_json(sample_count=len(samples)))
    sidecar_path.write_bytes(_float32_bytes(samples))
    wrong_sha = "sha256:" + "0" * 64

    with pytest.raises(DomainError) as error:
        NvbenchImportService(workspace).import_json(
            json_path, allow_external_path=True, expected_sha256=wrong_sha
        )
    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert "sha256 mismatch" in error.value.message


def test_nvbench_capture_invocation_uses_jsonbin_by_default(tmp_path: Path) -> None:
    bound = bind_adapter_options("nvbench", None, project_root=tmp_path)
    invocation = build_capture_invocation(
        "nvbench",
        ("./bench", "-a", "T=I32"),
        tmp_path / "output",
        executable=None,
        options=cast(dict[str, object], bound),
    )
    assert invocation.argv == (
        "./bench",
        "--jsonbin",
        str(tmp_path / "output" / "nvbench.json"),
        "-a",
        "T=I32",
    )


def test_nvbench_capture_invocation_falls_back_to_json_without_jsonbin(
    tmp_path: Path,
) -> None:
    bound = bind_adapter_options(
        "nvbench",
        {"enable_jsonbin": False},
        project_root=tmp_path,
    )
    invocation = build_capture_invocation(
        "nvbench",
        ("./bench",),
        tmp_path / "output",
        executable=None,
        options=cast(dict[str, object], bound),
    )
    assert invocation.argv == (
        "./bench",
        "--json",
        str(tmp_path / "output" / "nvbench.json"),
    )


def test_nvbench_capture_invocation_passes_optional_flags(tmp_path: Path) -> None:
    bound = bind_adapter_options(
        "nvbench",
        {
            "enable_jsonbin": True,
            "stopping_criterion": "entropy",
            "min_samples": 10,
            "timeout": 30.0,
            "devices": "0",
        },
        project_root=tmp_path,
    )
    invocation = build_capture_invocation(
        "nvbench",
        ("./bench",),
        tmp_path / "output",
        executable=None,
        options=cast(dict[str, object], bound),
    )
    assert invocation.argv == (
        "./bench",
        "--jsonbin",
        str(tmp_path / "output" / "nvbench.json"),
        "--stopping-criterion",
        "entropy",
        "--min-samples",
        "10",
        "--timeout",
        "30.0",
        "-d",
        "0",
    )


@pytest.mark.parametrize(
    "conflicting_arguments",
    [
        ("--json", "other.json"),
        ("--jsonbin", "other.json"),
        ("--json=other.json",),
        ("--jsonbin=other.json",),
    ],
)
def test_nvbench_capture_rejects_caller_owned_output(
    tmp_path: Path,
    conflicting_arguments: tuple[str, ...],
) -> None:
    bound = bind_adapter_options("nvbench", None, project_root=tmp_path)
    with pytest.raises(DomainError) as error:
        build_capture_invocation(
            "nvbench",
            ("./bench", *conflicting_arguments),
            tmp_path / "output",
            executable=None,
            options=cast(dict[str, object], bound),
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_nvbench_capture_invocation_rejects_empty_workload(tmp_path: Path) -> None:
    bound = bind_adapter_options("nvbench", None, project_root=tmp_path)
    with pytest.raises(DomainError) as error:
        build_capture_invocation(
            "nvbench",
            (),
            tmp_path / "output",
            executable=None,
            options=cast(dict[str, object], bound),
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_nvbench_options_reject_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as error:
        bind_adapter_options(
            "nvbench",
            {"arbitrary_flag": "--destroy"},
            project_root=tmp_path,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN
