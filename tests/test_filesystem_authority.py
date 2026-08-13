from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.filesystem_authority import BoundDirectoryReference, TrustedRoot
from tests.support.execution import executable_binding

pytestmark = [
    pytest.mark.unit,
    pytest.mark.process,
    pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor authority"),
]


def test_reference_rejects_non_normalized_relative_paths() -> None:
    for value in ("../escape", "nested/../escape", "/absolute", "nested//file"):
        with pytest.raises((DomainError, ValueError)):
            BoundDirectoryReference.model_validate(
                {
                    "relative_path": value,
                    "identity": {"device": 1, "inode": 1},
                }
            )


def test_trusted_root_refuses_replaced_operation_directory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    staging.mkdir()
    outside.mkdir()

    with TrustedRoot(staging) as root:
        with root.allocate_directory("operations/run") as operation:
            reference = operation.reference
            operation.write_bytes("result.json", b"owned")

        original = staging / "operations" / "run"
        parked = staging / "operations" / "parked"
        original.rename(parked)
        original.symlink_to(outside, target_is_directory=True)

        with pytest.raises(DomainError) as caught:
            root.open_directory(reference)

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert not (outside / "result.json").exists()
    assert (parked / "result.json").read_bytes() == b"owned"


def test_held_directory_descriptor_survives_lexical_path_swap(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    staging.mkdir()
    outside.mkdir()

    with TrustedRoot(staging) as root, root.allocate_directory("operations/run") as operation:
        original = staging / "operations" / "run"
        parked = staging / "operations" / "parked"
        original.rename(parked)
        original.symlink_to(outside, target_is_directory=True)

        operation.write_bytes("result.json", b"descriptor-owned")

    assert (parked / "result.json").read_bytes() == b"descriptor-owned"
    assert not (outside / "result.json").exists()


def test_subprocess_can_only_write_through_explicit_inherited_directory(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    staging.mkdir()
    outside.mkdir()

    with TrustedRoot(staging) as root, root.allocate_directory("operations/run") as operation:
        original = staging / "operations" / "run"
        parked = staging / "operations" / "parked"
        original.rename(parked)
        original.symlink_to(outside, target_is_directory=True)
        child_path = operation.child_process_path("result.json")
        script = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'child')"
        request = ExecutionRequest(
            argv=(sys.executable, "-c", script, str(child_path)),
            executable_binding=executable_binding(Path(sys.executable)),
            cwd=tmp_path,
            allowed_working_roots=(tmp_path,),
            inherited_directory_fds=operation.inherited_descriptors(),
        )

        outcome = SubprocessBroker().run_sync(request)

    assert outcome.process.exit_code == 0
    assert (parked / "result.json").read_bytes() == b"child"
    assert not (outside / "result.json").exists()


def test_output_manifest_rejects_hard_links(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    source = tmp_path / "source.json"
    source.write_text("secret")

    with TrustedRoot(staging) as root, root.allocate_directory("operations/run") as operation:
        os.link(source, operation.absolute_display_path("result.json"))

        with pytest.raises(DomainError) as caught:
            operation.admitted_files(frozenset({"result.json"}))

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_output_manifest_binds_the_exact_admitted_file(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with TrustedRoot(staging) as root, root.allocate_directory("operations/run") as operation:
        result_path = operation.absolute_display_path("result.json")
        result_path.write_text("admitted")
        (reference,) = operation.admitted_files(frozenset({"result.json"}))
        result_path.unlink()
        result_path.write_text("replacement")

        with pytest.raises(DomainError) as caught, operation.open_file(reference):
            pass

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
