from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flameox.command_binding import ExecutableResolver
from flameox.executable_models import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
)
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.runtime_errors import DomainError, ErrorCode

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


@pytest.mark.anyio
async def test_broker_executes_the_bound_executable_without_repeating_path_search(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "bin" / ("tool.exe" if os.name == "nt" else "tool")
    executable.parent.mkdir()
    try:
        executable.symlink_to(sys.executable)
    except OSError:
        pytest.skip("creating an executable symlink is unavailable on this host")
    binding = ExecutableResolver().resolve(
        ExecutableResolutionRequest(
            token="tool",
            cwd=tmp_path,
            environment={"PATH": "bin"},
            policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
        )
    )

    outcome = await SubprocessBroker().run(
        ExecutionRequest(
            argv=("tool", "-c", "print('bound executable')"),
            executable_binding=binding,
            cwd=tmp_path,
            environment_allowlist=(),
            environment_overrides={"PATH": ""},
            allowed_working_roots=(tmp_path,),
        )
    )

    assert outcome.stdout == b"bound executable\n"
    assert outcome.executable_binding == binding


@pytest.mark.anyio
async def test_broker_rejects_an_executable_changed_after_binding(tmp_path: Path) -> None:
    executable = tmp_path / ("tool.exe" if os.name == "nt" else "tool")
    shutil.copy2(sys.executable, executable)
    executable.chmod(0o755)
    binding = ExecutableResolver().resolve(
        ExecutableResolutionRequest(
            token=str(executable),
            cwd=tmp_path,
            environment={},
            policy=ExecutableTrustPolicy.PROJECT_BOUND,
            allowed_roots=(tmp_path,),
        )
    )
    with executable.open("ab") as stream:
        stream.write(b"changed after planning")

    with pytest.raises(DomainError) as caught:
        await SubprocessBroker().run(
            ExecutionRequest(
                argv=(str(executable), "-c", "from pathlib import Path; Path('executed').touch()"),
                executable_binding=binding,
                cwd=tmp_path,
                allowed_working_roots=(tmp_path,),
            )
        )

    assert caught.value.code is ErrorCode.MISSING_OR_CHANGED_INPUT
    assert not (tmp_path / "executed").exists()


@pytest.mark.skipif(os.name != "posix", reason="FIFOs require POSIX")
@pytest.mark.parametrize(
    "boundary",
    [
        "filesystem",
        "executable",
        "repository",
        "manifest",
        "artifact_metadata",
        "native_copy",
        "native_hash",
    ],
)
def test_special_files_are_rejected_without_blocking(tmp_path: Path, boundary: str) -> None:
    script = """
import os, shutil, sys
from pathlib import Path
from flameox.command_binding import ExecutableResolver
from flameox.executable_models import ExecutableResolutionRequest, ExecutableTrustPolicy
from flameox.filesystem import BoundedFileSystem
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RuntimeFailure
from flameox.source_files import NativeSource, copy_verified_file, sha256_file
root = Path(sys.argv[1])
path = root / 'native'
if sys.argv[2] in {'native_copy', 'native_hash'}:
    path.write_bytes(b'admitted')
    digest, size = sha256_file(path)
    source = NativeSource(path, digest, size, 'text', None, 'input')
    path.unlink()
repository_boundary = sys.argv[2] in {'repository', 'manifest', 'artifact_metadata'}
if repository_boundary:
    path.write_text('native evidence')
    runtime = AnalysisRuntime(evidence_directory=root / 'store')
    result = runtime.analyze('artifact.preview', [PathSource(path=str(path))], {})
    evidence_id = runtime.preserve_evidence(result['analysis_id'])['evidence_id']
    if sys.argv[2] == 'repository':
        path = root / 'store' / 'repository.json'
    elif sys.argv[2] == 'manifest':
        path = (root / 'store' / 'evidence' / 'sha256'
                / evidence_id[:2] / evidence_id / 'manifest.json')
    else:
        path = next((root / 'store' / 'artifacts' / 'sha256').glob('*/*/artifact.json'))
    path.unlink()
if sys.argv[2] == 'executable':
    shutil.copy2(sys.executable, path)
    resolver = ExecutableResolver()
    binding = resolver.resolve(ExecutableResolutionRequest(
        token=str(path), cwd=root, environment={},
        policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL))
    path.unlink()
os.mkfifo(path)
try:
    if repository_boundary:
        runtime.read_evidence(evidence_id)
    elif sys.argv[2] == 'filesystem':
        BoundedFileSystem((root,)).read_bytes(path, max_bytes=1024)
    elif sys.argv[2] == 'native_copy':
        copy_verified_file(source, root / 'copy')
    elif sys.argv[2] == 'native_hash':
        sha256_file(path)
    else:
        resolver.revalidate(binding)
except (DomainError, RuntimeFailure) as error:
    expected = ('REPOSITORY_CORRUPTION' if repository_boundary
                else 'EXECUTION_FAILURE' if sys.argv[2] == 'filesystem'
                else 'INVALID_INPUT' if sys.argv[2] == 'native_hash'
                else 'MISSING_OR_CHANGED_INPUT')
    assert error.code == expected
else:
    raise AssertionError('special file was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.anyio
@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
async def test_executable_symlink_loops_return_typed_failures_before_launch(tmp_path: Path) -> None:
    from flameox.runtime import AnalysisRuntime
    from flameox.runtime_contracts import CaptureTarget, RuntimeFailure

    executable = tmp_path / "tool"
    executable.symlink_to(sys.executable)
    resolver = ExecutableResolver()
    binding = resolver.require_host_tool(str(executable), cwd=tmp_path)
    executable.unlink()
    executable.symlink_to(executable)
    with pytest.raises(DomainError) as changed:
        await SubprocessBroker().run(
            ExecutionRequest(
                argv=(str(executable), "-c", "pass"),
                executable_binding=binding,
                cwd=tmp_path,
                allowed_working_roots=(tmp_path,),
            )
        )
    assert changed.value.code is ErrorCode.MISSING_OR_CHANGED_INPUT
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as admission:
            await runtime.capture_and_analyze(
                CaptureTarget(argv=[str(executable)], cwd=str(tmp_path), provider_id="direct"),
                "artifact.preview",
            )
        assert admission.value.code == "EXECUTION_FAILURE"
        assert str(tmp_path) not in admission.value.message
        assert not (tmp_path / "store").exists()
    finally:
        runtime.close()


@pytest.mark.anyio
@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
@pytest.mark.parametrize("boundary", ["cwd", "allowed_root"])
async def test_working_directory_symlink_loops_return_typed_failures(
    tmp_path: Path, boundary: str
) -> None:
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    binding = ExecutableResolver().require_host_tool(sys.executable, cwd=tmp_path)
    with pytest.raises(DomainError) as failure:
        await SubprocessBroker().run(
            ExecutionRequest(
                argv=(sys.executable, "-c", "raise SystemExit('must not launch')"),
                executable_binding=binding,
                cwd=loop if boundary == "cwd" else tmp_path,
                allowed_working_roots=(loop if boundary == "allowed_root" else tmp_path,),
            )
        )
    assert failure.value.code is ErrorCode.INVALID_INPUT
    assert str(tmp_path) not in failure.value.message


@pytest.mark.skipif(os.name != "posix", reason="FIFOs require POSIX")
@pytest.mark.parametrize("format_name", ["text", "json", "jsonl", "csv", "parquet", "decoder"])
def test_native_reader_rejects_fifo_replacement_after_admission(
    tmp_path: Path, format_name: str
) -> None:
    script = """
import os, sys
from pathlib import Path
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RuntimeFailure
from flameox.providers.perfetto import PerfettoProvider
root = Path(sys.argv[1])
path = root / "native"
path.write_text('{"row":1}')
runtime = AnalysisRuntime(evidence_directory=root / "store")
if sys.argv[2] == "decoder":
    path.unlink()
    os.mkfifo(path)
    try:
        PerfettoProvider._identity(path)
    except OSError:
        pass
    else:
        raise AssertionError("special decoder accepted")
else:
    original = runtime._read_rows
    def changed(*args, **kwargs):
        path.unlink()
        os.mkfifo(path)
        return original(*args, **kwargs)
    runtime._read_rows = changed
    try:
        runtime.analyze("artifact.preview", [PathSource(path=str(path), format=sys.argv[2])], {})
    except RuntimeFailure as error:
        assert error.code == "DECODE_FAILURE", error.code
    else:
        raise AssertionError("special input accepted")
runtime.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), format_name],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
