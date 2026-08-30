from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.application.kernel_builds import qualify_triton_compiler_target
from flameox.domain import (
    AcceleratorDevice,
    AcceleratorIdentityFacet,
    AcceleratorIdentityStatus,
    CompilerIdentity,
    CompilerQualification,
    CompilerTarget,
    CompilerTargetIdentity,
    DomainError,
    IdentityQuality,
    compiler_identity_id,
    compiler_target_identity_id,
    digest_model,
)


def _compiler() -> CompilerIdentity:
    return CompilerIdentity(
        adapter="triton.compiler",
        distribution="triton",
        version="3.7.1",
        content_digest="sha256:" + "1" * 64,
        interpreter_digest="sha256:" + "2" * 64,
    )


def _accelerator() -> AcceleratorIdentityFacet:
    return AcceleratorIdentityFacet(
        provider="cuda",
        status=AcceleratorIdentityStatus.AVAILABLE,
        identity_quality=IdentityQuality.EXACT,
        driver_version="575.57",
        runtime_version="12.8",
        devices=(AcceleratorDevice(index=0, compute_capability="8.6"),),
    )


def _events(path: Path, *, cache_hit: bool, target: str = "sm_86", version: str = "3.7.1") -> None:
    path.write_text(
        json.dumps(
            {
                "cache_hit": cache_hit,
                "target": {"backend": "cuda", "architecture": target, "warp_size": 32},
                "triton_version": version,
            }
        )
        + "\n"
    )


def _ptx(path: Path, *, target: str = "sm_86") -> None:
    path.write_text(f".version 8.6\n.target {target}\n")


def test_compiler_identity_is_derived_from_exact_distribution_and_target() -> None:
    qualification = CompilerQualification(
        compiler=_compiler(),
        target=CompilerTargetIdentity(
            backend="cuda",
            architecture="sm_86",
            warp_size=32,
            ptx_version="8.6",
            environment_id="sha256:" + "3" * 64,
        ),
    )

    assert compiler_identity_id(qualification) == digest_model(
        qualification.compiler.model_dump(mode="json")
    )
    assert qualification.target is not None
    assert compiler_target_identity_id(qualification) == digest_model(
        qualification.target.model_dump(mode="json")
    )
    assert compiler_identity_id(None) is None
    assert compiler_target_identity_id(CompilerQualification(compiler=_compiler())) is None
    with pytest.raises(ValidationError):
        CompilerTargetIdentity.model_validate(
            {"backend": "cuda", "architecture": "sm_86", "warp_size": 16, "environment_id": "bad"}
        )


def test_triton_target_listener_qualifies_cache_hits_like_fresh_compiles(tmp_path: Path) -> None:
    fresh_events = tmp_path / "fresh.jsonl"
    cached_events = tmp_path / "cached.jsonl"
    ptx = tmp_path / "kernel.ptx"
    _events(fresh_events, cache_hit=False)
    _events(cached_events, cache_hit=True)
    _ptx(ptx)

    fresh, fresh_limitations = qualify_triton_compiler_target(
        events_path=fresh_events,
        native_paths=(ptx,),
        compiler=_compiler(),
        environment_id="sha256:" + "3" * 64,
        accelerator=_accelerator(),
        target_intent=None,
    )
    cached, cached_limitations = qualify_triton_compiler_target(
        events_path=cached_events,
        native_paths=(ptx,),
        compiler=_compiler(),
        environment_id="sha256:" + "3" * 64,
        accelerator=_accelerator(),
        target_intent=None,
    )

    assert fresh == cached
    assert fresh is not None
    assert fresh_limitations == cached_limitations == ()


def test_triton_target_allows_only_matching_cross_compile_intent(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    ptx = tmp_path / "kernel.ptx"
    _events(events, cache_hit=False, target="sm_90")
    _ptx(ptx, target="sm_90")

    target, limitations = qualify_triton_compiler_target(
        events_path=events,
        native_paths=(ptx,),
        compiler=_compiler(),
        environment_id="sha256:" + "3" * 64,
        accelerator=_accelerator(),
        target_intent=CompilerTarget(backend="cuda", architecture="sm_90", warp_size=32),
    )

    assert target is not None
    assert target.architecture == "sm_90"
    assert limitations == ()


def test_triton_target_is_partial_without_authoritative_cuda_identity(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    ptx = tmp_path / "kernel.ptx"
    _events(events, cache_hit=False)
    _ptx(ptx)

    target, limitations = qualify_triton_compiler_target(
        events_path=events,
        native_paths=(ptx,),
        compiler=_compiler(),
        environment_id="sha256:" + "3" * 64,
        accelerator=None,
        target_intent=None,
    )

    assert target is None
    assert limitations == ("Observed CUDA environment identity was unavailable.",)


@pytest.mark.parametrize(
    ("event_target", "ptx_target", "version"),
    (("sm_86", "sm_90", "3.7.1"), ("sm_86", "sm_86", "3.7.2")),
)
def test_triton_target_rejects_contradictory_metadata(
    tmp_path: Path,
    event_target: str,
    ptx_target: str,
    version: str,
) -> None:
    events = tmp_path / "events.jsonl"
    ptx = tmp_path / "kernel.ptx"
    _events(events, cache_hit=False, target=event_target, version=version)
    _ptx(ptx, target=ptx_target)

    with pytest.raises(DomainError):
        qualify_triton_compiler_target(
            events_path=events,
            native_paths=(ptx,),
            compiler=_compiler(),
            environment_id="sha256:" + "3" * 64,
            accelerator=_accelerator(),
            target_intent=None,
        )
