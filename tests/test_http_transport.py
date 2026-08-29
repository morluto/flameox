from __future__ import annotations

import ast
import io
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from flameox.http_transport import (
    BoundedHttpClient,
    BoundedHttpError,
    HttpFailureKind,
    HttpMethod,
    LoopbackHttpRequest,
    ManagedDownloadRequest,
)

pytestmark = pytest.mark.unit


def _loopback_request(path: str = "/health", *, maximum: int = 1024) -> LoopbackHttpRequest:
    return LoopbackHttpRequest(
        base_url="http://127.0.0.1:8000",
        method=HttpMethod.GET,
        path=path,
        deadline_monotonic=time.monotonic() + 5,
        max_response_bytes=maximum,
    )


def _stream_response(
    status_code: int,
    body: bytes,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers=headers,
        stream=httpx.ByteStream(body),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", "http://example.com:8000"),
        ("base_url", "http://user:secret@127.0.0.1:8000"),
        ("base_url", "http://127.0.0.1:8000/control"),
        ("path", "//example.com/health"),
        ("path", "/v1/../health"),
    ],
)
def test_loopback_request_rejects_authority_and_path_escape(field: str, value: str) -> None:
    values: dict[str, object] = {
        "base_url": "http://127.0.0.1:8000",
        "method": HttpMethod.GET,
        "path": "/health",
        "deadline_monotonic": time.monotonic() + 5,
        "max_response_bytes": 1024,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        LoopbackHttpRequest.model_validate(values)


def test_loopback_client_rejects_redirects_and_never_follows_them() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://example.com/escape"})

    with (
        BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client,
        pytest.raises(BoundedHttpError) as caught,
    ):
        client.request_loopback(_loopback_request())

    assert caught.value.kind is HttpFailureKind.REDIRECT
    assert calls == ["http://127.0.0.1:8000/health"]


def test_loopback_client_enforces_streaming_byte_ceiling_and_identity_encoding() -> None:
    responses = iter(
        (
            _stream_response(200, b"too large"),
            _stream_response(200, b"compressed", headers={"content-encoding": "gzip"}),
        )
    )

    with BoundedHttpClient(
        sync_transport=httpx.MockTransport(lambda _request: next(responses))
    ) as client:
        with pytest.raises(BoundedHttpError) as oversized:
            client.request_loopback(_loopback_request(maximum=3))
        with pytest.raises(BoundedHttpError) as encoded:
            client.request_loopback(_loopback_request())

    assert oversized.value.kind is HttpFailureKind.RESPONSE_TOO_LARGE
    assert encoded.value.kind is HttpFailureKind.CONTENT_ENCODING


def test_json_decoding_requires_declared_type_and_bounded_valid_json() -> None:
    responses = iter(
        (
            _stream_response(200, b"{}", headers={"content-type": "text/plain"}),
            _stream_response(
                200,
                b"{not-json}",
                headers={"content-type": "application/json"},
            ),
            _stream_response(
                200,
                b'{"data":[{"id":"model"}]}',
                headers={"content-type": "application/json"},
            ),
        )
    )

    with BoundedHttpClient(
        sync_transport=httpx.MockTransport(lambda _request: next(responses))
    ) as client:
        with pytest.raises(BoundedHttpError) as wrong_type:
            client.request_loopback(_loopback_request()).json()
        with pytest.raises(BoundedHttpError) as malformed:
            client.request_loopback(_loopback_request()).json()
        payload = client.request_loopback(_loopback_request()).json()

    assert wrong_type.value.kind is HttpFailureKind.CONTENT_TYPE
    assert malformed.value.kind is HttpFailureKind.MALFORMED_JSON
    assert payload == {"data": [{"id": "model"}]}


def test_expired_deadline_fails_before_transport_access() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    request = _loopback_request().model_copy(update={"deadline_monotonic": time.monotonic() - 1})
    with (
        BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client,
        pytest.raises(BoundedHttpError) as caught,
    ):
        client.request_loopback(request)

    assert caught.value.kind is HttpFailureKind.TIMEOUT
    assert called is False


def test_managed_download_follows_only_allowlisted_redirects_and_hashes_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "downloads.example.com":
            return httpx.Response(302, headers={"location": "https://assets.example.com/tool"})
        return _stream_response(200, b"artifact")

    destination = io.BytesIO()
    request = ManagedDownloadRequest(
        url="https://downloads.example.com/tool",
        allowed_origins=("https://downloads.example.com", "https://assets.example.com"),
        deadline_monotonic=time.monotonic() + 5,
        max_response_bytes=100,
        max_redirects=1,
    )
    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        receipt = client.download(request, destination)

    assert destination.getvalue() == b"artifact"
    assert receipt.total_bytes == 8
    assert receipt.response_start == 0


def test_managed_download_rejects_redirect_outside_declared_origins() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            302,
            headers={"location": "https://unexpected.example.com/tool"},
        )
    )
    request = ManagedDownloadRequest(
        url="https://downloads.example.com/tool",
        allowed_origins=("https://downloads.example.com",),
        deadline_monotonic=time.monotonic() + 5,
        max_response_bytes=100,
        max_redirects=1,
    )

    with (
        BoundedHttpClient(sync_transport=transport) as client,
        pytest.raises(ValueError, match="escaped"),
    ):
        client.download(request, io.BytesIO())


def test_production_http_construction_is_owned_by_transport_boundary() -> None:
    source_root = Path(__file__).parents[1] / "src" / "flameox"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        if path.name == "http_transport.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "http.client",
                "urllib.request",
            }:
                violations.append(f"{path.relative_to(source_root)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"http.client", "urllib.request", "requests"}:
                        violations.append(f"{path.relative_to(source_root)}:{node.lineno}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "httpx"
                and node.func.attr in {"Client", "AsyncClient", "request", "stream"}
            ):
                violations.append(f"{path.relative_to(source_root)}:{node.lineno}")

    assert violations == []
