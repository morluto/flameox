from __future__ import annotations

import ipaddress
import json
import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import httpx
from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from flameox.models import ContractModel

_MAX_CONTROL_REQUEST_BYTES = 64 * 1024
_MAX_JSON_DEPTH = 12
_MAX_JSON_VALUES = 10_000
_MAX_JSON_STRING_BYTES = 1024 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)\Z")
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class BinarySink(Protocol):
    def write(self, data: bytes, /) -> int: ...


class HttpFailureKind(StrEnum):
    POLICY = "policy"
    CONNECT = "connect"
    TIMEOUT = "timeout"
    STATUS = "status"
    RESPONSE_TOO_LARGE = "response_too_large"
    CONTENT_TYPE = "content_type"
    CONTENT_ENCODING = "content_encoding"
    MALFORMED_JSON = "malformed_json"
    REDIRECT = "redirect"


class BoundedHttpError(RuntimeError):
    """Safe HTTP failure metadata without response bodies or credential-bearing URLs."""

    def __init__(
        self,
        kind: HttpFailureKind,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class LoopbackHttpRequest(ContractModel):
    base_url: str
    method: HttpMethod
    path: str
    deadline_monotonic: float
    max_response_bytes: int = Field(gt=0, le=16 * 1024 * 1024)
    json_body: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def closed_loopback_request(self) -> LoopbackHttpRequest:
        validate_loopback_base_url(self.base_url)
        _validate_control_path(self.path)
        if self.method in {HttpMethod.GET, HttpMethod.DELETE} and self.json_body is not None:
            raise ValueError(f"{self.method} loopback requests cannot contain JSON")
        return self


class ManagedDownloadRequest(ContractModel):
    url: str
    allowed_origins: tuple[str, ...] = Field(min_length=1, max_length=8)
    deadline_monotonic: float
    max_response_bytes: int = Field(gt=0, le=1024 * 1024 * 1024)
    max_redirects: int = Field(ge=0, le=4, default=0)
    resume_from: int = Field(ge=0, default=0)
    if_range: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def exact_https_authority(self) -> ManagedDownloadRequest:
        _validate_download_url(self.url, self.allowed_origins)
        for origin in self.allowed_origins:
            parsed = httpx.URL(origin)
            if parsed.scheme != "https" or not parsed.host or parsed.path not in {"", "/"}:
                raise ValueError("managed download origins must be exact HTTPS origins")
            if parsed.query or parsed.fragment or parsed.userinfo:
                raise ValueError("managed download origins cannot contain URL extras")
        if self.resume_from > 0 and self.if_range is None:
            raise ValueError("managed download resume requires a representation validator")
        if self.resume_from == 0 and self.if_range is not None:
            raise ValueError("managed download validators are only valid for resume")
        if self.resume_from >= self.max_response_bytes:
            raise ValueError("managed download resume offset exceeds its byte ceiling")
        return self


@dataclass(frozen=True, slots=True)
class BoundedHttpResponse:
    status_code: int
    body: bytes
    content_type: str | None

    def json(self) -> JsonValue:
        content_type = (self.content_type or "").split(";", 1)[0].strip().lower()
        if content_type not in {"application/json", "application/problem+json"}:
            raise BoundedHttpError(
                HttpFailureKind.CONTENT_TYPE,
                "HTTP response did not declare a supported JSON content type.",
                status_code=self.status_code,
            )
        try:
            raw_value = json.loads(self.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BoundedHttpError(
                HttpFailureKind.MALFORMED_JSON,
                "HTTP response contained malformed JSON.",
                status_code=self.status_code,
            ) from error
        try:
            value = _JSON_VALUE_ADAPTER.validate_python(raw_value)
        except ValidationError as error:
            raise BoundedHttpError(
                HttpFailureKind.MALFORMED_JSON,
                "HTTP response did not contain a JSON value.",
                status_code=self.status_code,
            ) from error
        _require_bounded_json(value)
        return value


@dataclass(frozen=True, slots=True)
class DownloadReceipt:
    total_bytes: int
    response_start: int


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    received_bytes: int
    expected_bytes: int
    elapsed_seconds: float
    resume_possible: bool
    validator: str | None


class BoundedHttpClient:
    """The only Flameox-owned HTTPX construction and transport-policy boundary."""

    def __init__(
        self,
        *,
        sync_transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        limits = httpx.Limits(
            max_connections=8,
            max_keepalive_connections=4,
            keepalive_expiry=15.0,
        )
        client_options: dict[str, Any] = {
            "follow_redirects": False,
            "trust_env": False,
            "limits": limits,
            "headers": {
                "Accept-Encoding": "identity",
                "User-Agent": "flameox-bounded-http/1",
            },
        }
        self._client_options = client_options
        self._sync_transport = sync_transport
        self._async_transport = async_transport
        self._sync: httpx.Client | None = None
        self._async: httpx.AsyncClient | None = None

    def __enter__(self) -> BoundedHttpClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    async def __aenter__(self) -> BoundedHttpClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    def _sync_client(self) -> httpx.Client:
        if self._sync is None:
            self._sync = httpx.Client(
                transport=self._sync_transport,
                **self._client_options,
            )
        return self._sync

    def _async_client(self) -> httpx.AsyncClient:
        if self._async is None:
            self._async = httpx.AsyncClient(
                transport=self._async_transport,
                **self._client_options,
            )
        return self._async

    def request_loopback(self, request: LoopbackHttpRequest) -> BoundedHttpResponse:
        url, content, headers = self._control_request_parts(request)
        try:
            with self._sync_client().stream(
                request.method.value,
                url,
                content=content,
                headers=headers,
                timeout=_timeout(request.deadline_monotonic),
            ) as response:
                return self._bounded_response(
                    response,
                    request.max_response_bytes,
                    request.deadline_monotonic,
                )
        except BoundedHttpError:
            raise
        except httpx.TimeoutException as error:
            raise BoundedHttpError(HttpFailureKind.TIMEOUT, "HTTP request timed out.") from error
        except httpx.HTTPError as error:
            raise BoundedHttpError(HttpFailureKind.CONNECT, "HTTP request failed.") from error

    async def request_loopback_async(
        self,
        request: LoopbackHttpRequest,
    ) -> BoundedHttpResponse:
        url, content, headers = self._control_request_parts(request)
        try:
            async with self._async_client().stream(
                request.method.value,
                url,
                content=content,
                headers=headers,
                timeout=_timeout(request.deadline_monotonic),
            ) as response:
                return await self._bounded_response_async(
                    response,
                    request.max_response_bytes,
                    request.deadline_monotonic,
                )
        except BoundedHttpError:
            raise
        except httpx.TimeoutException as error:
            raise BoundedHttpError(HttpFailureKind.TIMEOUT, "HTTP request timed out.") from error
        except httpx.HTTPError as error:
            raise BoundedHttpError(HttpFailureKind.CONNECT, "HTTP request failed.") from error

    def download(
        self,
        request: ManagedDownloadRequest,
        destination: BinarySink,
        *,
        cancel_check: Callable[[], None] | None = None,
        progress: Callable[[DownloadProgress], None] | None = None,
    ) -> DownloadReceipt:
        current = request.url
        redirects = 0
        started = time.monotonic()
        while True:
            _remaining(request.deadline_monotonic)
            headers = (
                {"Range": f"bytes={request.resume_from}-", "If-Range": request.if_range}
                if request.resume_from > 0 and request.if_range is not None
                else None
            )
            with self._download_response(
                current,
                request.deadline_monotonic,
                headers=headers,
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    if redirects >= request.max_redirects:
                        raise BoundedHttpError(
                            HttpFailureKind.REDIRECT,
                            "Managed download exceeded its allowed redirect chain.",
                            status_code=response.status_code,
                        )
                    location = response.headers.get("location")
                    if not location:
                        raise BoundedHttpError(
                            HttpFailureKind.REDIRECT,
                            "Managed download redirect omitted its target.",
                            status_code=response.status_code,
                        )
                    current = str(response.url.join(location))
                    _validate_download_url(current, request.allowed_origins)
                    redirects += 1
                    continue
                _require_success(response)
                _require_identity_encoding(response)
                response_start, range_supported, declared_response_bytes = _resume_response(
                    response,
                    requested_offset=request.resume_from,
                    maximum=request.max_response_bytes,
                )
                validator = _strong_validator(response)
                if response_start > 0 and validator is not None and validator != request.if_range:
                    if progress is not None:
                        progress(
                            DownloadProgress(
                                received_bytes=request.resume_from,
                                expected_bytes=request.max_response_bytes,
                                elapsed_seconds=max(0.0, time.monotonic() - started),
                                resume_possible=False,
                                validator=validator,
                            )
                        )
                    raise BoundedHttpError(
                        HttpFailureKind.POLICY,
                        "Managed download response changed its range validator.",
                        status_code=response.status_code,
                    )
                response_bytes = 0
                for chunk in response.iter_raw(chunk_size=64 * 1024):
                    if cancel_check is not None:
                        cancel_check()
                    _remaining(request.deadline_monotonic)
                    response_bytes += len(chunk)
                    total = response_start + response_bytes
                    if total > request.max_response_bytes:
                        raise BoundedHttpError(
                            HttpFailureKind.RESPONSE_TOO_LARGE,
                            "Managed download exceeded its byte ceiling.",
                        )
                    destination.write(chunk)
                    if progress is not None:
                        progress(
                            DownloadProgress(
                                received_bytes=total,
                                expected_bytes=request.max_response_bytes,
                                elapsed_seconds=max(0.0, time.monotonic() - started),
                                resume_possible=validator is not None
                                and (request.resume_from == 0 or range_supported),
                                validator=validator,
                            )
                        )
                if (
                    declared_response_bytes is not None
                    and response_bytes != declared_response_bytes
                ):
                    raise BoundedHttpError(
                        HttpFailureKind.POLICY,
                        "Managed download body contradicted its Content-Range.",
                        status_code=response.status_code,
                    )
                return DownloadReceipt(
                    total_bytes=response_start + response_bytes,
                    response_start=response_start,
                )

    @contextmanager
    def _download_response(
        self,
        url: str,
        deadline_monotonic: float,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[httpx.Response]:
        try:
            with self._sync_client().stream(
                "GET",
                url,
                headers=headers,
                timeout=_timeout(deadline_monotonic),
            ) as response:
                yield response
        except BoundedHttpError:
            raise
        except httpx.TimeoutException as error:
            raise BoundedHttpError(HttpFailureKind.TIMEOUT, "HTTP download timed out.") from error
        except httpx.HTTPError as error:
            raise BoundedHttpError(HttpFailureKind.CONNECT, "HTTP download failed.") from error

    @staticmethod
    def _control_request_parts(
        request: LoopbackHttpRequest,
    ) -> tuple[str, bytes | None, Mapping[str, str]]:
        base_url = validate_loopback_base_url(request.base_url)
        content: bytes | None = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if request.json_body is not None:
            content = json.dumps(
                request.json_body,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            if len(content) > _MAX_CONTROL_REQUEST_BYTES:
                raise BoundedHttpError(
                    HttpFailureKind.POLICY,
                    "Loopback control request exceeded its body limit.",
                )
            headers["Content-Type"] = "application/json"
        return f"{base_url}{request.path}", content, headers

    @staticmethod
    def _bounded_response(
        response: httpx.Response,
        maximum: int,
        deadline_monotonic: float,
    ) -> BoundedHttpResponse:
        _require_success(response)
        _require_identity_encoding(response)
        _require_content_length_bound(response, maximum)
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_raw(chunk_size=64 * 1024):
            _remaining(deadline_monotonic)
            total += len(chunk)
            if total > maximum:
                raise BoundedHttpError(
                    HttpFailureKind.RESPONSE_TOO_LARGE,
                    "HTTP response exceeded its byte ceiling.",
                    status_code=response.status_code,
                )
            chunks.append(chunk)
        return BoundedHttpResponse(
            status_code=response.status_code,
            body=b"".join(chunks),
            content_type=response.headers.get("content-type"),
        )

    @staticmethod
    async def _bounded_response_async(
        response: httpx.Response,
        maximum: int,
        deadline_monotonic: float,
    ) -> BoundedHttpResponse:
        _require_success(response)
        _require_identity_encoding(response)
        _require_content_length_bound(response, maximum)
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_raw(chunk_size=64 * 1024):
            _remaining(deadline_monotonic)
            total += len(chunk)
            if total > maximum:
                raise BoundedHttpError(
                    HttpFailureKind.RESPONSE_TOO_LARGE,
                    "HTTP response exceeded its byte ceiling.",
                    status_code=response.status_code,
                )
            chunks.append(chunk)
        return BoundedHttpResponse(
            status_code=response.status_code,
            body=b"".join(chunks),
            content_type=response.headers.get("content-type"),
        )


def validate_loopback_base_url(value: str) -> str:
    try:
        parsed = httpx.URL(value)
    except (TypeError, httpx.InvalidURL) as error:
        raise ValueError("server URL must be a valid unauthenticated HTTP URL") from error
    if parsed.scheme != "http" or not parsed.host or parsed.userinfo:
        raise ValueError("server URL must be an unauthenticated HTTP URL")
    try:
        loopback = ipaddress.ip_address(parsed.host).is_loopback
    except ValueError:
        loopback = parsed.host.lower() == "localhost"
    if not loopback or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("server URL must target a loopback origin without URL extras")
    if parsed.port is not None and not 1 <= parsed.port <= 65_535:
        raise ValueError("server URL port is invalid")
    return str(parsed.copy_with(path="")).rstrip("/")


def _validate_control_path(path: str) -> None:
    if (
        not path.startswith("/")
        or path.startswith("//")
        or "?" in path
        or "#" in path
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/")[1:])
    ):
        raise ValueError("loopback control path must be an absolute normalized path")
    if len(path.encode()) > 1000:
        raise ValueError("loopback control path is too long")


def _validate_download_url(value: str, allowed_origins: tuple[str, ...]) -> None:
    try:
        url = httpx.URL(value)
    except (TypeError, httpx.InvalidURL) as error:
        raise ValueError("managed download URL is invalid") from error
    if (
        url.scheme != "https"
        or not url.host
        or url.userinfo
        or url.fragment
        or _origin(url) not in {_origin(httpx.URL(origin)) for origin in allowed_origins}
    ):
        raise ValueError("managed download URL escaped its allowlisted HTTPS origins")


def _origin(url: httpx.URL) -> str:
    default = 443 if url.scheme == "https" else 80
    port = url.port or default
    suffix = "" if port == default else f":{port}"
    return f"{url.scheme}://{url.host}{suffix}"


def _timeout(deadline_monotonic: float) -> httpx.Timeout:
    remaining = _remaining(deadline_monotonic)
    stage = max(0.001, min(remaining, 10.0))
    return httpx.Timeout(stage, connect=stage, read=stage, write=stage, pool=stage)


def _remaining(deadline_monotonic: float) -> float:
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise BoundedHttpError(HttpFailureKind.TIMEOUT, "HTTP operation deadline expired.")
    return remaining


def _require_success(response: httpx.Response) -> None:
    if response.status_code in _REDIRECT_STATUSES:
        raise BoundedHttpError(
            HttpFailureKind.REDIRECT,
            "HTTP response attempted a disallowed redirect.",
            status_code=response.status_code,
        )
    if not 200 <= response.status_code < 300:
        raise BoundedHttpError(
            HttpFailureKind.STATUS,
            "HTTP response returned a non-success status.",
            status_code=response.status_code,
        )


def _require_identity_encoding(response: httpx.Response) -> None:
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity"}:
        raise BoundedHttpError(
            HttpFailureKind.CONTENT_ENCODING,
            "HTTP response used a disallowed content encoding.",
            status_code=response.status_code,
        )


def _require_content_length_bound(response: httpx.Response, maximum: int) -> None:
    length = _content_length(response)
    if length is not None and length > maximum:
        raise BoundedHttpError(
            HttpFailureKind.RESPONSE_TOO_LARGE,
            "HTTP response exceeded its byte ceiling.",
            status_code=response.status_code,
        )


def _content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        length = int(raw)
    except ValueError as error:
        raise BoundedHttpError(
            HttpFailureKind.POLICY,
            "HTTP response declared an invalid content length.",
            status_code=response.status_code,
        ) from error
    if length < 0:
        raise BoundedHttpError(
            HttpFailureKind.POLICY,
            "HTTP response declared an invalid content length.",
            status_code=response.status_code,
        )
    return length


def _resume_response(
    response: httpx.Response,
    *,
    requested_offset: int,
    maximum: int,
) -> tuple[int, bool, int | None]:
    _require_content_length_bound(response, maximum)
    if requested_offset == 0:
        if response.status_code != 200:
            raise BoundedHttpError(
                HttpFailureKind.POLICY,
                "Managed download returned partial content without a range request.",
                status_code=response.status_code,
            )
        return 0, False, _content_length(response)
    if response.status_code == 200:
        return 0, False, _content_length(response)
    if response.status_code != 206:
        raise BoundedHttpError(
            HttpFailureKind.STATUS,
            "Managed download resume returned an unsupported status.",
            status_code=response.status_code,
        )
    match = _CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
    if match is None:
        raise BoundedHttpError(
            HttpFailureKind.POLICY,
            "Managed download resume omitted a valid Content-Range.",
            status_code=response.status_code,
        )
    start, end, total = (int(value) for value in match.groups())
    declared_length = end - start + 1
    content_length = _content_length(response)
    if (
        start != requested_offset
        or end < start
        or total != maximum
        or end != total - 1
        or (content_length is not None and content_length != declared_length)
    ):
        raise BoundedHttpError(
            HttpFailureKind.POLICY,
            "Managed download resume returned a contradictory Content-Range.",
            status_code=response.status_code,
        )
    return requested_offset, True, declared_length


def _strong_validator(response: httpx.Response) -> str | None:
    etag: str | None = response.headers.get("etag")
    if (
        etag is None
        or etag.startswith("W/")
        or len(etag) > 500
        or len(etag) < 2
        or etag[0] != '"'
        or etag[-1] != '"'
        or any(character == '"' or ord(character) < 0x21 for character in etag[1:-1])
    ):
        return None
    return etag


def _require_bounded_json(value: Any) -> None:
    count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if count > _MAX_JSON_VALUES or depth > _MAX_JSON_DEPTH:
            raise BoundedHttpError(
                HttpFailureKind.MALFORMED_JSON,
                "HTTP JSON response exceeded its structural limits.",
            )
        if isinstance(item, str) and len(item.encode()) > _MAX_JSON_STRING_BYTES:
            raise BoundedHttpError(
                HttpFailureKind.MALFORMED_JSON,
                "HTTP JSON response contained an oversized string.",
            )
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise BoundedHttpError(
                        HttpFailureKind.MALFORMED_JSON,
                        "HTTP JSON response contained a non-string object key.",
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)
