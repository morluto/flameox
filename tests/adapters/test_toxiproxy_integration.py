from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

from flameox.adapters.toxiproxy import ToxiproxyClient


class _PayloadHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        payload = b"flameox-local-upstream\n" * 4096
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_raw_http_response(url: str) -> bytes:
    endpoint = urlsplit(url)
    assert endpoint.hostname is not None
    assert endpoint.port is not None
    chunks: list[bytes] = []
    with socket.create_connection((endpoint.hostname, endpoint.port), timeout=3) as connection:
        connection.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
        try:
            while chunk := connection.recv(4096):
                chunks.append(chunk)
        except ConnectionResetError:
            pass
    return b"".join(chunks)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _stop_upstream(upstream: ThreadingHTTPServer, thread: threading.Thread) -> None:
    upstream.shutdown()
    thread.join(timeout=2)
    upstream.server_close()


@pytest.fixture
def toxiproxy_server() -> Generator[
    tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]], None, None
]:
    executable = os.environ.get("FLAMEOX_TOXIPROXY_SERVER")
    if not executable:
        pytest.skip("set FLAMEOX_TOXIPROXY_SERVER to run real Toxiproxy integration tests")
    binary = Path(executable)
    if not binary.is_file():
        pytest.skip("configured Toxiproxy server does not exist")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _PayloadHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    admin_port = _free_port()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [str(binary), "-host", "127.0.0.1", "-port", str(admin_port)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client.health():
            time.sleep(0.05)
        if not client.health():
            pytest.fail("Toxiproxy did not become ready")
        yield client, upstream, process
    finally:
        if process is not None:
            _stop_process(process)
        _stop_upstream(upstream, upstream_thread)


@contextmanager
def _response_fault(
    toxiproxy_server: tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]],
    toxic_type: str,
    attributes: dict[str, int],
) -> Iterator[tuple[str, bytes]]:
    client, upstream, _ = toxiproxy_server
    name = f"integration-{toxic_type}"
    listen_port = _free_port()
    client.create_proxy(
        name=name,
        listen=f"127.0.0.1:{listen_port}",
        upstream=f"127.0.0.1:{upstream.server_port}",
    )
    try:
        url = f"http://127.0.0.1:{listen_port}/"
        with urlopen(url, timeout=2) as response:
            baseline = response.read()
        client.add_toxic(
            proxy=name,
            name="treatment",
            toxic_type=toxic_type,
            attributes=attributes,
        )
        yield url, baseline
    finally:
        client.delete_proxy(name)


def test_real_toxiproxy_latency_delays_the_complete_response(
    toxiproxy_server: tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]],
) -> None:
    with _response_fault(
        toxiproxy_server,
        "latency",
        {"latency": 25, "jitter": 0},
    ) as (url, baseline):
        started = time.monotonic()
        with urlopen(url, timeout=3) as response:
            treated = response.read()
        elapsed = time.monotonic() - started

    assert treated == baseline
    assert elapsed >= 0.02


def test_real_toxiproxy_bandwidth_limits_response_rate(
    toxiproxy_server: tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]],
) -> None:
    with _response_fault(toxiproxy_server, "bandwidth", {"rate": 128}) as (url, baseline):
        started = time.monotonic()
        with urlopen(url, timeout=3) as response:
            treated = response.read()
        elapsed = time.monotonic() - started

    assert treated == baseline
    assert elapsed >= 0.4


def test_real_toxiproxy_limit_data_truncates_response(
    toxiproxy_server: tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]],
) -> None:
    with _response_fault(toxiproxy_server, "limit_data", {"bytes": 32}) as (url, baseline):
        treated = _read_raw_http_response(url)

    assert len(treated) == 32
    assert treated.startswith(b"HTTP/")
    assert len(baseline) > len(treated)


@pytest.mark.parametrize(
    ("toxic_type", "attributes"),
    [
        pytest.param("reset_peer", {}, id="reset-peer"),
        pytest.param("timeout", {"timeout": 100}, id="timeout"),
    ],
)
def test_real_toxiproxy_connection_faults_fail_requests(
    toxiproxy_server: tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]],
    toxic_type: str,
    attributes: dict[str, int],
) -> None:
    client, upstream, _ = toxiproxy_server
    name = f"integration-{toxic_type}"
    listen_port = _free_port()
    client.create_proxy(
        name=name,
        listen=f"127.0.0.1:{listen_port}",
        upstream=f"127.0.0.1:{upstream.server_port}",
    )
    try:
        url = f"http://127.0.0.1:{listen_port}/"
        with urlopen(url, timeout=2) as response:
            assert response.read()
        client.add_toxic(
            proxy=name,
            name="treatment",
            toxic_type=toxic_type,
            attributes=attributes,
        )

        with pytest.raises((URLError, TimeoutError, OSError)):
            urlopen(url, timeout=1).read()
    finally:
        client.delete_proxy(name)
