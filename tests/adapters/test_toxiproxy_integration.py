from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
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
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    process = subprocess.Popen(
        [str(binary), "-host", "127.0.0.1", "-port", str(admin_port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not client.health():
        time.sleep(0.05)
    if not client.health():
        process.terminate()
        process.wait(timeout=2)
        upstream.shutdown()
        upstream_thread.join(timeout=2)
        upstream.server_close()
        pytest.fail("Toxiproxy did not become ready")
    try:
        yield client, upstream, process
    finally:
        process.terminate()
        process.wait(timeout=2)
        upstream.shutdown()
        upstream_thread.join(timeout=2)
        upstream.server_close()


@pytest.mark.parametrize(
    ("toxic_type", "attributes", "assertion"),
    [
        ("latency", {"latency": 25, "jitter": 0}, "latency"),
        ("reset_peer", {}, "failure"),
        ("timeout", {"timeout": 100}, "failure"),
        ("bandwidth", {"rate": 128}, "response"),
        ("limit_data", {"bytes": 32}, "truncated"),
    ],
)
def test_real_toxiproxy_faults_against_local_http_upstream(
    toxiproxy_server: tuple[ToxiproxyClient, ThreadingHTTPServer, subprocess.Popen[bytes]],
    toxic_type: str,
    attributes: dict[str, int],
    assertion: str,
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
        baseline_url = f"http://127.0.0.1:{listen_port}/"
        with urlopen(baseline_url, timeout=2) as response:
            baseline = response.read()
        client.add_toxic(
            proxy=name,
            name="treatment",
            toxic_type=toxic_type,
            attributes=attributes,
        )
        started = time.monotonic()
        if assertion == "failure":
            with pytest.raises((URLError, TimeoutError, OSError)):
                urlopen(baseline_url, timeout=1).read()
        else:
            with urlopen(baseline_url, timeout=3) as response:
                treated = response.read()
            if assertion == "latency":
                assert time.monotonic() - started >= 0.02
            elif assertion == "truncated":
                assert len(treated) < len(baseline)
            else:
                assert treated
    finally:
        client.delete_proxy(name)
