from __future__ import annotations

import pytest

from flameox.adapters.toxiproxy import ToxiproxyClient


def test_toxiproxy_client_shapes_proxy_and_toxic_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        self: ToxiproxyClient,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        calls.append((method, path, payload))
        return {"name": "proxy"}

    monkeypatch.setattr(ToxiproxyClient, "_request", request)
    client = ToxiproxyClient()
    client.create_proxy(name="proxy", listen="127.0.0.1:10001", upstream="127.0.0.1:10002")
    client.update_proxy("proxy", enabled=False)
    client.add_toxic(
        proxy="proxy",
        name="latency",
        toxic_type="latency",
        stream="downstream",
        toxicity=1.0,
        attributes={"latency": 25, "jitter": 0},
    )

    assert calls == [
        (
            "POST",
            "/proxies",
            {
                "name": "proxy",
                "listen": "127.0.0.1:10001",
                "upstream": "127.0.0.1:10002",
                "enabled": True,
            },
        ),
        ("PUT", "/proxies/proxy", {"enabled": False}),
        (
            "POST",
            "/proxies/proxy/toxics",
            {
                "name": "latency",
                "type": "latency",
                "stream": "downstream",
                "toxicity": 1.0,
                "attributes": {"latency": 25, "jitter": 0},
            },
        ),
    ]


@pytest.mark.parametrize(
    ("toxic_type", "attributes"),
    [
        ("latency", {"latency": 1, "jitter": 0}),
        ("timeout", {"timeout": 1}),
        ("reset_peer", {}),
        ("bandwidth", {"rate": 1}),
        ("slicer", {"average_size": 1, "size_variation": 0, "delay": 0}),
        ("limit_data", {"bytes": 1}),
        ("slow_close", {"delay": 1}),
    ],
)
def test_toxiproxy_client_accepts_all_declared_toxic_shapes(
    monkeypatch: pytest.MonkeyPatch,
    toxic_type: str,
    attributes: dict[str, int],
) -> None:
    monkeypatch.setattr(
        ToxiproxyClient,
        "_request",
        lambda self, method, path, payload=None: {},
    )
    ToxiproxyClient().add_toxic(
        proxy="proxy",
        name="toxic",
        toxic_type=toxic_type,
        attributes=attributes,
    )


@pytest.mark.parametrize("endpoint", ["0.0.0.0:1", "example.test:1", "127.0.0.1:0"])
def test_toxiproxy_client_rejects_non_loopback_or_invalid_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError):
        ToxiproxyClient().create_proxy(
            name="proxy",
            listen=endpoint,
            upstream="127.0.0.1:10002",
        )


def test_toxiproxy_client_rejects_unknown_toxic() -> None:
    with pytest.raises(ValueError):
        ToxiproxyClient().add_toxic(proxy="proxy", name="toxic", toxic_type="unknown")
