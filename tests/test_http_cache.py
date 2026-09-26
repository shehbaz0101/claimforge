"""Unit tests for cache keys and retry behavior. No live network except localhost."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from claimforge.http_cache import (
    CachedHttpClient,
    CachedResponse,
    HttpRequestError,
    TransientNetworkError,
    TransportResponse,
    UrllibTransport,
    canonical_url,
    make_cache_key,
    parse_retry_after,
)


class ScriptedTransport:
    def __init__(self, script: list[TransportResponse | Exception]) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        self.calls.append((url, dict(headers)))
        if not self.script:
            raise AssertionError("scripted transport ran out of responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(status: int, body: bytes = b"", **headers: str) -> TransportResponse:
    return TransportResponse(status_code=status, headers=headers, body=body)


def _client(tmp_path: Path, transport: ScriptedTransport, **kwargs: object) -> CachedHttpClient:
    slept: list[float] = []
    kwargs.setdefault("sleep", slept.append)
    kwargs.setdefault("jitter", 0.0)
    kwargs.setdefault("backoff_base", 0.5)
    kwargs.setdefault("backoff_max", 30.0)
    client = CachedHttpClient(tmp_path, transport=transport, **kwargs)  # type: ignore[arg-type]
    client.slept = slept  # type: ignore[attr-defined]
    return client


def test_cache_key_is_stable_sha256_of_canonical_request() -> None:
    url = "https://API.OpenAlex.org/works?b=2&a=1#section"
    canonical = "https://api.openalex.org/works?a=1&b=2"
    assert canonical_url(url) == canonical
    expected = hashlib.sha256(f"GET {canonical}".encode()).hexdigest()
    assert make_cache_key(url) == expected
    assert make_cache_key("https://api.openalex.org/works?a=1&b=2", method="get") == expected
    assert make_cache_key("https://api.openalex.org/works?a=2&b=2") != expected


def test_cache_key_decodes_query_before_sorting() -> None:
    spaced = canonical_url("https://example.com/search?q=physics+informed")
    encoded = canonical_url("https://example.com/search?q=physics%20informed")
    assert spaced == encoded
    assert make_cache_key(spaced) == make_cache_key(encoded)


def test_parse_retry_after_seconds_and_http_date() -> None:
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    assert parse_retry_after("3", now) == 3.0
    assert parse_retry_after("0", now) == 0.0
    assert parse_retry_after("-4", now) == 0.0
    assert parse_retry_after("Fri, 25 Sep 2026 00:00:10 GMT", now) == 10.0
    assert parse_retry_after("not-a-date", now) is None
    assert parse_retry_after("nan", now) is None
    assert parse_retry_after("inf", now) is None


def test_successful_get_is_cached_and_reused(tmp_path: Path) -> None:
    transport = ScriptedTransport([_response(200, b'{"ok": true}', **{"Content-Type": "application/json"})])
    client = _client(tmp_path, transport)

    first = client.get("https://example.com/works?b=1&a=2")
    second = client.get("https://example.com/works?a=2&b=1")

    assert first.from_cache is False
    assert second.from_cache is True
    assert second.body == b'{"ok": true}'
    assert second.json() == {"ok": True}
    assert len(transport.calls) == 1
    assert transport.calls[0][1]["User-Agent"].startswith("ClaimForge/")
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    stored = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert stored["v"] == 1
    assert stored["url"] == "https://example.com/works?a=2&b=1"


def test_retry_after_replaces_exponential_backoff(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            _response(429, b"slow", **{"Retry-After": "2.5"}),
            _response(200, b"ok"),
        ]
    )
    client = _client(tmp_path, transport, backoff_base=10.0, max_retries=2)

    response = client.get("https://example.com/item")

    assert response.status_code == 200
    assert response.body == b"ok"
    assert client.slept == [2.5]  # type: ignore[attr-defined]
    assert len(transport.calls) == 2


def test_http_date_retry_after_uses_injected_clock(tmp_path: Path) -> None:
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    transport = ScriptedTransport(
        [
            _response(503, b"later", **{"Retry-After": "Fri, 25 Sep 2026 00:00:12 GMT"}),
            _response(200, b"ok"),
        ]
    )
    client = _client(tmp_path, transport, now=lambda: now, max_retries=1)

    assert client.get("https://example.com/item").status_code == 200
    assert client.slept == [12.0]  # type: ignore[attr-defined]


def test_invalid_retry_after_falls_back_to_backoff(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            _response(502, b"bad", **{"Retry-After": "soon"}),
            _response(500, b"still"),
            _response(200, b"ok"),
        ]
    )
    client = _client(tmp_path, transport, backoff_base=0.5, max_retries=3)

    assert client.get("https://example.com/item").body == b"ok"
    assert client.slept == [0.5, 1.0]  # type: ignore[attr-defined]


def test_retry_budget_exhausted_raises(tmp_path: Path) -> None:
    transport = ScriptedTransport([_response(429, b"nope", **{"Retry-After": "1"})] * 3)
    client = _client(tmp_path, transport, max_retries=2)

    with pytest.raises(HttpRequestError) as caught:
        client.get("https://example.com/item")

    assert caught.value.status_code == 429
    assert client.slept == [1.0, 1.0]  # type: ignore[attr-defined]
    assert list(tmp_path.glob("*.json")) == []


def test_network_errors_retry_then_succeed(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            TransientNetworkError("timed out"),
            _response(200, b"back"),
        ]
    )
    client = _client(tmp_path, transport, max_retries=1, backoff_base=0.25)

    response = client.get("https://example.com/item")

    assert response.body == b"back"
    assert response.from_cache is False
    assert client.slept == [0.25]  # type: ignore[attr-defined]


def test_network_errors_raise_when_retries_run_out(tmp_path: Path) -> None:
    transport = ScriptedTransport([TransientNetworkError("reset")] * 2)
    client = _client(tmp_path, transport, max_retries=1)

    with pytest.raises(TransientNetworkError, match="reset"):
        client.get("https://example.com/item")
    assert len(client.slept) == 1  # type: ignore[attr-defined]


def test_non_retryable_status_is_returned_and_not_cached(tmp_path: Path) -> None:
    transport = ScriptedTransport([_response(404, b"missing")])
    client = _client(tmp_path, transport)

    response = client.get("https://example.com/missing")

    assert response.status_code == 404
    assert response.from_cache is False
    assert client.slept == []  # type: ignore[attr-defined]
    assert list(tmp_path.glob("*.json")) == []


def test_use_cache_false_skips_read_and_write(tmp_path: Path) -> None:
    transport = ScriptedTransport([_response(200, b"one"), _response(200, b"two")])
    client = _client(tmp_path, transport)

    assert client.get("https://example.com/item", use_cache=False).body == b"one"
    assert client.get("https://example.com/item", use_cache=False).body == b"two"
    assert len(transport.calls) == 2
    assert list(tmp_path.glob("*.json")) == []


def test_corrupt_cache_is_ignored(tmp_path: Path) -> None:
    transport = ScriptedTransport([_response(200, b"fresh")])
    client = _client(tmp_path, transport)
    key = make_cache_key("https://example.com/item")
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")

    response = client.get("https://example.com/item")

    assert response.body == b"fresh"
    assert response.from_cache is False
    assert len(transport.calls) == 1


def test_retry_after_is_capped(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            _response(429, b"later", **{"Retry-After": "999"}),
            _response(200, b"ok"),
        ]
    )
    client = _client(tmp_path, transport, max_retry_after=5, max_retries=1)

    assert client.get("https://example.com/item").status_code == 200
    assert client.slept == [5.0]  # type: ignore[attr-defined]


def test_backoff_respects_max(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            _response(503, b"a"),
            _response(503, b"b"),
            _response(200, b"ok"),
        ]
    )
    client = _client(
        tmp_path,
        transport,
        backoff_base=8,
        backoff_max=10,
        max_retries=2,
    )

    assert client.get("https://example.com/item").body == b"ok"
    assert client.slept == [8.0, 10.0]  # type: ignore[attr-defined]


def test_urllib_transport_maps_status_and_body() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            if self.path.startswith("/ok"):
                body = "héllo".encode()
                self.send_response(200)
            else:
                body = b"slow down"
                self.send_response(429)
                self.send_header("Retry-After", "7")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        transport = UrllibTransport()
        ok = transport.get(f"http://127.0.0.1:{port}/ok", {}, 5)
        limited = transport.get(f"http://127.0.0.1:{port}/limited", {}, 5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert ok.status_code == 200
    assert ok.body == "héllo".encode()
    assert limited.status_code == 429
    assert limited.body == b"slow down"
    assert any(key.lower() == "retry-after" and value == "7" for key, value in limited.headers.items())


def test_urllib_transport_closed_port_is_transient() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    transport = UrllibTransport()
    with pytest.raises(TransientNetworkError):
        transport.get(f"http://127.0.0.1:{port}/", {}, 2)


def test_cached_response_text() -> None:
    response = CachedResponse(
        url="https://example.com/",
        status_code=200,
        headers={},
        body="π".encode(),
        from_cache=False,
    )
    assert response.text() == "π"
