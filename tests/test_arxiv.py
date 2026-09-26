"""arXiv evidence search with a mocked HTTP client. No live network."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from claimforge.arxiv import (
    ARXIV_ACCEPT,
    ARXIV_USER_AGENT,
    ArxivError,
    build_arxiv_search_url,
    parse_arxiv_atom,
    search_arxiv,
)
from claimforge.http_cache import CachedHttpClient, CachedResponse, TransportResponse
from claimforge.models import EvidenceSource

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention   Is All You Need</title>
    <summary>The dominant sequence transduction models are recurrent.</summary>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <arxiv:doi>10.1000/attn</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v2</id>
    <title>Old preprint</title>
    <summary>A short abstract.</summary>
  </entry>
  <entry>
    <title>Missing identifier</title>
    <summary>Skipped.</summary>
  </entry>
</feed>
"""


class FakeClient:
    def __init__(self, response: CachedResponse) -> None:
        self.response = response
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> CachedResponse:
        self.urls.append(url)
        self.headers.append(dict(headers or {}))
        return self.response


def _cached(status: int, body: bytes) -> CachedResponse:
    return CachedResponse(
        url="https://export.arxiv.org/api/query",
        status_code=status,
        headers={},
        body=body,
        from_cache=False,
    )


def test_search_url_ands_content_terms() -> None:
    url = build_arxiv_search_url(
        "Physics-informed neural networks reduce the error on the Burgers equation.",
        max_results=3,
    )
    assert url.startswith("https://export.arxiv.org/api/query?")
    assert "search_query=" in url
    assert "all:physics+AND+all:informed" in url
    assert "all:burgers" in url
    assert "max_results=3" in url
    assert "physics-informed" not in url
    assert "all:the+" not in url and not url.endswith("all:the")


def test_search_url_rejects_bad_limits_and_empty_text() -> None:
    with pytest.raises(ValueError):
        build_arxiv_search_url("neural networks", max_results=0)
    with pytest.raises(ValueError):
        build_arxiv_search_url("???", max_results=1)


def test_search_arxiv_parses_atom_from_mocked_http() -> None:
    client = FakeClient(_cached(200, ATOM.encode()))
    evidence = search_arxiv(client, "attention is all you need", max_results=2)

    assert len(evidence) == 2
    first, second = evidence
    assert first.source is EvidenceSource.arxiv
    assert first.arxiv_id == "1706.03762"
    assert first.doi == "10.1000/attn"
    assert first.title == "Attention Is All You Need"
    assert first.snippet == "The dominant sequence transduction models are recurrent."
    assert first.url == "https://arxiv.org/abs/1706.03762"
    assert first.work_id is None
    assert first.score is None
    assert second.arxiv_id == "hep-th/9901001"
    assert client.headers[0]["Accept"] == ARXIV_ACCEPT
    assert client.headers[0]["User-Agent"] == ARXIV_USER_AGENT
    assert "export.arxiv.org" in client.urls[0]


def test_search_arxiv_sends_user_agent_and_atom_accept() -> None:
    """Regression: arXiv returned HTTP 406 with an empty body without these."""

    client = FakeClient(_cached(200, ATOM.encode()))
    search_arxiv(client, "attention is all you need", max_results=1)

    assert client.headers[0]["User-Agent"] == (
        "ClaimForge/0.1 (mailto:github.com/shehbaz0101/claimforge)"
    )
    assert client.headers[0]["Accept"] == "application/atom+xml"


def test_cached_client_forwards_arxiv_polite_headers(tmp_path: Path) -> None:
    class Transport:
        def __init__(self) -> None:
            self.headers: list[dict[str, str]] = []

        def get(
            self,
            url: str,
            headers: Mapping[str, str],
            timeout: float,
        ) -> TransportResponse:
            self.headers.append(dict(headers))
            return TransportResponse(status_code=200, headers={}, body=ATOM.encode())

    transport = Transport()
    client = CachedHttpClient(
        tmp_path,
        transport=transport,
        sleep=lambda _seconds: None,
        max_retries=0,
        jitter=0.0,
    )
    evidence = search_arxiv(client, "attention is all you need", max_results=1)

    assert evidence
    assert transport.headers[0]["User-Agent"] == ARXIV_USER_AGENT
    assert transport.headers[0]["Accept"] == ARXIV_ACCEPT


def test_parse_arxiv_atom_rejects_invalid_xml() -> None:
    with pytest.raises(ArxivError, match="invalid XML"):
        parse_arxiv_atom(b"<feed>")


def test_search_arxiv_rejects_non_success_status() -> None:
    client = FakeClient(_cached(404, b"missing"))
    with pytest.raises(ArxivError, match="HTTP 404"):
        search_arxiv(client, "neural networks", max_results=1)
