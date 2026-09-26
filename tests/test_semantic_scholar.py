"""Semantic Scholar evidence search with a mocked HTTP client. No live network."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from claimforge.http_cache import CachedResponse, HttpRequestError
from claimforge.models import EvidenceSource
from claimforge.semantic_scholar import (
    SemanticScholarError,
    build_semantic_scholar_search_url,
    search_semantic_scholar,
)

PAPER = {
    "paperId": "abc123",
    "title": "Physics-informed neural networks",
    "abstract": "Networks reduce error on the Burgers equation.",
    "url": "https://www.semanticscholar.org/paper/abc123",
    "externalIds": {"DOI": "https://doi.org/10.1000/pinn", "ArXiv": "1706.03762v3"},
}


class FakeClient:
    def __init__(self, response: CachedResponse | Exception) -> None:
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
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _cached(status: int, payload: object) -> CachedResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return CachedResponse(
        url="https://api.semanticscholar.org/graph/v1/paper/search",
        status_code=status,
        headers={},
        body=body,
        from_cache=False,
    )


def test_search_url_has_no_api_key() -> None:
    url = build_semantic_scholar_search_url("Burgers equation", limit=4)
    assert url.startswith("https://api.semanticscholar.org/graph/v1/paper/search?")
    assert "Burgers" in url or "Burgers".lower() in url.lower()
    assert "limit=4" in url
    assert "api-key" not in url.lower()
    assert "paperId" in url


def test_search_parses_papers_from_mocked_http() -> None:
    payload = {"data": [PAPER, {"title": "missing id"}, "not-an-object", {"paperId": "x"}]}
    client = FakeClient(_cached(200, payload))

    evidence = search_semantic_scholar(client, "Burgers equation", limit=4)

    assert len(evidence) == 1
    item = evidence[0]
    assert item.source is EvidenceSource.semantic_scholar
    assert item.work_id == "abc123"
    assert item.doi == "10.1000/pinn"
    assert item.arxiv_id == "1706.03762"
    assert item.snippet.startswith("Networks reduce")
    assert item.url == "https://www.semanticscholar.org/paper/abc123"
    assert item.score is None
    assert client.headers[0] == {}
    assert "abc123-secret" not in client.urls[0]


def test_optional_api_key_is_a_header_not_part_of_the_url() -> None:
    client = FakeClient(_cached(200, {"data": []}))
    search_semantic_scholar(
        client,
        "Burgers equation",
        limit=1,
        api_key="secret-token",
    )
    assert client.headers[0]["x-api-key"] == "secret-token"
    assert "secret-token" not in client.urls[0]


@pytest.mark.parametrize("status", [401, 403, 429])
def test_soft_fail_statuses_return_no_evidence(status: int) -> None:
    client = FakeClient(_cached(status, {"message": "nope"}))
    assert search_semantic_scholar(client, "Burgers equation", limit=2) == []


def test_retry_exhausted_429_is_a_soft_fail() -> None:
    client = FakeClient(
        HttpRequestError("limited", status_code=429, url="https://api.semanticscholar.org")
    )
    assert search_semantic_scholar(client, "Burgers equation", limit=2) == []


def test_retry_exhausted_500_still_raises() -> None:
    client = FakeClient(
        HttpRequestError("down", status_code=500, url="https://api.semanticscholar.org")
    )
    with pytest.raises(HttpRequestError):
        search_semantic_scholar(client, "Burgers equation", limit=1)


def test_other_http_errors_and_bad_payloads_raise() -> None:
    with pytest.raises(SemanticScholarError, match="HTTP 404"):
        search_semantic_scholar(FakeClient(_cached(404, b"missing")), "Burgers", limit=1)
    with pytest.raises(SemanticScholarError, match="invalid JSON"):
        search_semantic_scholar(FakeClient(_cached(200, b"not-json")), "Burgers", limit=1)
    with pytest.raises(SemanticScholarError, match="missing data"):
        search_semantic_scholar(FakeClient(_cached(200, {"total": 0})), "Burgers", limit=1)
