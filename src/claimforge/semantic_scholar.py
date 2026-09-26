"""Semantic Scholar Graph API paper search.

Unauthenticated calls are the default. ``CLAIMFORGE_S2_API_KEY`` is optional
and is never required. HTTP 401 and 403 are a ``rejected`` soft failure.
HTTP 429 is a ``rate_limited`` soft failure. Both return an empty list so
one catalog cannot fail CI or drop the other sources. Other HTTP errors and
network errors still propagate. The retriever classifies those as
``unavailable`` or ``invalid_response`` and keeps the catalogs that answered.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlencode

from claimforge.http_cache import CachedResponse, ClaimForgeError, HttpRequestError
from claimforge.literature import (
    clip_text,
    make_evidence_id,
    normalize_arxiv_id,
    normalize_doi,
)
from claimforge.models import Evidence, EvidenceSource

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,url,externalIds"
_SOFT_STATUS = frozenset({401, 403, 429})


class SemanticScholarError(ClaimForgeError):
    """Semantic Scholar returned a response the evidence client cannot use."""


class SupportsGet(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> CachedResponse:
        """Fetch one absolute URL."""


def build_semantic_scholar_search_url(text: str, *, limit: int) -> str:
    """Build an unauthenticated paper-search URL. The key is not placed here."""

    if limit < 1:
        raise ValueError("limit must be >= 1")
    query = clip_text(text, limit=300)
    if not query:
        raise ValueError("claim text has no searchable terms")
    params = {"query": query, "limit": str(limit), "fields": _FIELDS}
    return f"{SEARCH_ENDPOINT}?{urlencode(params, safe=',')}"


def search_semantic_scholar(
    client: SupportsGet,
    text: str,
    *,
    limit: int,
    api_key: str | None = None,
) -> list[Evidence]:
    """Search Semantic Scholar.

    A missing key sends no credential header. 401 and 403 (``rejected``) and
    429 (``rate_limited``) return an empty list. Other HTTP failures still
    raise so the caller can record them.
    """

    url = build_semantic_scholar_search_url(text, limit=limit)
    headers: dict[str, str] = {}
    key = api_key.strip() if isinstance(api_key, str) else ""
    if key:
        headers["x-api-key"] = key
    try:
        response = client.get(url, headers=headers or None)
    except HttpRequestError as exc:
        if exc.status_code in _SOFT_STATUS:
            logger.warning(
                "Semantic Scholar search soft-fail (%s): HTTP %s",
                _soft_kind(exc.status_code),
                exc.status_code,
            )
            return []
        raise
    if response.status_code in _SOFT_STATUS:
        logger.warning(
            "Semantic Scholar search soft-fail (%s): HTTP %s",
            _soft_kind(response.status_code),
            response.status_code,
        )
        return []
    if response.status_code != 200:
        snippet = response.body[:200].decode("utf-8", errors="replace")
        raise SemanticScholarError(
            f"Semantic Scholar search returned HTTP {response.status_code}: {snippet}"
        )
    return parse_semantic_scholar_payload(response.body)


def _soft_kind(status: int) -> str:
    """``rate_limited`` for 429. ``rejected`` for 401 and 403."""

    if status == 429:
        return "rate_limited"
    return "rejected"


def parse_semantic_scholar_payload(body: bytes) -> list[Evidence]:
    """Parse a Graph API ``/paper/search`` body."""

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SemanticScholarError("Semantic Scholar search returned invalid JSON") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise SemanticScholarError("Semantic Scholar search response is missing data")
    evidence: list[Evidence] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        parsed = _paper_evidence(item)
        if parsed is not None:
            evidence.append(parsed)
    return evidence


def _paper_evidence(item: Mapping[str, object]) -> Evidence | None:
    paper_id = str(item.get("paperId") or "").strip()
    if not paper_id:
        return None
    title = str(item.get("title") or "").strip()
    abstract = item.get("abstract")
    snippet = abstract.strip() if isinstance(abstract, str) else ""
    if not title and not snippet:
        return None
    external = item.get("externalIds")
    doi = None
    arxiv_id = None
    if isinstance(external, Mapping):
        doi = normalize_doi(external.get("DOI"))
        arxiv_id = normalize_arxiv_id(external.get("ArXiv"))
    raw_url = item.get("url")
    url = raw_url.strip() if isinstance(raw_url, str) and raw_url.strip() else ""
    if not url:
        url = f"https://www.semanticscholar.org/paper/{paper_id}"
    return Evidence(
        id=make_evidence_id("semantic_scholar", paper_id),
        title=title,
        snippet=snippet,
        source=EvidenceSource.semantic_scholar,
        work_id=paper_id,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
    )
