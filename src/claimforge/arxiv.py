"""arXiv Atom API search. No API key.

The export endpoint returns Atom XML. Requests go through
``CachedHttpClient`` so the body is cached like any other GET.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import quote

from claimforge.http_cache import CachedResponse, ClaimForgeError
from claimforge.literature import make_evidence_id, normalize_arxiv_id, normalize_doi, search_terms
from claimforge.models import Evidence, EvidenceSource

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_ACCEPT = "application/atom+xml, application/xml;q=0.9, */*;q=0.8"


class ArxivError(ClaimForgeError):
    """arXiv returned a response the evidence client cannot use."""


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


def build_arxiv_search_url(text: str, *, max_results: int) -> str:
    """Build an arXiv search URL from claim text.

    The Atom API has no key. Hyphenated compounds are split, then terms are
    joined with AND so a short claim does not match every paper that shares
    one common word.
    """

    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    terms = _arxiv_terms(text)
    if not terms:
        raise ValueError("claim text has no searchable terms")
    expression = "+AND+".join(f"all:{quote(term, safe='')}" for term in terms)
    return (
        f"{ARXIV_ENDPOINT}?search_query={expression}"
        f"&start=0&max_results={max_results}"
    )


def _arxiv_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in search_terms(text, limit=8):
        parts = [part for part in term.split("-") if len(part) >= 3] or [term]
        for part in parts:
            if part in seen:
                continue
            seen.add(part)
            terms.append(part)
            if len(terms) >= 8:
                return terms
    return terms


def search_arxiv(
    client: SupportsGet,
    text: str,
    *,
    max_results: int,
) -> list[Evidence]:
    """Search arXiv and return evidence. The caller caches ``client``."""

    try:
        url = build_arxiv_search_url(text, max_results=max_results)
    except ValueError as exc:
        raise ArxivError(str(exc)) from exc
    response = client.get(url, headers={"Accept": _ACCEPT})
    if response.status_code != 200:
        snippet = response.body[:200].decode("utf-8", errors="replace")
        raise ArxivError(f"arXiv search returned HTTP {response.status_code}: {snippet}")
    return parse_arxiv_atom(response.body)


def parse_arxiv_atom(body: bytes) -> list[Evidence]:
    """Parse an arXiv Atom feed into evidence records."""

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ArxivError("arXiv search returned invalid XML") from exc
    evidence: list[Evidence] = []
    for entry in root.findall(f"{_ATOM}entry"):
        item = _entry_evidence(entry)
        if item is not None:
            evidence.append(item)
    return evidence


def _entry_evidence(entry: ET.Element) -> Evidence | None:
    raw_id = _text(entry, f"{_ATOM}id")
    arxiv_id = normalize_arxiv_id(raw_id)
    if arxiv_id is None:
        return None
    title = _text(entry, f"{_ATOM}title")
    summary = _text(entry, f"{_ATOM}summary")
    if not title and not summary:
        return None
    doi = normalize_doi(_text(entry, f"{_ARXIV_NS}doi"))
    return Evidence(
        id=make_evidence_id("arxiv", arxiv_id),
        title=title,
        snippet=summary,
        source=EvidenceSource.arxiv,
        doi=doi,
        arxiv_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _text(entry: ET.Element, tag: str) -> str:
    element = entry.find(tag)
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())
