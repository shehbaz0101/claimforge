"""Multi-source evidence retrieval.

OpenAlex is the primary catalog. arXiv and Semantic Scholar add papers that
OpenAlex may miss. Every GET uses the shared disk cache. Semantic Scholar
401/429 responses are empty contributions, not failures of the whole pack.

Returned scores are lexical overlap. Day 4 replaces that with embeddings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

from claimforge.arxiv import search_arxiv
from claimforge.extract import abstract_text_from_work
from claimforge.http_cache import CachedResponse, ClaimForgeError
from claimforge.literature import (
    clip_text,
    lexical_score,
    make_evidence_id,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)
from claimforge.models import Claim, Evidence, EvidenceSource
from claimforge.openalex import EVIDENCE_SELECT, search_work_records
from claimforge.semantic_scholar import search_semantic_scholar

logger = logging.getLogger(__name__)

DEFAULT_PER_SOURCE = 8
DEFAULT_TOP_K = 8
_SOURCE_RANK = {
    EvidenceSource.openalex: 0,
    EvidenceSource.semantic_scholar: 1,
    EvidenceSource.arxiv: 2,
}
_TITLE_KEY_MIN = 12


class EvidenceRetrievalError(ClaimForgeError):
    """Every source failed before it could return a pack."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


class SupportsCachedGet(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> CachedResponse:
        """Fetch one absolute URL."""


def retrieve_evidence(
    client: SupportsCachedGet,
    claim: Claim | str,
    *,
    per_source: int = DEFAULT_PER_SOURCE,
    top_k: int = DEFAULT_TOP_K,
    mailto: str | None = None,
    s2_api_key: str | None = None,
) -> list[Evidence]:
    """Gather, deduplicate, and return the top evidence for a claim.

    ``claim`` may be a ``Claim`` or raw claim text. OpenAlex receives the
    optional polite-pool ``mailto``. ``s2_api_key`` is optional; when it is
    missing the Semantic Scholar call stays unauthenticated.
    """

    text = claim.text if isinstance(claim, Claim) else claim
    query = " ".join(text.split())
    if not query:
        raise ValueError("claim text must not be empty")
    if per_source < 1:
        raise ValueError("per_source must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    packs: list[list[Evidence]] = []
    failures: list[str] = []
    openalex_query = clip_text(query, limit=250)

    try:
        records = search_work_records(
            client,
            openalex_query,
            per_page=per_source,
            mailto=mailto,
            select=EVIDENCE_SELECT,
        )
    except ClaimForgeError as exc:
        logger.warning("OpenAlex evidence search failed: %s", exc)
        failures.append(f"openalex: {exc}")
    else:
        packs.append(
            [
                item
                for item in (evidence_from_openalex_record(record) for record in records)
                if item is not None
            ]
        )

    try:
        packs.append(search_arxiv(client, query, max_results=per_source))
    except ClaimForgeError as exc:
        logger.warning("arXiv evidence search failed: %s", exc)
        failures.append(f"arxiv: {exc}")

    try:
        packs.append(
            search_semantic_scholar(
                client,
                query,
                limit=per_source,
                api_key=s2_api_key,
            )
        )
    except ClaimForgeError as exc:
        logger.warning("Semantic Scholar evidence search failed: %s", exc)
        failures.append(f"semantic_scholar: {exc}")

    if len(failures) == 3:
        raise EvidenceRetrievalError(failures)
    return merge_evidence(packs, query=query, top_k=top_k)


def evidence_from_openalex_record(record: Mapping[str, object]) -> Evidence | None:
    """Map one OpenAlex work dict onto ``Evidence``."""

    work_id = str(record.get("id") or "").strip()
    if not work_id:
        return None
    title = str(record.get("display_name") or "").strip()
    snippet = abstract_text_from_work(record)
    if not title and not snippet:
        return None
    doi = _openalex_doi(record)
    arxiv_id = _openalex_arxiv_id(record)
    if work_id.startswith(("http://", "https://")):
        url = work_id
    else:
        url = f"https://openalex.org/{work_id}"
    return Evidence(
        id=make_evidence_id("openalex", work_id),
        title=title,
        snippet=snippet,
        source=EvidenceSource.openalex,
        work_id=work_id,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
    )


def merge_evidence(
    groups: Sequence[Sequence[Evidence]],
    *,
    query: str,
    top_k: int,
) -> list[Evidence]:
    """Deduplicate packs and return the highest-scoring ``top_k`` records.

    The same DOI or arXiv id collapses to one record. Identifiers are copied
    across sources. A shared title merges only when one copy has no DOI,
    arXiv id, or work id, so two different catalog records are kept apart.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    items = [item for group in groups for item in group]
    if not items:
        return []

    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    key_owner: dict[str, int] = {}
    for index, item in enumerate(items):
        for key in _strong_keys(item):
            owner = key_owner.get(key)
            if owner is None:
                key_owner[key] = index
            else:
                union(owner, index)

    title_owners: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        title_key = normalize_title(item.title)
        if len(title_key) < _TITLE_KEY_MIN:
            continue
        owners = title_owners.setdefault(title_key, [])
        joined = False
        for owner in owners:
            if find(owner) == find(index):
                joined = True
                break
            if _title_merge_allowed(items, find, owner, index):
                union(owner, index)
                joined = True
                break
        if not joined:
            owners.append(index)

    clusters: dict[int, list[Evidence]] = {}
    for index, item in enumerate(items):
        clusters.setdefault(find(index), []).append(item)

    merged = [_merge_cluster(cluster, query) for cluster in clusters.values()]
    merged.sort(
        key=lambda item: (
            -(item.score or 0.0),
            _SOURCE_RANK[item.source],
            item.title.casefold(),
            item.id,
        )
    )
    return merged[:top_k]


def _openalex_doi(record: Mapping[str, object]) -> str | None:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return doi
    ids = record.get("ids")
    if isinstance(ids, Mapping):
        return normalize_doi(ids.get("doi"))
    return None


def _openalex_arxiv_id(record: Mapping[str, object]) -> str | None:
    ids = record.get("ids")
    if not isinstance(ids, Mapping):
        return None
    return normalize_arxiv_id(ids.get("arxiv"))


def _strong_keys(item: Evidence) -> list[str]:
    keys: list[str] = []
    if item.doi:
        keys.append("doi:" + item.doi.casefold())
    if item.arxiv_id:
        keys.append("arxiv:" + item.arxiv_id.casefold())
    if item.work_id and item.source is EvidenceSource.openalex:
        keys.append("openalex:" + item.work_id)
    if item.work_id and item.source is EvidenceSource.semantic_scholar:
        keys.append("s2:" + item.work_id)
    return keys


def _title_merge_allowed(
    items: Sequence[Evidence],
    find,
    left: int,
    right: int,
) -> bool:
    """Title-match only when one cluster has no strong identifier."""

    left_members = [item for index, item in enumerate(items) if find(index) == find(left)]
    right_members = [item for index, item in enumerate(items) if find(index) == find(right)]
    return not (_has_strong_id(left_members) and _has_strong_id(right_members))


def _has_strong_id(members: Sequence[Evidence]) -> bool:
    return any(item.doi or item.arxiv_id or item.work_id for item in members)


def _merge_cluster(members: Sequence[Evidence], query: str) -> Evidence:
    ordered = sorted(members, key=lambda item: _SOURCE_RANK[item.source])
    primary = ordered[0]
    title = primary.title or max((item.title for item in ordered), key=len)
    snippet = max(ordered, key=lambda item: (len(item.snippet), -_SOURCE_RANK[item.source])).snippet
    doi = next((item.doi for item in ordered if item.doi), None)
    arxiv_id = next((item.arxiv_id for item in ordered if item.arxiv_id), None)
    work_id = next((item.work_id for item in ordered if item.work_id), None)
    source = primary.source
    if doi:
        evidence_id = make_evidence_id("doi", doi.casefold())
    elif arxiv_id:
        evidence_id = make_evidence_id("arxiv", arxiv_id.casefold())
    elif work_id:
        evidence_id = make_evidence_id(source.value, work_id)
    else:
        evidence_id = make_evidence_id("title", normalize_title(title) or title.casefold())
    return Evidence(
        id=evidence_id,
        title=title,
        snippet=snippet,
        source=source,
        work_id=work_id,
        doi=doi,
        arxiv_id=arxiv_id,
        url=primary.url,
        score=lexical_score(query, title, snippet),
    )
