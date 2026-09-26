"""Multi-source evidence retrieval.

OpenAlex is the primary catalog. arXiv and Semantic Scholar add papers that
OpenAlex may miss. Every GET uses the shared disk cache. Semantic Scholar
401/403/429 responses are empty contributions, not failures of the whole pack.

A hard failure of one catalog is logged and skipped. The other catalogs are
still returned. The kinds are ``rate_limited`` (HTTP 429), ``rejected``
(HTTP 401 or 403), ``unavailable`` (timeout, connection error, or HTTP 500,
502, 503, 504), ``invalid_response`` (any other error, including an unexpected
exception), and ``offline`` (a cache miss while offline mode is on). An
unexpected exception in one catalog does not abort the others and does not
crash ``verify`` or ``serve``. The command or route fails only when every
catalog fails in online mode.

After ``CLAIMFORGE_SOURCE_FAILURE_LIMIT`` consecutive hard failures for one
catalog in this process (default 3), later retrievals skip that catalog.
That covers the rest of a multi-claim CLI call and later ``/verify`` requests
in the same server process. A success resets the streak. The counter is not
stored on disk. ``0`` disables the skip.

Offline mode (``CLAIMFORGE_OFFLINE=1`` or ``--offline``) does not use the
network. A matching file under ``data/fixtures/cassettes`` (or
``CLAIMFORGE_FIXTURE_DIR``) is the pack. Otherwise only a warm disk cache is
read. If neither has the claim, the result is an empty pack, which the judge
records as insufficient.

After dedupe, the pack is re-scored by ``claimforge.rank``. ``auto`` uses a
local embedding model when sentence-transformers is installed and TF-IDF
cosine otherwise. The score on each record is that cosine, in ``[0, 1]``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from claimforge.arxiv import search_arxiv
from claimforge.cassettes import load_matching_cassette
from claimforge.circuit import SourceCircuit, process_source_circuit
from claimforge.extract import abstract_text_from_work
from claimforge.http_cache import (
    CachedResponse,
    ClaimForgeError,
    HttpRequestError,
    OfflineCacheMiss,
    TransientNetworkError,
)
from claimforge.literature import (
    clip_text,
    make_evidence_id,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)
from claimforge.models import Claim, Evidence, EvidenceSource
from claimforge.openalex import EVIDENCE_SELECT, search_work_records
from claimforge.rank import TextRanker, pack_top_k, resolve_ranker
from claimforge.semantic_scholar import search_semantic_scholar
from claimforge.settings import offline_enabled

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


def classify_provider_failure(exc: BaseException) -> str:
    """Name the soft-fail kind for a catalog error.

    ``rate_limited`` is HTTP 429. ``rejected`` is HTTP 401 or 403.
    ``unavailable`` is a timeout, a connection error, or HTTP 500, 502, 503,
    or 504. ``offline`` is a cache miss while offline mode is on.
    Anything else, including an unexpected exception, is ``invalid_response``.
    """

    if isinstance(exc, OfflineCacheMiss):
        return "offline"
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limited"
    if status in {401, 403}:
        return "rejected"
    if isinstance(exc, TransientNetworkError):
        return "unavailable"
    if isinstance(status, int) and status in {500, 502, 503, 504}:
        return "unavailable"
    message = str(exc)
    if "HTTP 429" in message:
        return "rate_limited"
    if "HTTP 401" in message or "HTTP 403" in message:
        return "rejected"
    if isinstance(exc, HttpRequestError):
        return "unavailable"
    return "invalid_response"


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
    ranker: str | TextRanker = "auto",
    embedding_cache_dir: Path | None = None,
    embedding_model: str | None = None,
    circuit: SourceCircuit | None = None,
    offline: bool | None = None,
) -> list[Evidence]:
    """Gather, deduplicate, rank, and return the top evidence for a claim.

    ``claim`` may be a ``Claim`` or raw claim text. OpenAlex receives the
    optional polite-pool ``mailto``. ``s2_api_key`` is optional; when it is
    missing the Semantic Scholar call stays unauthenticated.

    ``ranker`` is ``auto``, ``lexical``, ``embeddings``, or a ranker object.
    ``auto`` prefers a local embedding model when sentence-transformers is
    installed and otherwise scores with TF-IDF cosine. The ranker is resolved
    before any catalog request so a missing extra fails without network I/O.

    ``offline`` forces cassette and cache reads. ``None`` follows
    ``--offline`` and ``CLAIMFORGE_OFFLINE``. A missing pack is an empty list
    in offline mode, not an error. ``circuit`` defaults to the process circuit.
    """

    text = claim.text if isinstance(claim, Claim) else claim
    query = " ".join(text.split())
    if not query:
        raise ValueError("claim text must not be empty")
    if per_source < 1:
        raise ValueError("per_source must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    active = resolve_ranker(
        ranker,
        cache_dir=embedding_cache_dir,
        model_name=embedding_model,
    )
    use_offline = offline_enabled() if offline is None else offline
    if use_offline:
        cassette = load_matching_cassette(claim if isinstance(claim, Claim) else query)
        if cassette is not None:
            logger.info("offline mode: using a cassette pack (%s records)", len(cassette))
            # Keep the ids stored in the file. merge_evidence rewrites them.
            return pack_top_k(cassette, query, top_k=top_k, ranker=active)

    packs: list[list[Evidence]] = []
    failures: list[str] = []
    breaker = circuit if circuit is not None else process_source_circuit()
    openalex_query = clip_text(query, limit=250)

    openalex_pack = _run_source(
        "openalex",
        breaker,
        failures,
        lambda: [
            item
            for item in (
                evidence_from_openalex_record(record)
                for record in search_work_records(
                    client,
                    openalex_query,
                    per_page=per_source,
                    mailto=mailto,
                    select=EVIDENCE_SELECT,
                )
            )
            if item is not None
        ],
    )
    if openalex_pack is not None:
        packs.append(openalex_pack)

    arxiv_pack = _run_source(
        "arxiv",
        breaker,
        failures,
        lambda: search_arxiv(client, query, max_results=per_source),
    )
    if arxiv_pack is not None:
        packs.append(arxiv_pack)

    scholar_pack = _run_source(
        "semantic_scholar",
        breaker,
        failures,
        lambda: search_semantic_scholar(
            client,
            query,
            limit=per_source,
            api_key=s2_api_key,
        ),
    )
    if scholar_pack is not None:
        packs.append(scholar_pack)

    if len(failures) == 3:
        if use_offline:
            logger.info(
                "offline mode: no cassette or cached response; returning an empty pack"
            )
            return []
        raise EvidenceRetrievalError(failures)
    return merge_evidence(packs, query=query, top_k=top_k, ranker=active)


def _run_source(
    name: str,
    circuit: SourceCircuit,
    failures: list[str],
    fetch: Callable[[], list[Evidence]],
) -> list[Evidence] | None:
    """Search one catalog. Return None when that catalog is skipped or failed.

    A failure is recorded and the other catalogs still run. The returned
    error string starts with the source name and the soft-fail kind.
    """

    if not circuit.allow(name):
        message = (
            f"{name}: skipped after {circuit.limit} consecutive failures"
        )
        logger.warning(
            "%s skipped for the rest of this process after %s consecutive failures",
            name,
            circuit.limit,
        )
        failures.append(message)
        return None
    try:
        evidence = fetch()
    except Exception as exc:
        kind = classify_provider_failure(exc)
        if kind != "offline":
            circuit.record_failure(name)
        log = logger.info if kind == "offline" else logger.warning
        log("%s evidence search failed (%s): %s", name, kind, exc)
        failures.append(f"{name}: {kind}: {exc}")
        return None
    circuit.record_success(name)
    return evidence


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
    ranker: TextRanker | None = None,
) -> list[Evidence]:
    """Deduplicate packs, re-score them, and return the top ``top_k`` records.

    The same DOI or arXiv id collapses to one record. Identifiers are copied
    across sources. A shared title merges only when one copy has no DOI,
    arXiv id, or work id, so two different catalog records are kept apart.
    Scores come from ``ranker``, or from TF-IDF cosine when ``ranker`` is
    omitted. Passing ``auto`` is the job of ``retrieve_evidence``.
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

    merged = [_merge_cluster(cluster) for cluster in clusters.values()]
    return pack_top_k(merged, query, top_k=top_k, ranker=ranker)


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


def _merge_cluster(members: Sequence[Evidence]) -> Evidence:
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
    )
