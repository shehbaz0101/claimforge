"""Small OpenAlex works search used by the smoke command and claim extraction.

OpenAlex does not require an API key. An optional mailto query parameter joins
the polite pool when ``CLAIMFORGE_OPENALEX_MAILTO`` is set in the environment.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

from claimforge.http_cache import CachedResponse, ClaimForgeError

WORKS_ENDPOINT = "https://api.openalex.org/works"
DEFAULT_QUERY = "physics informed neural network"
DEFAULT_PER_PAGE = 3
_SELECT = "id,display_name"
WORK_RECORD_SELECT = "id,display_name,abstract_inverted_index"
EVIDENCE_SELECT = "id,display_name,doi,ids,abstract_inverted_index"


class OpenAlexError(ClaimForgeError):
    """OpenAlex returned a response the smoke client cannot use."""


class SupportsGet(Protocol):
    def get(self, url: str) -> CachedResponse:
        """Fetch one absolute URL."""


@dataclass(frozen=True, slots=True)
class Work:
    id: str
    title: str


def build_works_search_url(
    query: str,
    *,
    per_page: int = DEFAULT_PER_PAGE,
    mailto: str | None = None,
    select: str = _SELECT,
) -> str:
    """Build the OpenAlex works search URL for a short query."""

    if not query.strip():
        raise ValueError("query must not be empty")
    if per_page < 1:
        raise ValueError("per_page must be >= 1")
    if not select.strip():
        raise ValueError("select must not be empty")
    params: dict[str, str] = {
        "search": query,
        "per_page": str(per_page),
        "select": select,
    }
    if mailto:
        params["mailto"] = mailto
    return f"{WORKS_ENDPOINT}?{urlencode(params, safe=',')}"


def search_works(
    client: SupportsGet,
    query: str,
    *,
    per_page: int = DEFAULT_PER_PAGE,
    mailto: str | None = None,
) -> list[Work]:
    """Return a short list of works for ``query``."""

    url = build_works_search_url(query, per_page=per_page, mailto=mailto)
    works: list[Work] = []
    for item in _result_items(client.get(url)):
        if not isinstance(item, Mapping):
            continue
        work_id = str(item.get("id") or "").strip()
        if not work_id:
            continue
        title = str(item.get("display_name") or "").strip()
        works.append(Work(id=work_id, title=title))
    return works


def search_work_records(
    client: SupportsGet,
    query: str,
    *,
    per_page: int = DEFAULT_PER_PAGE,
    mailto: str | None = None,
    select: str = WORK_RECORD_SELECT,
) -> list[dict[str, Any]]:
    """Return OpenAlex work dicts.

    The default ``select`` includes the abstract inverted index used by claim
    extraction. Evidence retrieval passes ``EVIDENCE_SELECT``, which also
    asks for DOI fields.
    """

    url = build_works_search_url(
        query,
        per_page=per_page,
        mailto=mailto,
        select=select,
    )
    records: list[dict[str, Any]] = []
    for item in _result_items(client.get(url)):
        if not isinstance(item, Mapping):
            continue
        work_id = str(item.get("id") or "").strip()
        if not work_id:
            continue
        records.append(dict(item))
    return records


def _result_items(response: CachedResponse) -> list[object]:
    if response.status_code != 200:
        snippet = response.body[:200].decode("utf-8", errors="replace")
        raise OpenAlexError(
            f"OpenAlex works search returned HTTP {response.status_code}: {snippet}"
        )
    try:
        payload = json.loads(response.body)
    except json.JSONDecodeError as exc:
        raise OpenAlexError("OpenAlex works search returned invalid JSON") from exc
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise OpenAlexError("OpenAlex works search response is missing results")
    return results


def format_works(query: str, works: list[Work]) -> str:
    """Render smoke-command output: a heading plus titles and OpenAlex ids."""

    lines = [f"OpenAlex works for: {query}"]
    if not works:
        lines.append("No works returned.")
        return "\n".join(lines)
    for index, work in enumerate(works, start=1):
        title = work.title or "(untitled)"
        lines.append(f"{index}. {title}")
        lines.append(f"   id: {work.id}")
    return "\n".join(lines)
