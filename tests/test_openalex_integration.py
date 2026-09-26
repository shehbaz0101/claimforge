"""One live OpenAlex call. Skips on network failure or a transient HTTP status."""

from __future__ import annotations

from pathlib import Path

import pytest

from claimforge.http_cache import CachedHttpClient, HttpRequestError, TransientNetworkError
from claimforge.openalex import DEFAULT_QUERY, search_works

pytestmark = pytest.mark.integration


def test_live_openalex_search_returns_works(tmp_path: Path) -> None:
    client = CachedHttpClient(
        cache_dir=tmp_path,
        timeout=20,
        max_retries=2,
        backoff_base=0.5,
        jitter=0.0,
    )
    try:
        works = search_works(client, DEFAULT_QUERY, per_page=3)
    except TransientNetworkError as exc:
        pytest.skip(f"OpenAlex unreachable: {exc}")
    except HttpRequestError as exc:
        if exc.status_code == 429 or exc.status_code >= 500:
            pytest.skip(f"OpenAlex transient HTTP {exc.status_code}")
        raise

    assert 1 <= len(works) <= 3
    for work in works:
        assert work.id.startswith("https://openalex.org/W")
        assert work.title.strip()
