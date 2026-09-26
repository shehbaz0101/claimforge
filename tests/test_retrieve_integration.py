"""Live evidence retrieval. Skips when a provider is unreachable or rate-limited."""

from __future__ import annotations

from pathlib import Path

import pytest

from claimforge.http_cache import CachedHttpClient, HttpRequestError, TransientNetworkError
from claimforge.models import EvidenceSource
from claimforge.retrieve import EvidenceRetrievalError, retrieve_evidence

pytestmark = pytest.mark.integration

CLAIM_TEXT = "Physics-informed neural networks reduce the error on the Burgers equation."


def test_live_retrieve_evidence_returns_a_pack(tmp_path: Path) -> None:
    client = CachedHttpClient(
        cache_dir=tmp_path,
        timeout=20,
        max_retries=1,
        backoff_base=0.2,
        backoff_max=1.0,
        max_retry_after=1.0,
        jitter=0.0,
    )
    try:
        evidence = retrieve_evidence(
            client,
            CLAIM_TEXT,
            per_source=2,
            top_k=5,
            ranker="lexical",
        )
    except TransientNetworkError as exc:
        pytest.skip(f"literature provider unreachable: {exc}")
    except HttpRequestError as exc:
        if exc.status_code in {401, 429} or exc.status_code >= 500:
            pytest.skip(f"literature provider HTTP {exc.status_code}")
        raise
    except EvidenceRetrievalError as exc:
        pytest.skip(f"every provider failed: {exc}")

    if not evidence:
        pytest.skip("providers returned no evidence")

    assert 1 <= len(evidence) <= 5
    for item in evidence:
        assert item.id.startswith("ev_")
        assert item.title.strip() or item.snippet.strip()
        assert item.url.startswith("http")
        assert item.source in set(EvidenceSource)
        assert item.score is not None and item.score >= 0
