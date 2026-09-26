"""Offline ranker tests. The MiniLM test skips unless the extra is installed."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from claimforge.literature import lexical_score
from claimforge.models import Evidence, EvidenceSource
from claimforge.rank import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingRanker,
    RankerError,
    TfidfRanker,
    active_ranker_name,
    embeddings_available,
    pack_top_k,
    resolve_ranker,
)


def _evidence(
    *,
    source: EvidenceSource,
    title: str,
    snippet: str = "",
    suffix: str = "1",
) -> Evidence:
    return Evidence(
        id=f"ev_{source.value}_{suffix}",
        title=title,
        snippet=snippet,
        source=source,
        url=f"https://example.test/{source.value}/{suffix}",
        work_id=f"https://openalex.org/W{suffix}" if source is EvidenceSource.openalex else None,
        arxiv_id=f"2101.0000{suffix}" if source is EvidenceSource.arxiv else None,
    )


def test_tfidf_cosine_is_one_for_the_same_term_and_zero_otherwise() -> None:
    scores = TfidfRanker().score("alpha", ["alpha", "beta"])
    assert scores == [1.0, 0.0]


def test_tfidf_prefers_a_tighter_match_than_term_overlap() -> None:
    query = "neural networks burgers"
    exact = "Neural networks burgers"
    padded = "Neural networks burgers " + " ".join(f"topic{i}" for i in range(30))
    assert lexical_score(query, exact, "") == lexical_score(query, padded, "")
    scores = TfidfRanker().score(query, [exact, padded])
    assert scores[0] > scores[1]
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_lexical_pack_prefers_title_matches_and_keeps_top_k() -> None:
    query = "Burgers equation neural networks"
    titled = _evidence(
        source=EvidenceSource.arxiv,
        title="Neural networks for the Burgers equation",
        suffix="t",
    )
    buried = _evidence(
        source=EvidenceSource.openalex,
        title="A survey",
        snippet="neural networks and the Burgers equation appear here among other topics",
        suffix="b",
    )
    unrelated = _evidence(
        source=EvidenceSource.semantic_scholar,
        title="Lattice gauge theory",
        snippet="Quantum chromodynamics on the lattice.",
        suffix="u",
    )
    packed = pack_top_k([buried, unrelated, titled], query, top_k=2)
    assert [item.id for item in packed] == [titled.id, buried.id]
    assert packed[0].score is not None and packed[1].score is not None
    assert packed[0].score > packed[1].score
    assert unrelated.id not in {item.id for item in packed}


def test_pack_breaks_score_ties_toward_openalex() -> None:
    class Flat:
        name = "lexical"

        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return [0.5, 0.5]

    arxiv = _evidence(source=EvidenceSource.arxiv, title="Same paper", suffix="a")
    openalex = _evidence(source=EvidenceSource.openalex, title="Same paper", suffix="o")
    packed = pack_top_k([arxiv, openalex], "query", top_k=2, ranker=Flat())
    assert [item.source for item in packed] == [
        EvidenceSource.openalex,
        EvidenceSource.arxiv,
    ]


def test_pack_clips_cosine_into_the_unit_interval() -> None:
    item = _evidence(source=EvidenceSource.arxiv, title="Alpha", suffix="c")

    class Wide:
        name = "embeddings"

        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return [1.2, -0.4][: len(documents)]

    high = pack_top_k([item], "alpha", top_k=1, ranker=Wide())
    assert high[0].score == 1.0

    class Negative:
        name = "embeddings"

        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return [-0.4 for _ in documents]

    low = pack_top_k([item], "alpha", top_k=1, ranker=Negative())
    assert low[0].score == 0.0


def test_pack_rejects_non_finite_and_short_scores() -> None:
    item = _evidence(source=EvidenceSource.arxiv, title="Alpha", suffix="n")

    class Bad:
        name = "lexical"

        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return [float("nan")]

    with pytest.raises(RankerError, match="non-finite"):
        pack_top_k([item], "alpha", top_k=1, ranker=Bad())

    class Short:
        name = "lexical"

        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return []

    with pytest.raises(RankerError, match="scores"):
        pack_top_k([item], "alpha", top_k=1, ranker=Short())


def test_pack_top_k_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        pack_top_k([], "alpha", top_k=0)


def test_embedding_ranker_caches_vectors_and_reuses_them(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def encode(texts: Sequence[str]) -> list[list[float]]:
        calls.append(list(texts))
        table = {
            "same": [1.0, 0.0],
            "other": [0.0, 1.0],
        }
        return [table.get(text, [0.0, 1.0]) for text in texts]

    ranker = EmbeddingRanker(cache_dir=tmp_path, encode=encode, model_name="stub-model")
    first = ranker.score("same", ["same", "other"])
    second = ranker.score("same", ["same", "other"])

    assert first == [1.0, 0.0]
    assert second == first
    assert calls == [["same", "other"]]
    cached = list(tmp_path.rglob("*.json"))
    assert len(cached) == 2
    payload = json.loads(cached[0].read_text(encoding="utf-8"))
    assert payload["v"] == 1
    assert payload["model"] == "stub-model"
    assert payload["vector"]


def test_embedding_cache_ignores_a_corrupt_file_and_rewrites_it(tmp_path: Path) -> None:
    calls: list[int] = []

    def encode(texts: Sequence[str]) -> list[list[float]]:
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    ranker = EmbeddingRanker(cache_dir=tmp_path, encode=encode, model_name="stub-model")
    ranker.score("query", ["doc"])
    assert calls == [2]
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    again = ranker.score("query", ["doc"])
    assert again[0] > 0.0
    assert calls == [2, 1]
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "stub-model"


def test_auto_ranker_uses_lexical_without_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    ranker = resolve_ranker("auto")
    assert isinstance(ranker, TfidfRanker)
    assert ranker.name == "lexical"
    assert active_ranker_name("auto") == "lexical"


def test_auto_ranker_prefers_embeddings_when_the_extra_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: True)
    monkeypatch.setenv("CLAIMFORGE_EMBEDDING_MODEL", "custom-mini")
    ranker = resolve_ranker("auto", cache_dir=tmp_path)
    assert isinstance(ranker, EmbeddingRanker)
    assert ranker.name == "embeddings"
    assert ranker.model_name == "custom-mini"
    assert ranker.cache_dir == tmp_path
    assert active_ranker_name("auto") == "embeddings"


def test_explicit_model_name_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: True)
    monkeypatch.setenv("CLAIMFORGE_EMBEDDING_MODEL", "from-env")
    ranker = resolve_ranker("embeddings", cache_dir=tmp_path, model_name=DEFAULT_EMBEDDING_MODEL)
    assert isinstance(ranker, EmbeddingRanker)
    assert ranker.model_name == DEFAULT_EMBEDDING_MODEL


def test_embeddings_choice_requires_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    with pytest.raises(RankerError, match="sentence-transformers"):
        resolve_ranker("embeddings")


def test_resolve_ranker_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        resolve_ranker("bm25")


@pytest.mark.skipif(
    not embeddings_available(),
    reason='sentence-transformers is not installed; pip install -e ".[embeddings]"',
)
def test_minilm_ranks_a_related_sentence_higher(tmp_path: Path) -> None:
    ranker = EmbeddingRanker(cache_dir=tmp_path)
    query = "Physics-informed neural networks reduce the error on the Burgers equation."
    related = "Physics-informed neural networks reduce Burgers equation error."
    unrelated = "This note discusses lattice gauge theory and quantum chromodynamics."
    scores = ranker.score(query, [related, unrelated])
    assert scores[0] > scores[1]
    assert ranker.score(query, [related, unrelated]) == scores
    assert list(tmp_path.rglob("*.json"))
