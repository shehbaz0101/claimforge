"""Rank an evidence pack and keep the top matches for the judge.

Two rankers share one packer:

* ``embeddings`` loads a local sentence-transformers model (default
  ``all-MiniLM-L6-v2``) and scores each candidate by cosine similarity.
  Vectors are cached under ``data/cache/embeddings/``. Install it with
  ``pip install -e ".[embeddings]"``. The model is imported only when this
  ranker encodes uncached text.
* ``lexical`` is TF-IDF cosine over the candidate pack. It is the default
  when sentence-transformers is not installed, including CI. No network and
  no extra dependency.

``auto`` prefers embeddings when that extra imports, and otherwise uses the
lexical ranker. Reported scores are cosine similarity clipped to ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from claimforge.http_cache import ClaimForgeError
from claimforge.literature import tokens
from claimforge.models import Evidence, EvidenceSource

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_CACHE = Path("data/cache/embeddings")
_CACHE_VERSION = 1
_MODEL_DIR = re.compile(r"[^A-Za-z0-9._-]+")
# Same source preference as retrieval merge: OpenAlex, Semantic Scholar, arXiv.
_SOURCE_RANK = {
    EvidenceSource.openalex: 0,
    EvidenceSource.semantic_scholar: 1,
    EvidenceSource.arxiv: 2,
}


class RankerError(ClaimForgeError):
    """The ranker could not score the pack."""


class TextRanker(Protocol):
    """Score documents against one query. One finite score per document."""

    name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return cosine-like scores aligned with ``documents``."""


class TfidfRanker:
    """Pack-local TF-IDF cosine. Title text is repeated by the packer.

    Inverse document frequency is computed on the candidate set only, so a
    term that appears in every abstract does not dominate a small pack.
    """

    name = "lexical"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return tfidf_cosine_scores(query, documents)


class EmbeddingRanker:
    """Cosine similarity with a local sentence-transformers model.

    ``encode`` replaces the model in tests. The real model is loaded on the
    first uncached encode, not at import or construction.
    """

    name = "embeddings"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        encode: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.model_name = model_name.strip()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_EMBEDDING_CACHE
        self._encode = encode
        self._model: object | None = None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        vectors = self.embed_many([query, *documents])
        query_vector = vectors[0]
        return [_cosine(query_vector, document) for document in vectors[1:]]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, reading and writing the disk cache."""

        cached: dict[str, list[float]] = {}
        missing: list[str] = []
        seen_missing: set[str] = set()
        for text in texts:
            if text in cached or text in seen_missing:
                continue
            loaded = self._read_cache(text)
            if loaded is not None:
                cached[text] = loaded
                continue
            seen_missing.add(text)
            missing.append(text)
        if missing:
            encoded = self._encode_uncached(missing)
            if len(encoded) != len(missing):
                raise RankerError(
                    f"encoder returned {len(encoded)} vectors for {len(missing)} texts"
                )
            for text, vector in zip(missing, encoded, strict=True):
                self._write_cache(text, vector)
                cached[text] = vector
        return [cached[text] for text in texts]

    def _encode_uncached(self, texts: Sequence[str]) -> list[list[float]]:
        if self._encode is not None:
            return [_as_vector(row) for row in self._encode(texts)]
        model = self._load_model()
        try:
            matrix = model.encode(  # type: ignore[attr-defined]
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RankerError(f"embedding model failed: {exc}") from exc
        return [_as_vector(row) for row in matrix]

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RankerError(
                "sentence-transformers is not installed. "
                'Install the embedding ranker with: pip install -e ".[embeddings]"'
            ) from exc
        logger.info("loading embedding model %s", self.model_name)
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise RankerError(
                f"could not load embedding model {self.model_name}: {exc}"
            ) from exc
        return self._model

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()
        slug = _MODEL_DIR.sub("_", self.model_name).strip("._") or "model"
        return self.cache_dir / slug / f"{digest}.json"

    def _read_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("v") != _CACHE_VERSION or payload.get("model") != self.model_name:
                raise ValueError("cache record does not match this model")
            vector = payload["vector"]
            if not isinstance(vector, list):
                raise ValueError("vector must be a list")
            values = _as_vector(vector)
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            RankerError,
        ) as exc:
            logger.warning("ignoring unreadable embedding cache %s (%s)", path, exc)
            return None
        return values

    def _write_cache(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"v": _CACHE_VERSION, "model": self.model_name, "vector": vector}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(path)


def embeddings_available() -> bool:
    """True when sentence-transformers can be imported. Does not load a model."""

    return importlib.util.find_spec("sentence_transformers") is not None


def resolve_ranker(
    choice: str | TextRanker = "auto",
    *,
    cache_dir: Path | None = None,
    model_name: str | None = None,
) -> TextRanker:
    """Return the ranker for ``auto``, ``lexical``, or ``embeddings``.

    ``auto`` uses embeddings when sentence-transformers is installed.
    ``CLAIMFORGE_EMBEDDING_MODEL`` overrides the default model name when
    ``model_name`` is omitted. A ranker instance is returned unchanged.
    """

    if not isinstance(choice, str):
        return choice
    normalized = choice.strip().lower()
    if normalized in {"lexical", "tfidf"}:
        return TfidfRanker()
    if normalized == "embeddings":
        if not embeddings_available():
            raise RankerError(
                "sentence-transformers is not installed. "
                'Install the embedding ranker with: pip install -e ".[embeddings]"'
            )
        return EmbeddingRanker(model_name=_embedding_model_name(model_name), cache_dir=cache_dir)
    if normalized in {"", "auto"}:
        if embeddings_available():
            return EmbeddingRanker(
                model_name=_embedding_model_name(model_name),
                cache_dir=cache_dir,
            )
        return TfidfRanker()
    raise ValueError("ranker must be 'auto', 'lexical', or 'embeddings'")


def active_ranker_name(choice: str = "auto") -> str:
    """Name of the ranker ``choice`` would use, without loading a model."""

    normalized = choice.strip().lower()
    if normalized in {"lexical", "tfidf"}:
        return "lexical"
    if normalized == "embeddings":
        return "embeddings"
    if embeddings_available():
        return "embeddings"
    return "lexical"


def evidence_text(item: Evidence, *, repeat_title: bool = False) -> str:
    """Text the ranker compares to the claim.

    The lexical ranker repeats the title so a title hit outweighs a snippet
    hit. The embedding ranker keeps a single title plus the snippet.
    """

    title = item.title.strip()
    snippet = item.snippet.strip()
    parts: list[str] = []
    if title:
        parts.append(title)
        if repeat_title:
            parts.append(title)
    if snippet:
        parts.append(snippet)
    return "\n".join(parts)


def pack_top_k(
    evidence: Sequence[Evidence],
    query: str,
    *,
    top_k: int,
    ranker: TextRanker | None = None,
) -> list[Evidence]:
    """Re-score ``evidence`` and return the best ``top_k`` records.

    Scores are cosine similarity clipped to ``[0, 1]``. Ties break toward
    OpenAlex, then Semantic Scholar, then arXiv, then title and id.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not evidence:
        return []
    active = ranker if ranker is not None else TfidfRanker()
    documents = [
        evidence_text(item, repeat_title=active.name == "lexical") for item in evidence
    ]
    scores = active.score(query, documents)
    if len(scores) != len(evidence):
        raise RankerError(
            f"ranker returned {len(scores)} scores for {len(evidence)} documents"
        )
    rescored = [
        item.model_copy(update={"score": _clip_score(score)})
        for item, score in zip(evidence, scores, strict=True)
    ]
    rescored.sort(
        key=lambda item: (
            -(item.score or 0.0),
            _SOURCE_RANK[item.source],
            item.title.casefold(),
            item.id,
        )
    )
    return rescored[:top_k]


def tfidf_cosine_scores(query: str, documents: Sequence[str]) -> list[float]:
    """TF-IDF cosine of ``query`` against each document, in ``[0, 1]``."""

    if not documents:
        return []
    document_tokens = [tokens(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for unique in (set(document) for document in document_tokens):
        document_frequency.update(unique)
    n_docs = len(documents)

    def weight(term_counts: Mapping[str, int]) -> dict[str, float]:
        weighted: dict[str, float] = {}
        for term, count in term_counts.items():
            if count <= 0:
                continue
            idf = math.log((1.0 + n_docs) / (1.0 + document_frequency[term])) + 1.0
            weighted[term] = (1.0 + math.log(count)) * idf
        return weighted

    query_weights = weight(Counter(tokens(query)))
    scores: list[float] = []
    for document in document_tokens:
        scores.append(_cosine_weights(query_weights, weight(Counter(document))))
    return scores


def _embedding_model_name(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env = os.environ.get("CLAIMFORGE_EMBEDDING_MODEL", "").strip()
    return env or DEFAULT_EMBEDDING_MODEL


def _as_vector(row: Sequence[float]) -> list[float]:
    values = [float(item) for item in row]
    if not values or any(not math.isfinite(item) for item in values):
        raise RankerError("encoder returned an empty or non-finite vector")
    return values


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0 or dot <= 0.0:
        return 0.0
    return min(1.0, dot / (math.sqrt(left_norm) * math.sqrt(right_norm)))


def _cosine_weights(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    if dot <= 0.0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return min(1.0, dot / (left_norm * right_norm))


def _clip_score(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RankerError("ranker returned a non-numeric score") from exc
    if not math.isfinite(number):
        raise RankerError("ranker returned a non-finite score")
    if number < 0.0:
        number = 0.0
    if number > 1.0:
        number = 1.0
    return round(number, 4)
