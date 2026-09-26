"""Identifier cleanup and the Day 3 term-overlap score.

The evidence pack is scored in ``claimforge.rank`` (embedding cosine, or
TF-IDF cosine when sentence-transformers is not installed). ``lexical_score``
remains the earlier overlap heuristic. These helpers stay free of HTTP so
each provider can normalize ids the same way.
"""

from __future__ import annotations

import hashlib
import re

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_DOI_BODY = re.compile(r"^10\.\S+/\S+\Z")
_ARXIV_ID = re.compile(
    r"(?i)(?:https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/|arxiv:)?"
    r"(\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})"
    r"(?:v\d+)?(?:\.pdf)?"
)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "we",
        "our",
        "their",
        "its",
        "it",
        "as",
        "at",
        "than",
        "then",
        "into",
        "over",
        "under",
        "which",
        "who",
        "what",
        "when",
        "where",
        "how",
        "not",
        "no",
        "nor",
        "but",
        "if",
        "so",
        "such",
        "can",
        "may",
        "using",
        "via",
        "between",
        "within",
        "without",
        "across",
        "after",
        "before",
        "also",
        "show",
        "shows",
        "showed",
        "shown",
        "demonstrate",
        "demonstrates",
        "demonstrated",
        "find",
        "found",
        "result",
        "results",
    }
)


def normalize_doi(value: object) -> str | None:
    """Return a bare DOI, or None when ``value`` is not a DOI."""

    if not isinstance(value, str):
        return None
    text = _DOI_PREFIX.sub("", value.strip()).strip().rstrip("/")
    if not text or _DOI_BODY.match(text) is None:
        return None
    return text


def normalize_arxiv_id(value: object) -> str | None:
    """Return an arXiv id without a version suffix, or None."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    match = _ARXIV_ID.fullmatch(text.replace(" ", ""))
    if match is None:
        return None
    return match.group(1).casefold()


def normalize_title(value: str) -> str:
    """Lowercase a title and collapse punctuation so copies can share a key."""

    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def make_evidence_id(kind: str, key: str) -> str:
    """Stable ``ev_`` id from an identifier kind and its canonical key."""

    digest = hashlib.sha256(f"{kind}\n{key}".encode()).hexdigest()
    return "ev_" + digest[:16]


def clip_text(text: str, limit: int = 300) -> str:
    """Collapse whitespace and trim ``text`` on a word boundary."""

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    clipped = collapsed[:limit].rsplit(" ", 1)[0]
    return clipped or collapsed[:limit]


def tokens(text: str) -> list[str]:
    """Return casefolded word tokens, keeping hyphenated compounds intact."""

    return [match.group(0).casefold() for match in _TOKEN.finditer(text)]


def content_terms(text: str, *, limit: int = 8) -> list[str]:
    """Return unique query terms with stopwords and tiny tokens removed."""

    chosen: list[str] = []
    seen: set[str] = set()
    for token in tokens(text):
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        chosen.append(token)
        if len(chosen) >= limit:
            break
    return chosen


def search_terms(text: str, *, limit: int = 8) -> list[str]:
    """Terms for a boolean query. Falls back to raw tokens when cues dominate."""

    terms = content_terms(text, limit=limit)
    if terms:
        return terms
    fallback: list[str] = []
    seen: set[str] = set()
    for token in tokens(text):
        if token in seen:
            continue
        seen.add(token)
        fallback.append(token)
        if len(fallback) >= limit:
            break
    return fallback


def lexical_score(query: str, title: str, snippet: str) -> float:
    """Overlap of query terms with the title and snippet, in ``[0, 1]``.

    A term in the title counts twice a term that appears only in the snippet.
    The retriever no longer uses this for the pack. ``claimforge.rank`` scores
    with TF-IDF cosine or embedding cosine.
    """

    wanted = content_terms(query, limit=24) or tokens(query)[:24]
    if not wanted:
        return 0.0
    title_tokens = set(tokens(title))
    body_tokens = set(tokens(f"{title} {snippet}"))
    hits = 0.0
    for term in wanted:
        if term in title_tokens:
            hits += 2.0
        elif term in body_tokens:
            hits += 1.0
    return round(hits / (2.0 * len(wanted)), 4)
