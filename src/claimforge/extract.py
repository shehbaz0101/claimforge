"""Extract claims from paper abstracts.

The default path is deterministic and does not call a model. Sentence
splitting plus cue patterns keep result, method, and factual sentences.
An OpenAI-compatible chat call is used only when both
``CLAIMFORGE_LLM_API_KEY`` and ``CLAIMFORGE_LLM_MODEL`` are set. Any other
LLM setting left unset skips that path. A failed LLM call falls back to
the rule extractor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from claimforge.http_cache import USER_AGENT, ClaimForgeError
from claimforge.models import Claim, ClaimType

logger = logging.getLogger(__name__)

_LLM_API_KEY = "CLAIMFORGE_LLM_API_KEY"
_LLM_MODEL = "CLAIMFORGE_LLM_MODEL"
_LLM_PROVIDER = "CLAIMFORGE_LLM_PROVIDER"
_LLM_BASE_URL = "CLAIMFORGE_LLM_BASE_URL"
_DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
_LLM_DISABLED = frozenset({"none", "off", "rules", "rule"})
_MIN_WORDS = 4

_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "fig.",
    "figs.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "vs.",
    "cf.",
    "approx.",
    "no.",
    "vol.",
    "pp.",
    "eq.",
    "eqs.",
    "ref.",
    "refs.",
    "ca.",
    "resp.",
    "al.",
)
_PERIOD_PLACEHOLDER = "\u2048"

# (type, pattern, base confidence). Higher confidence wins; ties prefer
# result, then method, then factual, then other.
_CUES: tuple[tuple[ClaimType, re.Pattern[str], float], ...] = (
    (
        ClaimType.result,
        re.compile(
            r"\bwe (?:show|showed|have shown|demonstrate|demonstrated|find|found|"
            r"observe|observed|report|reported|conclude|concluded)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        ClaimType.result,
        re.compile(
            r"\b(?:our |the |these )?results (?:show|shows|showed|indicate|indicates|"
            r"suggest|suggests|demonstrate|demonstrates|reveal|reveals|revealed)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        ClaimType.result,
        re.compile(
            r"\b(?:our |the |these )?findings (?:show|shows|indicate|indicates|"
            r"suggest|suggests|demonstrate|demonstrates|reveal|reveals)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        ClaimType.result,
        re.compile(
            r"\bevidence (?:that|suggests|suggest|indicates|indicate|shows|show)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        ClaimType.result,
        re.compile(
            r"\b(?:this|the) (?:study|work|paper|analysis) "
            r"(?:shows|showed|demonstrates|demonstrated|finds|found|reports|reported|"
            r"indicates|indicate|suggests|suggest)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        ClaimType.result,
        re.compile(
            r"\b(?:demonstrate|demonstrates|demonstrated|demonstrating that)\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        ClaimType.method,
        re.compile(
            r"\bwe (?:propose|proposed|introduce|introduced|develop|developed|"
            r"present|presented|describe|described|design|designed)\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        ClaimType.method,
        re.compile(
            r"\bwe (?:use|used|employ|employed|apply|applied)\b",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        ClaimType.method,
        re.compile(
            r"\bour (?:method|approach|model|framework|algorithm|technique)\b",
            re.IGNORECASE,
        ),
        0.65,
    ),
    (
        ClaimType.factual,
        re.compile(r"\b(?:is|are|was|were) associated with\b", re.IGNORECASE),
        0.6,
    ),
    (
        ClaimType.factual,
        re.compile(r"\b(?:leads|lead|leading) to\b", re.IGNORECASE),
        0.55,
    ),
    (
        ClaimType.factual,
        re.compile(
            r"\b(?:increase|decrease|reduce|improve|increases|decreases|reduces|"
            r"improves|increased|decreased|reduced|improved)\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
)
_TYPE_RANK = {
    ClaimType.result: 3,
    ClaimType.method: 2,
    ClaimType.factual: 1,
    ClaimType.other: 0,
}

_SYSTEM_PROMPT = (
    "You extract atomic scientific claims from a paper abstract. "
    "Reply with a JSON object only, shaped as "
    '{"claims": [{"text": string, "claim_type": "factual"|"method"|"result"|"other", '
    '"confidence": number between 0 and 1}]}. '
    "Keep each claim a single sentence grounded in the abstract. "
    "Return an empty claims array when the abstract states no claim."
)


class LlmExtractError(ClaimForgeError):
    """The optional LLM extractor was configured but did not return claims."""


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """OpenAI-compatible chat settings. The API key is never logged."""

    provider: str
    model: str
    api_key: str
    base_url: str


def load_llm_config() -> LlmConfig | None:
    """Read ``CLAIMFORGE_LLM_*``. Return None when the LLM path should be skipped.

    Both the API key and the model must be set. ``CLAIMFORGE_LLM_PROVIDER``
    values ``none``, ``off``, and ``rules`` force the rule extractor even if
    a key is present. ``CLAIMFORGE_LLM_BASE_URL`` defaults to the OpenAI
    chat endpoint.
    """

    provider = os.environ.get(_LLM_PROVIDER, "").strip().lower()
    if provider in _LLM_DISABLED:
        return None
    api_key = os.environ.get(_LLM_API_KEY, "").strip()
    model = os.environ.get(_LLM_MODEL, "").strip()
    if not api_key or not model:
        return None
    base_url = os.environ.get(_LLM_BASE_URL, "").strip() or _DEFAULT_LLM_BASE_URL
    return LlmConfig(
        provider=provider or "openai",
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
    )


def extract_claims_from_abstract(
    abstract: str,
    *,
    work_id: str,
    title: str,
) -> list[Claim]:
    """Extract claims from one abstract.

    Uses the rule extractor unless LLM settings are complete. An LLM failure
    falls back to the rules so a missing or broken endpoint still returns
    claims. An empty abstract returns an empty list without a model call.
    """

    if not abstract or not abstract.strip():
        return []
    config = load_llm_config()
    if config is not None:
        try:
            return _extract_with_llm(abstract, work_id=work_id, title=title, config=config)
        except LlmExtractError as exc:
            logger.warning("LLM claim extraction failed; using rules: %s", exc)
    return _extract_with_rules(abstract, work_id=work_id, title=title)


def extract_from_openalex_work(work: Mapping[str, Any]) -> list[Claim]:
    """Extract claims from one OpenAlex work record.

    Accepts a plain ``abstract`` string or OpenAlex's ``abstract_inverted_index``.
    A record with no id or no abstract text yields an empty list.
    """

    work_id = str(work.get("id") or "").strip()
    if not work_id:
        return []
    title = str(work.get("display_name") or work.get("title") or "")
    return extract_claims_from_abstract(
        abstract_text_from_work(work),
        work_id=work_id,
        title=title,
    )


def abstract_text_from_work(work: Mapping[str, Any]) -> str:
    """Return abstract prose from a work dict, or an empty string."""

    abstract = work.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return abstract.strip()
    inverted = work.get("abstract_inverted_index")
    if isinstance(inverted, Mapping):
        return abstract_from_inverted_index(inverted)
    return ""


def abstract_from_inverted_index(index: Mapping[str, Any]) -> str:
    """Rebuild an OpenAlex abstract from ``word -> [positions]``."""

    placed: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and not isinstance(position, bool):
                placed.append((position, word))
    placed.sort(key=lambda item: item[0])
    return " ".join(word for _, word in placed).strip()


def split_sentences(text: str) -> list[str]:
    """Split prose on sentence boundaries without breaking common abbreviations."""

    normalized = " ".join(text.split())
    if not normalized:
        return []
    protected = normalized

    def _protect(match: re.Match[str]) -> str:
        return match.group(0).replace(".", _PERIOD_PLACEHOLDER)

    for abbreviation in sorted(_ABBREVIATIONS, key=len, reverse=True):
        protected = re.sub(
            rf"(?<![A-Za-z]){re.escape(abbreviation)}",
            _protect,
            protected,
            flags=re.IGNORECASE,
        )
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_PERIOD_PLACEHOLDER}\2", protected)
    chunks = re.split(r"(?<=[.!?])\s+", protected)
    sentences: list[str] = []
    for chunk in chunks:
        sentence = chunk.replace(_PERIOD_PLACEHOLDER, ".").strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def claim_id(work_id: str, text: str, ordinal: int) -> str:
    """Stable id from the work, the claim text, and its order in that work."""

    raw = f"{work_id}\n{ordinal}\n{text.casefold()}".encode()
    return "clm_" + hashlib.sha256(raw).hexdigest()[:16]


def _extract_with_rules(abstract: str, *, work_id: str, title: str) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[str] = set()
    for sentence in split_sentences(abstract):
        if sentence.endswith("?"):
            continue
        if len(sentence.split()) < _MIN_WORDS:
            continue
        scored = _score_sentence(sentence)
        if scored is None:
            continue
        claim_type, confidence = scored
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            Claim(
                id=claim_id(work_id, sentence, len(claims)),
                text=sentence,
                source_work_id=work_id,
                source_title=title,
                confidence=confidence,
                claim_type=claim_type,
            )
        )
    return claims


def _score_sentence(sentence: str) -> tuple[ClaimType, float] | None:
    matches: list[tuple[ClaimType, float]] = []
    for claim_type, pattern, confidence in _CUES:
        if pattern.search(sentence):
            matches.append((claim_type, confidence))
    if not matches:
        return None
    best_confidence = max(confidence for _, confidence in matches)
    confidence = min(0.95, best_confidence + 0.05 * (len(matches) - 1))
    claim_type = max(matches, key=lambda item: (_TYPE_RANK[item[0]], item[1]))[0]
    return claim_type, round(confidence, 2)


def _extract_with_llm(
    abstract: str,
    *,
    work_id: str,
    title: str,
    config: LlmConfig,
) -> list[Claim]:
    prompt = f"Title: {title.strip() or '(untitled)'}\n\nAbstract:\n{abstract.strip()}"
    content = _chat_completion(config, prompt)
    payload = _parse_llm_json(content)
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise LlmExtractError("LLM response is missing a claims array")
    claims: list[Claim] = []
    seen: set[str] = set()
    for item in raw_claims:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        if len(text.split()) < _MIN_WORDS:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            Claim(
                id=claim_id(work_id, text, len(claims)),
                text=text,
                source_work_id=work_id,
                source_title=title,
                confidence=_optional_confidence(item.get("confidence")),
                claim_type=_optional_claim_type(item.get("claim_type")),
            )
        )
    return claims


def _optional_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0.0 or number > 1.0:
        return None
    return round(number, 4)


def _optional_claim_type(value: object) -> ClaimType | None:
    if not isinstance(value, str):
        return None
    try:
        return ClaimType(value.strip().lower())
    except ValueError:
        return None


def _parse_llm_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LlmExtractError("LLM response was not JSON") from exc
    if not isinstance(payload, dict):
        raise LlmExtractError("LLM response JSON was not an object")
    return payload


def _chat_completion(config: LlmConfig, user_prompt: str) -> str:
    """POST one chat completion. Tests replace this function; it is the only LLM I/O."""

    url = f"{config.base_url}/chat/completions"
    body = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise LlmExtractError(f"LLM endpoint returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LlmExtractError("LLM endpoint request failed") from exc
    except TimeoutError as exc:
        raise LlmExtractError("LLM endpoint timed out") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LlmExtractError("LLM endpoint returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise LlmExtractError("LLM endpoint returned an unexpected body")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmExtractError("LLM endpoint returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LlmExtractError("LLM endpoint returned an empty message")
    return content
