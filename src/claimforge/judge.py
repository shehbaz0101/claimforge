"""Judge a claim against a ranked evidence pack.

The default path is a deterministic rubric and does not call a model.

* ``relevance`` is the average of the max and the mean ranker score on the
  packed evidence, in ``[0, 1]``. Missing scores are ignored. A pack with no
  scores scores 0.
* ``coverage`` rises with non-empty snippets and distinct catalogs, in
  ``[0, 1]``. Two non-empty snippets from two catalogs score 1. One
  non-empty snippet from one catalog scores 0.5.
* ``stance_lexical`` is signed cue overlap in ``[-1, 1]``. Support and
  refute phrases, plus a few directional verbs (``reduce`` versus
  ``increase``), vote only when the snippet shares claim terms. Positive
  leans support. Negative leans refute. Cue-free snippets pull the score
  toward 0.

Support requires high relevance, high coverage, and a support lean. Refute
requires a refute lean and adequate relevance. Every other pack is
insufficient, including an empty one.

An OpenAI-compatible judge runs only when both ``CLAIMFORGE_LLM_API_KEY``
and ``CLAIMFORGE_LLM_MODEL`` are set, the same gate as claim extraction.
``CLAIMFORGE_LLM_PROVIDER=rules`` forces the rubric. A failed model call
falls back to the rubric. Unset variables skip the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from claimforge.extract import LlmConfig, load_llm_config
from claimforge.http_cache import USER_AGENT
from claimforge.literature import content_terms, tokens
from claimforge.models import Claim, Evidence, Verdict, VerdictLabel

logger = logging.getLogger(__name__)

# Both bars are inclusive. Relevance is (max + mean) / 2 of ranker scores.
RELEVANCE_HIGH = 0.55
COVERAGE_HIGH = 0.75
RELEVANCE_ADEQUATE = 0.35
STANCE_LEAN = 0.20

_OVERLAP_MIN_TERMS = 2
_OVERLAP_MIN_RATIO = 0.34
_NEGATION_WINDOW = 3

_SUPPORT_RE = re.compile(
    r"\b(?:supports?|supported|confirm(?:s|ed)?|consistent with|"
    r"replicat(?:e|es|ed)|evidence (?:that|for)|agrees? with|"
    r"demonstrat(?:e|es|ed))\b",
    re.IGNORECASE,
)
_REFUTE_RE = re.compile(
    r"\b(?:refut(?:e|es|ed)|contradict(?:s|ed|ion)?|inconsistent with|"
    r"contrary to|disprov(?:e|es|ed)|rejects?|no evidence|lack of evidence|"
    r"fails? to|failed to|no significant|no effect|not associated|"
    r"unable to replicate|(?:do|does|did) not (?:support|confirm|replicate)|"
    r"cannot (?:support|confirm|replicate)|can not (?:support|confirm|replicate))\b",
    re.IGNORECASE,
)
_NEGATORS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "cannot",
        "can't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "fail",
        "fails",
        "failed",
        "lack",
        "lacks",
    }
)
_LEMMA = {
    "reduce": "reduce",
    "reduces": "reduce",
    "reduced": "reduce",
    "reducing": "reduce",
    "decrease": "decrease",
    "decreases": "decrease",
    "decreased": "decrease",
    "decreasing": "decrease",
    "lower": "lower",
    "lowers": "lower",
    "lowered": "lower",
    "lowering": "lower",
    "increase": "increase",
    "increases": "increase",
    "increased": "increase",
    "increasing": "increase",
    "improve": "improve",
    "improves": "improve",
    "improved": "improve",
    "improving": "improve",
    "raise": "raise",
    "raises": "raise",
    "raised": "raise",
    "raising": "raise",
    "worsen": "worsen",
    "worsens": "worsen",
    "worsened": "worsen",
    "worsening": "worsen",
}
_OPPOSITE: dict[str, frozenset[str]] = {
    "reduce": frozenset({"increase", "raise", "worsen"}),
    "decrease": frozenset({"increase", "raise", "worsen"}),
    "lower": frozenset({"increase", "raise"}),
    "improve": frozenset({"worsen"}),
    "increase": frozenset({"decrease", "reduce", "lower"}),
    "raise": frozenset({"lower", "decrease", "reduce"}),
    "worsen": frozenset({"improve"}),
}

_SYSTEM_PROMPT = (
    "You judge whether packed scientific evidence supports a claim. "
    "Reply with a JSON object only, shaped as "
    '{"label": "support"|"refute"|"insufficient", '
    '"confidence": number between 0 and 1, '
    '"rationale": string of at most 400 characters, '
    '"evidence_ids": [string], '
    '"rubric_scores": {"relevance": number from 0 to 1, '
    '"coverage": number from 0 to 1, '
    '"stance_lexical": number from -1 to 1}}. '
    "Use support only when the evidence clearly agrees, refute when it "
    "clearly disagrees, and insufficient when the pack is missing, "
    "off-topic, or mixed. stance_lexical is negative for refute and "
    "positive for support."
)


class LlmJudgeError(Exception):
    """The optional LLM judge was configured but did not return a verdict."""


def claim_from_text(text: str) -> Claim:
    """Build a stable claim for raw text that did not come from an abstract.

    The id is ``clm_`` plus a short hash of the collapsed text. The source
    work id is ``claimforge:text``.
    """

    stripped = " ".join(text.split())
    if not stripped:
        raise ValueError("claim text must not be empty")
    digest = hashlib.sha256(stripped.casefold().encode()).hexdigest()[:16]
    return Claim(
        id=f"clm_{digest}",
        text=stripped,
        source_work_id="claimforge:text",
        source_title="",
    )


def relevance_score(evidence: Sequence[Evidence]) -> float:
    """Average of the max and the mean ranker score, in ``[0, 1]``."""

    scores = [_clipped_score(item.score) for item in evidence if item.score is not None]
    if not scores:
        return 0.0
    peak = max(scores)
    mean = sum(scores) / len(scores)
    return round((peak + mean) / 2.0, 4)


def coverage_score(evidence: Sequence[Evidence]) -> float:
    """Score non-empty snippets and distinct catalogs, in ``[0, 1]``.

    Two non-empty snippets from two sources score 1. One non-empty snippet
    from one source scores 0.5. Empty snippets do not count.
    """

    nonempty = [item for item in evidence if item.snippet.strip()]
    if not nonempty:
        return 0.0
    snippet_part = min(len(nonempty) / 2.0, 1.0)
    source_part = min(len({item.source for item in nonempty}) / 2.0, 1.0)
    return round((snippet_part + source_part) / 2.0, 4)


def stance_lexical_score(claim_text: str, evidence: Sequence[Evidence]) -> float:
    """Signed support/refute cue overlap in ``[-1, 1]``.

    Each non-empty snippet votes -1, 0, or +1. Ranker scores weight the
    votes. A missing score weighs 1. A zero score does not vote. Snippets
    with no stance cue vote 0 and pull the mean toward the middle.
    """

    weighted: list[tuple[float, float]] = []
    for item in evidence:
        if not item.snippet.strip():
            continue
        weight = 1.0 if item.score is None else _clipped_score(item.score)
        if weight <= 0.0:
            continue
        weighted.append((weight, _snippet_stance(claim_text, item.snippet)))
    if not weighted:
        return 0.0
    total = sum(weight for weight, _vote in weighted)
    signed = sum(weight * vote for weight, vote in weighted) / total
    return round(min(1.0, max(-1.0, signed)), 4)


def rubric_verdict(claim: Claim, evidence: Sequence[Evidence]) -> Verdict:
    """Score ``claim`` with the rubric. Does not read the environment."""

    packed = list(evidence)
    relevance = relevance_score(packed)
    coverage = coverage_score(packed)
    stance = stance_lexical_score(claim.text, packed)
    label = _label_for(relevance, coverage, stance)
    return Verdict(
        claim_id=claim.id,
        label=label,
        confidence=_confidence(label, relevance, coverage, stance),
        rationale=_rationale(label, relevance, coverage, stance, packed),
        evidence_ids=[item.id for item in packed],
        rubric_scores={
            "relevance": relevance,
            "coverage": coverage,
            "stance_lexical": stance,
        },
    )


def judge_claim(claim: Claim, evidence: Sequence[Evidence]) -> Verdict:
    """Judge one claim against packed evidence.

    Uses the rubric unless both LLM settings are present. A failed model
    call returns the rubric verdict.
    """

    rubric = rubric_verdict(claim, evidence)
    config = load_llm_config()
    if config is None:
        return rubric
    try:
        return _judge_with_llm(claim, list(evidence), config)
    except LlmJudgeError as exc:
        logger.warning("LLM judge failed; using the rubric: %s", exc)
        return rubric


def _label_for(relevance: float, coverage: float, stance: float) -> VerdictLabel:
    if relevance >= RELEVANCE_HIGH and coverage >= COVERAGE_HIGH and stance >= STANCE_LEAN:
        return VerdictLabel.support
    if stance <= -STANCE_LEAN and relevance >= RELEVANCE_ADEQUATE:
        return VerdictLabel.refute
    return VerdictLabel.insufficient


def _confidence(
    label: VerdictLabel,
    relevance: float,
    coverage: float,
    stance: float,
) -> float:
    """Confidence in the assigned label, in ``[0, 1]``."""

    if label is VerdictLabel.support:
        raw = 0.5 * relevance + 0.2 * coverage + 0.3 * max(stance, 0.0)
    elif label is VerdictLabel.refute:
        raw = 0.6 * relevance + 0.4 * max(-stance, 0.0)
    else:
        weakness = 1.0 - max(relevance, coverage * 0.5, abs(stance))
        raw = 0.35 + 0.45 * max(weakness, 0.0)
    return round(min(1.0, max(0.0, raw)), 4)


def _rationale(
    label: VerdictLabel,
    relevance: float,
    coverage: float,
    stance: float,
    evidence: Sequence[Evidence],
) -> str:
    if not evidence:
        return "No evidence was packed for this claim, so the verdict is insufficient."
    if label is VerdictLabel.support:
        return (
            f"Relevance {relevance:.2f} and coverage {coverage:.2f} are high, "
            f"and lexical cues lean support ({stance:+.2f})."
        )
    if label is VerdictLabel.refute:
        return (
            f"Lexical cues lean refute ({stance:+.2f}) "
            f"with adequate relevance ({relevance:.2f})."
        )
    return (
        "The packed evidence does not meet the support or refute rule "
        f"(relevance {relevance:.2f}, coverage {coverage:.2f}, stance {stance:+.2f})."
    )


def _clipped_score(score: float) -> float:
    return min(1.0, max(0.0, float(score)))


def _term_set(text: str) -> set[str]:
    chosen = content_terms(text, limit=24) or tokens(text)[:24]
    expanded = set(chosen)
    for term in chosen:
        lemma = _LEMMA.get(term)
        if lemma:
            expanded.add(lemma)
    return expanded


def _overlaps(claim_text: str, snippet: str) -> bool:
    claim_terms = _term_set(claim_text)
    if not claim_terms:
        return False
    shared = claim_terms & _term_set(snippet)
    if len(shared) < _OVERLAP_MIN_TERMS:
        return False
    return len(shared) / len(claim_terms) >= _OVERLAP_MIN_RATIO


def _word_tokens(text: str) -> list[str]:
    return tokens(text)


def _negated(words: Sequence[str], index: int) -> bool:
    start = max(0, index - _NEGATION_WINDOW)
    return any(word in _NEGATORS for word in words[start:index])


def _claim_lemmas(claim_text: str) -> set[str]:
    found: set[str] = set()
    for word in _word_tokens(claim_text):
        lemma = _LEMMA.get(word)
        if lemma:
            found.add(lemma)
    return found


def _direction_refuted(claim_text: str, snippet: str) -> bool:
    lemmas = _claim_lemmas(claim_text)
    if not lemmas:
        return False
    opposites: set[str] = set()
    for lemma in lemmas:
        opposites.update(_OPPOSITE.get(lemma, ()))
    words = _word_tokens(snippet)
    for index, word in enumerate(words):
        lemma = _LEMMA.get(word)
        if lemma is None:
            continue
        negated = _negated(words, index)
        if lemma in lemmas and negated:
            return True
        if lemma in opposites and not negated:
            return True
    return False


def _direction_affirmed(claim_text: str, snippet: str) -> bool:
    lemmas = _claim_lemmas(claim_text)
    if not lemmas:
        return False
    words = _word_tokens(snippet)
    for index, word in enumerate(words):
        lemma = _LEMMA.get(word)
        if lemma in lemmas and not _negated(words, index):
            return True
    return False


def _snippet_stance(claim_text: str, snippet: str) -> float:
    if not _overlaps(claim_text, snippet):
        return 0.0
    if _REFUTE_RE.search(snippet) or _direction_refuted(claim_text, snippet):
        return -1.0
    if _SUPPORT_RE.search(snippet) or _direction_affirmed(claim_text, snippet):
        return 1.0
    return 0.0


def _judge_with_llm(
    claim: Claim,
    evidence: Sequence[Evidence],
    config: LlmConfig,
) -> Verdict:
    payload = {
        "claim": {"id": claim.id, "text": claim.text},
        "evidence": [
            {
                "id": item.id,
                "title": item.title,
                "snippet": item.snippet,
                "source": item.source.value,
                "score": item.score,
            }
            for item in evidence
        ],
    }
    content = _chat_completion(config, json.dumps(payload, ensure_ascii=False))
    return _verdict_from_llm(claim, evidence, content)


def _verdict_from_llm(
    claim: Claim,
    evidence: Sequence[Evidence],
    content: str,
) -> Verdict:
    payload = _parse_llm_json(content)
    label_raw = payload.get("label")
    if not isinstance(label_raw, str):
        raise LlmJudgeError("LLM response is missing a label")
    try:
        label = VerdictLabel(label_raw.strip().lower())
    except ValueError as exc:
        raise LlmJudgeError("LLM response label is not support, refute, or insufficient") from exc

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise LlmJudgeError("LLM response is missing a confidence")
    if not (0.0 <= float(confidence) <= 1.0):
        raise LlmJudgeError("LLM response confidence is outside 0 to 1")

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise LlmJudgeError("LLM response is missing a rationale")

    pack_ids = [item.id for item in evidence]
    evidence_ids = _llm_evidence_ids(payload.get("evidence_ids"), pack_ids)
    rubric_scores = _llm_rubric_scores(payload.get("rubric_scores"), claim, evidence)
    try:
        return Verdict(
            claim_id=claim.id,
            label=label,
            confidence=round(float(confidence), 4),
            rationale=rationale,
            evidence_ids=evidence_ids,
            rubric_scores=rubric_scores,
        )
    except ValueError as exc:
        raise LlmJudgeError("LLM response did not match the verdict schema") from exc


def _llm_evidence_ids(raw: object, pack_ids: Sequence[str]) -> list[str]:
    if raw is None:
        return list(pack_ids)
    if not isinstance(raw, list):
        raise LlmJudgeError("LLM response evidence_ids must be a list")
    allowed = set(pack_ids)
    chosen: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip() in allowed:
            chosen.append(item.strip())
    return chosen or list(pack_ids)


def _llm_rubric_scores(
    raw: object,
    claim: Claim,
    evidence: Sequence[Evidence],
) -> dict[str, float | None]:
    if raw is None:
        return dict(rubric_verdict(claim, evidence).rubric_scores)
    if not isinstance(raw, Mapping):
        raise LlmJudgeError("LLM response rubric_scores must be an object")
    scores: dict[str, float | None] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise LlmJudgeError("LLM response rubric criterion must be a string")
        if value is None:
            scores[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LlmJudgeError("LLM response rubric score must be a number or null")
        scores[key] = float(value)
    if not scores:
        return dict(rubric_verdict(claim, evidence).rubric_scores)
    return scores


def _parse_llm_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LlmJudgeError("LLM response was not JSON") from exc
    if not isinstance(payload, dict):
        raise LlmJudgeError("LLM response JSON was not an object")
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
        raise LlmJudgeError(f"LLM endpoint returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LlmJudgeError("LLM endpoint request failed") from exc
    except TimeoutError as exc:
        raise LlmJudgeError("LLM endpoint timed out") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LlmJudgeError("LLM endpoint returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise LlmJudgeError("LLM endpoint returned an unexpected body")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmJudgeError("LLM endpoint returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LlmJudgeError("LLM endpoint returned an empty message")
    return content
