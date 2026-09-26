"""Strict, JSON-serializable claim, evidence, and verdict schemas.

Day 2 stores one atomic claim and the OpenAlex work it was taken from.
Day 3 adds an evidence record gathered for that claim. Day 5 adds the
verdict a judge records for that claim.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimType(StrEnum):
    """Coarse kind of scientific claim. Values are the JSON spellings."""

    factual = "factual"
    method = "method"
    result = "result"
    other = "other"


class Claim(BaseModel):
    """One claim extracted from a work.

    ``confidence`` and ``claim_type`` may be omitted when an extractor cannot
    score or classify the sentence. Extra fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    source_work_id: str
    source_title: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_type: ClaimType | None = None

    @field_validator("id", "text", "source_work_id")
    @classmethod
    def _require_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("source_title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        return value.strip()

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-ready dict (enum values, no Python-only types)."""

        return self.model_dump(mode="json")


class EvidenceSource(StrEnum):
    """Catalog that produced an evidence record. Values are the JSON spellings."""

    openalex = "openalex"
    arxiv = "arxiv"
    semantic_scholar = "semantic_scholar"


class Evidence(BaseModel):
    """One work that may support or refute a claim.

    ``snippet`` holds the abstract when the source returned one, otherwise a
    short snippet. ``work_id``, ``doi``, and ``arxiv_id`` are set only when
    that source has them. ``score`` is optional until ranking. When the
    retriever sets it, the value is cosine similarity from the active ranker
    (embeddings, or TF-IDF when that extra is not installed), clipped to
    ``[0, 1]``. Extra fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    snippet: str = ""
    source: EvidenceSource
    work_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str
    score: float | None = Field(default=None, ge=0.0)

    @field_validator("id", "url")
    @classmethod
    def _require_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("title", "snippet")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("work_id", "doi", "arxiv_id", mode="before")
    @classmethod
    def _blank_identifier(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("must be a string")
        stripped = value.strip()
        return stripped or None

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-ready dict (enum values, no Python-only types)."""

        return self.model_dump(mode="json")


class VerdictLabel(StrEnum):
    """Judge outcome. Values are the JSON spellings."""

    support = "support"
    refute = "refute"
    insufficient = "insufficient"


# Canonical rubric keys and the closed range each score must fall in.
# ``stance_lexical`` is signed: negative leans refute, positive leans support.
RUBRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "relevance": (0.0, 1.0),
    "coverage": (0.0, 1.0),
    "stance_lexical": (-1.0, 1.0),
}
_RATIONALE_LIMIT = 500


class Verdict(BaseModel):
    """One judge outcome for a claim.

    ``rubric_scores`` maps a criterion name to a float, or null when that
    criterion was not scored. The Day 5 rubric always fills ``relevance``,
    ``coverage``, and ``stance_lexical``. Extra fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_ids: list[str]
    rubric_scores: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("claim_id")
    @classmethod
    def _require_claim_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("confidence")
    @classmethod
    def _finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    @field_validator("rationale")
    @classmethod
    def _short_rationale(cls, value: str) -> str:
        stripped = " ".join(value.split())
        if not stripped:
            raise ValueError("must not be blank")
        if len(stripped) > _RATIONALE_LIMIT:
            raise ValueError(f"must be at most {_RATIONALE_LIMIT} characters")
        return stripped

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("evidence id must be a string")
            stripped = item.strip()
            if not stripped:
                raise ValueError("evidence id must not be blank")
            cleaned.append(stripped)
        return cleaned

    @field_validator("rubric_scores")
    @classmethod
    def _rubric_scores(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        cleaned: dict[str, float | None] = {}
        for key, score in value.items():
            if not isinstance(key, str):
                raise ValueError("rubric criterion must be a string")
            name = key.strip()
            if not name:
                raise ValueError("rubric criterion must not be blank")
            if score is None:
                cleaned[name] = None
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("rubric score must be a number or null")
            number = float(score)
            if not math.isfinite(number):
                raise ValueError("rubric score must be finite")
            low, high = RUBRIC_BOUNDS.get(name, (-1.0, 1.0))
            if number < low or number > high:
                raise ValueError(f"{name} must be between {low} and {high}")
            cleaned[name] = round(number, 4)
        return cleaned

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-ready dict (enum values, no Python-only types)."""

        return self.model_dump(mode="json")


def dump_claims(claims: list[Claim]) -> list[dict[str, object]]:
    """Serialize claims for JSON output."""

    return [claim.to_json_dict() for claim in claims]


def dump_evidence(evidence: list[Evidence]) -> list[dict[str, object]]:
    """Serialize evidence for JSON output."""

    return [item.to_json_dict() for item in evidence]


def dump_verdicts(verdicts: list[Verdict]) -> list[dict[str, object]]:
    """Serialize verdicts for JSON output."""

    return [verdict.to_json_dict() for verdict in verdicts]
