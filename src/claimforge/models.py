"""Strict, JSON-serializable claim and evidence schemas.

Day 2 stores one atomic claim and the OpenAlex work it was taken from.
Day 3 adds an evidence record gathered for that claim. A verdict is later.
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
    that source has them. ``score`` is optional and, when the retriever sets
    it, is a lexical overlap in ``[0, 1]`` rather than an embedding rank.
    Extra fields are rejected.
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


def dump_claims(claims: list[Claim]) -> list[dict[str, object]]:
    """Serialize claims for JSON output."""

    return [claim.to_json_dict() for claim in claims]


def dump_evidence(evidence: list[Evidence]) -> list[dict[str, object]]:
    """Serialize evidence for JSON output."""

    return [item.to_json_dict() for item in evidence]
