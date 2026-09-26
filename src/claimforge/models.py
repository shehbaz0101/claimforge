"""Strict, JSON-serializable claim schema.

Day 2 stores one atomic claim and the OpenAlex work it was taken from.
Later days can add evidence and a verdict without changing these fields.
"""

from __future__ import annotations

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


def dump_claims(claims: list[Claim]) -> list[dict[str, object]]:
    """Serialize claims for JSON output."""

    return [claim.to_json_dict() for claim in claims]
