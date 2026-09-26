"""Claim schema: strict fields and JSON serialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from claimforge.models import Claim, ClaimType, dump_claims


def _claim(**overrides: object) -> Claim:
    payload: dict[str, object] = {
        "id": "clm_abc",
        "text": "We show that the bound is tight.",
        "source_work_id": "https://openalex.org/W1",
        "source_title": "A paper",
        "confidence": 0.8,
        "claim_type": ClaimType.result,
    }
    payload.update(overrides)
    return Claim.model_validate(payload)


def test_claim_round_trips_through_json() -> None:
    claim = _claim()
    encoded = json.dumps(dump_claims([claim]))
    decoded = json.loads(encoded)
    assert decoded == [
        {
            "id": "clm_abc",
            "text": "We show that the bound is tight.",
            "source_work_id": "https://openalex.org/W1",
            "source_title": "A paper",
            "confidence": 0.8,
            "claim_type": "result",
        }
    ]
    restored = Claim.model_validate(decoded[0])
    assert restored == claim
    assert restored.claim_type is ClaimType.result


def test_optional_fields_may_be_omitted() -> None:
    claim = _claim(confidence=None, claim_type=None)
    dumped = claim.to_json_dict()
    assert dumped["confidence"] is None
    assert dumped["claim_type"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": "nope"},
        {"id": "  "},
        {"text": ""},
        {"source_work_id": "   "},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"claim_type": "opinion"},
    ],
)
def test_claim_rejects_invalid_payloads(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _claim(**overrides)


def test_claim_type_values() -> None:
    assert [item.value for item in ClaimType] == ["factual", "method", "result", "other"]
