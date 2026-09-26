"""Claim schema: strict fields and JSON serialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from claimforge.models import Claim, ClaimType, Evidence, EvidenceSource, dump_claims, dump_evidence


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


def _evidence(**overrides: object) -> Evidence:
    payload: dict[str, object] = {
        "id": "ev_abc",
        "title": "Physics-informed neural networks",
        "snippet": "We study the Burgers equation.",
        "source": EvidenceSource.openalex,
        "work_id": "https://openalex.org/W1",
        "doi": "10.1000/pinn",
        "arxiv_id": "1706.03762",
        "url": "https://openalex.org/W1",
        "score": 0.5,
    }
    payload.update(overrides)
    return Evidence.model_validate(payload)


def test_evidence_round_trips_through_json() -> None:
    evidence = _evidence()
    decoded = json.loads(json.dumps(dump_evidence([evidence])))
    assert decoded == [
        {
            "id": "ev_abc",
            "title": "Physics-informed neural networks",
            "snippet": "We study the Burgers equation.",
            "source": "openalex",
            "work_id": "https://openalex.org/W1",
            "doi": "10.1000/pinn",
            "arxiv_id": "1706.03762",
            "url": "https://openalex.org/W1",
            "score": 0.5,
        }
    ]
    restored = Evidence.model_validate(decoded[0])
    assert restored == evidence
    assert restored.source is EvidenceSource.openalex


def test_evidence_identifiers_are_optional() -> None:
    evidence = _evidence(work_id="  ", doi=None, arxiv_id="", score=None, snippet="  ")
    dumped = evidence.to_json_dict()
    assert dumped["work_id"] is None
    assert dumped["doi"] is None
    assert dumped["arxiv_id"] is None
    assert dumped["score"] is None
    assert dumped["snippet"] == ""


def test_evidence_source_values() -> None:
    assert [item.value for item in EvidenceSource] == [
        "openalex",
        "arxiv",
        "semantic_scholar",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": "nope"},
        {"id": " "},
        {"url": ""},
        {"source": "pubmed"},
        {"score": -0.1},
        {"score": float("nan")},
        {"score": float("inf")},
        {"work_id": 12},
    ],
)
def test_evidence_rejects_invalid_payloads(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _evidence(**overrides)
