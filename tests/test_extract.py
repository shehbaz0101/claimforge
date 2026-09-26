"""Rule-based claim extraction. No network."""

from __future__ import annotations

import json

import pytest

from claimforge.extract import (
    abstract_from_inverted_index,
    claim_id,
    extract_claims_from_abstract,
    extract_from_openalex_work,
    load_llm_config,
    split_sentences,
)
from claimforge.models import ClaimType

FIXTURE_MULTI = (
    "Neural networks are widely used in scientific computing. "
    "We show that physics-informed neural networks reduce the error on the Burgers equation. "
    "Our method uses a residual loss to enforce the differential equation. "
    "These findings demonstrate that the approach outperforms a purely data-driven baseline."
)

FIXTURE_EMPTY = ""

WORK_ID = "https://openalex.org/W100"
TITLE = "Physics-informed neural networks"


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLAIMFORGE_LLM_API_KEY",
        "CLAIMFORGE_LLM_MODEL",
        "CLAIMFORGE_LLM_PROVIDER",
        "CLAIMFORGE_LLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_multi_claim_abstract_keeps_claim_sentences_in_order() -> None:
    claims = extract_claims_from_abstract(FIXTURE_MULTI, work_id=WORK_ID, title=TITLE)

    assert [claim.text for claim in claims] == [
        "We show that physics-informed neural networks reduce the error on the Burgers equation.",
        "Our method uses a residual loss to enforce the differential equation.",
        "These findings demonstrate that the approach outperforms a purely data-driven baseline.",
    ]
    assert [claim.claim_type for claim in claims] == [
        ClaimType.result,
        ClaimType.method,
        ClaimType.result,
    ]
    assert len({claim.id for claim in claims}) == 3
    for claim in claims:
        assert claim.source_work_id == WORK_ID
        assert claim.source_title == TITLE
        assert claim.confidence is not None
        assert 0.0 <= claim.confidence <= 1.0
        assert claim.id == claim_id(WORK_ID, claim.text, claims.index(claim))

    again = extract_claims_from_abstract(FIXTURE_MULTI, work_id=WORK_ID, title=TITLE)
    assert [claim.to_json_dict() for claim in again] == [claim.to_json_dict() for claim in claims]
    json.dumps([claim.to_json_dict() for claim in claims])


@pytest.mark.parametrize("abstract", [FIXTURE_EMPTY, "   \n\t  ", "\n"])
def test_empty_abstract_returns_no_claims(abstract: str) -> None:
    assert extract_claims_from_abstract(abstract, work_id=WORK_ID, title=TITLE) == []


def test_background_only_abstract_returns_no_claims() -> None:
    abstract = (
        "Neural networks are widely used in scientific computing. "
        "This paper reviews prior work on the topic."
    )
    assert extract_claims_from_abstract(abstract, work_id=WORK_ID, title=TITLE) == []


@pytest.mark.parametrize(
    ("sentence", "claim_type"),
    [
        ("We show that the bound is tight.", ClaimType.result),
        ("The results indicate a large effect in the trial.", ClaimType.result),
        ("Our findings suggest the effect is stable across folds.", ClaimType.result),
        ("There is evidence that the treatment helps patients.", ClaimType.result),
        ("The study demonstrates that the loss falls quickly.", ClaimType.result),
        ("The study indicates that most of the effect is stable.", ClaimType.result),
        ("We propose a method for sparse recovery tasks.", ClaimType.method),
        ("Our approach uses a residual penalty during training.", ClaimType.method),
        ("Exposure is associated with higher cardiovascular risk.", ClaimType.factual),
    ],
)
def test_cue_sentences_are_classified(sentence: str, claim_type: ClaimType) -> None:
    claims = extract_claims_from_abstract(sentence, work_id=WORK_ID, title=TITLE)
    assert len(claims) == 1
    assert claims[0].claim_type is claim_type
    assert claims[0].text == sentence


def test_split_sentences_keeps_abbreviations_and_decimals() -> None:
    text = "We show that e.g. the error is 0.5 points lower. Our method uses residuals."
    assert split_sentences(text) == [
        "We show that e.g. the error is 0.5 points lower.",
        "Our method uses residuals.",
    ]


def test_inverted_index_work_extracts_claims() -> None:
    sentence = "We show that the estimator reduces variance in the sample."
    index: dict[str, list[int]] = {}
    for position, word in enumerate(sentence.split()):
        index.setdefault(word, []).append(position)
    assert abstract_from_inverted_index(index) == sentence

    claims = extract_from_openalex_work(
        {
            "id": WORK_ID,
            "display_name": TITLE,
            "abstract_inverted_index": index,
        }
    )
    assert len(claims) == 1
    assert claims[0].text == sentence
    assert claims[0].source_work_id == WORK_ID
    assert claims[0].claim_type is ClaimType.result


def test_openalex_work_without_abstract_or_id_returns_no_claims() -> None:
    assert extract_from_openalex_work({"id": WORK_ID, "display_name": TITLE}) == []
    assert (
        extract_from_openalex_work(
            {
                "display_name": TITLE,
                "abstract": "We show that the model works on this task.",
            }
        )
        == []
    )
    assert (
        extract_from_openalex_work(
            {
                "id": WORK_ID,
                "title": "Plain title",
                "abstract": "   ",
            }
        )
        == []
    )


def test_llm_is_skipped_when_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(config: object, user_prompt: str) -> str:
        raise AssertionError("LLM client should not be called")

    monkeypatch.setattr("claimforge.extract._chat_completion", explode)
    assert load_llm_config() is None
    claims = extract_claims_from_abstract(
        "We show that the rule path still runs.",
        work_id=WORK_ID,
        title=TITLE,
    )
    assert len(claims) == 1
    assert claims[0].claim_type is ClaimType.result


def test_llm_path_uses_model_claims_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("CLAIMFORGE_LLM_BASE_URL", "https://llm.example/v1")

    def fake_chat(config: object, user_prompt: str) -> str:
        assert getattr(config, "api_key") == "test-key"
        assert getattr(config, "model") == "gpt-test"
        assert getattr(config, "base_url") == "https://llm.example/v1"
        assert "Burgers" in user_prompt
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "The model reduces error on the Burgers equation.",
                        "claim_type": "result",
                        "confidence": 0.91,
                    },
                    {"text": "too short", "claim_type": "other", "confidence": 0.2},
                    {"text": "Not a known type but still a claim sentence.", "claim_type": "speculation"},
                ]
            }
        )

    monkeypatch.setattr("claimforge.extract._chat_completion", fake_chat)
    claims = extract_claims_from_abstract(FIXTURE_MULTI, work_id=WORK_ID, title=TITLE)
    assert [claim.text for claim in claims] == [
        "The model reduces error on the Burgers equation.",
        "Not a known type but still a claim sentence.",
    ]
    assert claims[0].claim_type is ClaimType.result
    assert claims[0].confidence == 0.91
    assert claims[1].claim_type is None
    assert claims[0].source_work_id == WORK_ID


def test_llm_failure_falls_back_to_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    from claimforge.extract import LlmExtractError

    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")

    def failing_chat(config: object, user_prompt: str) -> str:
        raise LlmExtractError("down")

    monkeypatch.setattr("claimforge.extract._chat_completion", failing_chat)
    claims = extract_claims_from_abstract(
        "We show that the fallback path keeps this result.",
        work_id=WORK_ID,
        title=TITLE,
    )
    assert len(claims) == 1
    assert claims[0].text.startswith("We show that")


def test_llm_requires_both_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    assert load_llm_config() is None
    monkeypatch.delenv("CLAIMFORGE_LLM_API_KEY")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    assert load_llm_config() is None


def test_chat_completion_posts_openai_compatible_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from claimforge.extract import LlmConfig, _chat_completion

    captured: dict[str, object] = {}

    class _Response:
        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"claims\": []}"}}]}).encode()

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def fake_urlopen(request: object, timeout: float = 0) -> _Response:
        captured["url"] = getattr(request, "full_url")
        captured["auth"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("claimforge.extract.urllib.request.urlopen", fake_urlopen)
    text = _chat_completion(
        LlmConfig(
            provider="openai",
            model="gpt-test",
            api_key="test-key",
            base_url="https://llm.example/v1",
        ),
        "Abstract here",
    )
    assert text == '{"claims": []}'
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-test"
    assert body["temperature"] == 0
    assert captured["timeout"] == 30


def test_provider_rules_disables_llm_even_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("CLAIMFORGE_LLM_PROVIDER", "rules")
    assert load_llm_config() is None
