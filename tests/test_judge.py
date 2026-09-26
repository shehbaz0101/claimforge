"""Rubric judge and the verify/judge commands. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimforge.cli import main
from claimforge.judge import (
    COVERAGE_HIGH,
    RELEVANCE_ADEQUATE,
    RELEVANCE_HIGH,
    STANCE_LEAN,
    claim_from_text,
    coverage_score,
    judge_claim,
    relevance_score,
    rubric_verdict,
    stance_lexical_score,
)
from claimforge.models import Claim, Evidence, EvidenceSource, VerdictLabel
from claimforge.retrieve import EvidenceRetrievalError

CLAIM_TEXT = "Physics-informed neural networks reduce the error on the Burgers equation."
CLAIM = Claim(
    id="clm_burgers",
    text=CLAIM_TEXT,
    source_work_id="https://openalex.org/W100",
    source_title="Physics-informed neural networks",
)

SUPPORT_SNIPPETS = (
    "This study supports the finding that physics-informed neural networks "
    "reduce the error on the Burgers equation.",
    "We confirm that physics-informed neural networks reduce the error on the Burgers equation.",
)
REFUTE_SNIPPETS = (
    "A replication finds that physics-informed neural networks do not reduce "
    "the error on the Burgers equation.",
    "Results contradict the claim that physics-informed neural networks reduce "
    "the error on the Burgers equation.",
)
UNRELATED_SNIPPETS = (
    "We survey convolutional architectures for image classification on ImageNet.",
    "Transformers improve machine translation quality on WMT benchmarks.",
)


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLAIMFORGE_LLM_API_KEY",
        "CLAIMFORGE_LLM_MODEL",
        "CLAIMFORGE_LLM_PROVIDER",
        "CLAIMFORGE_LLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _evidence(
    evidence_id: str,
    snippet: str,
    *,
    score: float | None,
    source: EvidenceSource = EvidenceSource.openalex,
    title: str = "Related work",
) -> Evidence:
    return Evidence(
        id=evidence_id,
        title=title,
        snippet=snippet,
        source=source,
        url=f"https://example.test/{evidence_id}",
        score=score,
    )


def _pack(
    snippets: tuple[str, ...] | list[str],
    scores: tuple[float | None, ...] | list[float | None],
    sources: tuple[EvidenceSource, ...] | None = None,
) -> list[Evidence]:
    chosen = sources or (
        EvidenceSource.openalex,
        EvidenceSource.arxiv,
        EvidenceSource.semantic_scholar,
    )
    return [
        _evidence(
            f"ev_{index}",
            snippet,
            score=score,
            source=chosen[index % len(chosen)],
        )
        for index, (snippet, score) in enumerate(zip(snippets, scores, strict=True))
    ]


def test_support_when_relevance_coverage_and_stance_are_high() -> None:
    evidence = _pack(SUPPORT_SNIPPETS, (0.8, 0.6))
    verdict = judge_claim(CLAIM, evidence)

    assert verdict.label is VerdictLabel.support
    assert verdict.claim_id == CLAIM.id
    assert verdict.evidence_ids == ["ev_0", "ev_1"]
    assert verdict.rubric_scores == {
        "relevance": 0.75,
        "coverage": 1.0,
        "stance_lexical": 1.0,
    }
    assert verdict.confidence == pytest.approx(0.875)
    assert verdict.rationale == (
        "Relevance 0.75 and coverage 1.00 are high, "
        "and lexical cues lean support (+1.00)."
    )
    assert rubric_verdict(CLAIM, evidence) == verdict
    assert judge_claim(CLAIM, evidence) == verdict


def test_directional_verb_supports_without_the_word_support() -> None:
    evidence = _pack(
        (
            "Physics-informed neural networks reduce the error on the Burgers "
            "equation across three benchmarks.",
            "Experiments show physics-informed neural networks reduce the error "
            "on the Burgers equation.",
        ),
        (0.8, 0.7),
    )
    verdict = judge_claim(CLAIM, evidence)
    assert verdict.label is VerdictLabel.support
    assert verdict.rubric_scores["stance_lexical"] == 1.0


def test_refute_when_cues_disagree_and_relevance_is_adequate() -> None:
    evidence = _pack(
        REFUTE_SNIPPETS,
        (0.8, 0.6),
        (EvidenceSource.openalex, EvidenceSource.semantic_scholar),
    )
    verdict = judge_claim(CLAIM, evidence)

    assert verdict.label is VerdictLabel.refute
    assert verdict.rubric_scores == {
        "relevance": 0.75,
        "coverage": 1.0,
        "stance_lexical": -1.0,
    }
    assert verdict.confidence == pytest.approx(0.85)
    assert verdict.rationale == (
        "Lexical cues lean refute (-1.00) with adequate relevance (0.75)."
    )
    assert "ev_0" in verdict.evidence_ids


def test_negated_support_phrase_and_opposite_direction_refute() -> None:
    aspirin = Claim(
        id="clm_aspirin",
        text="Aspirin reduces headache risk in adults.",
        source_work_id="https://openalex.org/W2",
        source_title="Aspirin",
    )
    negated = _pack(
        ("A review does not support that aspirin reduces headache risk in adults.",),
        (0.9,),
    )
    assert judge_claim(aspirin, negated).label is VerdictLabel.refute

    opposite = _pack(
        ("Physics-informed neural networks increase the error on the Burgers equation.",),
        (0.8,),
    )
    verdict = judge_claim(CLAIM, opposite)
    assert verdict.label is VerdictLabel.refute
    assert verdict.rubric_scores["stance_lexical"] == -1.0


def test_unrelated_pack_is_insufficient_even_with_high_scores() -> None:
    evidence = _pack(UNRELATED_SNIPPETS, (0.8, 0.6))
    verdict = judge_claim(CLAIM, evidence)

    assert verdict.label is VerdictLabel.insufficient
    assert verdict.rubric_scores["relevance"] == 0.75
    assert verdict.rubric_scores["coverage"] == 1.0
    assert verdict.rubric_scores["stance_lexical"] == 0.0
    assert "does not meet the support or refute rule" in verdict.rationale


def test_empty_pack_is_insufficient() -> None:
    verdict = judge_claim(CLAIM, [])
    assert verdict.label is VerdictLabel.insufficient
    assert verdict.evidence_ids == []
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.rubric_scores == {
        "relevance": 0.0,
        "coverage": 0.0,
        "stance_lexical": 0.0,
    }
    assert verdict.rationale.startswith("No evidence was packed")


def test_support_cues_without_ranker_scores_stay_insufficient() -> None:
    evidence = _pack(SUPPORT_SNIPPETS, (None, None))
    verdict = judge_claim(CLAIM, evidence)
    assert verdict.label is VerdictLabel.insufficient
    assert verdict.rubric_scores["relevance"] == 0.0
    assert verdict.rubric_scores["stance_lexical"] == 1.0


def test_one_supporting_snippet_does_not_clear_coverage() -> None:
    evidence = _pack(SUPPORT_SNIPPETS[:1], (0.9,))
    verdict = judge_claim(CLAIM, evidence)
    assert coverage_score(evidence) == 0.5
    assert coverage_score(evidence) < COVERAGE_HIGH
    assert verdict.label is VerdictLabel.insufficient


def test_two_snippets_from_one_catalog_can_support() -> None:
    evidence = _pack(
        SUPPORT_SNIPPETS,
        (0.8, 0.6),
        (EvidenceSource.openalex, EvidenceSource.openalex),
    )
    assert coverage_score(evidence) == COVERAGE_HIGH
    assert judge_claim(CLAIM, evidence).label is VerdictLabel.support


def test_mixed_stance_is_insufficient() -> None:
    evidence = _pack(
        (SUPPORT_SNIPPETS[0], REFUTE_SNIPPETS[0]),
        (0.8, 0.8),
    )
    verdict = judge_claim(CLAIM, evidence)
    assert verdict.rubric_scores["stance_lexical"] == 0.0
    assert verdict.label is VerdictLabel.insufficient


def test_title_cues_do_not_set_stance() -> None:
    evidence = [
        _evidence(
            "ev_title",
            UNRELATED_SNIPPETS[0],
            score=0.9,
            title=(
                "This study supports that physics-informed neural networks "
                "reduce the error on the Burgers equation."
            ),
        ),
        _evidence(
            "ev_other",
            UNRELATED_SNIPPETS[1],
            score=0.8,
            source=EvidenceSource.arxiv,
            title="We confirm the Burgers equation result.",
        ),
    ]
    assert stance_lexical_score(CLAIM.text, evidence) == 0.0
    assert judge_claim(CLAIM, evidence).label is VerdictLabel.insufficient


def test_relevance_is_the_average_of_max_and_mean() -> None:
    evidence = _pack(SUPPORT_SNIPPETS, (0.2, 1.4))
    assert relevance_score(evidence) == pytest.approx((1.0 + ((0.2 + 1.0) / 2.0)) / 2.0)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (RELEVANCE_HIGH, VerdictLabel.support),
        (RELEVANCE_HIGH - 0.01, VerdictLabel.insufficient),
    ],
)
def test_support_relevance_threshold(score: float, expected: VerdictLabel) -> None:
    evidence = _pack(SUPPORT_SNIPPETS, (score, score))
    assert relevance_score(evidence) == pytest.approx(score)
    assert judge_claim(CLAIM, evidence).label is expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (RELEVANCE_ADEQUATE, VerdictLabel.refute),
        (RELEVANCE_ADEQUATE - 0.01, VerdictLabel.insufficient),
    ],
)
def test_refute_relevance_threshold(score: float, expected: VerdictLabel) -> None:
    evidence = _pack(REFUTE_SNIPPETS[:1], (score,))
    verdict = judge_claim(CLAIM, evidence)
    assert verdict.rubric_scores["stance_lexical"] == -1.0
    assert abs(verdict.rubric_scores["stance_lexical"]) >= STANCE_LEAN
    assert verdict.label is expected


def test_claim_from_text_is_stable() -> None:
    first = claim_from_text(f"  {CLAIM_TEXT}  ")
    second = claim_from_text(CLAIM_TEXT)
    assert first == second
    assert first.id.startswith("clm_")
    assert len(first.id) == 20
    assert first.source_work_id == "claimforge:text"
    assert first.source_title == ""
    with pytest.raises(ValueError):
        claim_from_text("   ")


def test_llm_is_skipped_when_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(config: object, user_prompt: str) -> str:
        raise AssertionError("LLM judge should not be called")

    monkeypatch.setattr("claimforge.judge._chat_completion", explode)
    evidence = _pack(SUPPORT_SNIPPETS, (0.8, 0.6))
    assert judge_claim(CLAIM, evidence).label is VerdictLabel.support


def test_llm_requires_both_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(config: object, user_prompt: str) -> str:
        raise AssertionError("LLM judge should not be called")

    monkeypatch.setattr("claimforge.judge._chat_completion", explode)
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    assert judge_claim(CLAIM, _pack(REFUTE_SNIPPETS, (0.8, 0.6))).label is VerdictLabel.refute
    monkeypatch.delenv("CLAIMFORGE_LLM_API_KEY")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    assert judge_claim(CLAIM, []).label is VerdictLabel.insufficient


def test_provider_rules_skips_the_llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(config: object, user_prompt: str) -> str:
        raise AssertionError("LLM judge should not be called")

    monkeypatch.setattr("claimforge.judge._chat_completion", explode)
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("CLAIMFORGE_LLM_PROVIDER", "rules")
    assert judge_claim(CLAIM, _pack(SUPPORT_SNIPPETS, (0.8, 0.6))).label is VerdictLabel.support


def test_llm_path_uses_the_model_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("CLAIMFORGE_LLM_BASE_URL", "https://llm.example/v1")
    evidence = _pack(REFUTE_SNIPPETS, (0.8, 0.6))

    def fake_chat(config: object, user_prompt: str) -> str:
        assert getattr(config, "api_key") == "test-key"
        assert getattr(config, "model") == "gpt-test"
        assert getattr(config, "base_url") == "https://llm.example/v1"
        assert CLAIM_TEXT in user_prompt
        assert "ev_0" in user_prompt
        return json.dumps(
            {
                "label": "insufficient",
                "confidence": 0.42,
                "rationale": "The model withholds a side.",
                "evidence_ids": ["ev_missing", "ev_1"],
                "rubric_scores": {
                    "relevance": 0.4,
                    "coverage": 0.5,
                    "stance_lexical": 0.0,
                },
            }
        )

    monkeypatch.setattr("claimforge.judge._chat_completion", fake_chat)
    verdict = judge_claim(CLAIM, evidence)
    assert verdict.label is VerdictLabel.insufficient
    assert verdict.claim_id == CLAIM.id
    assert verdict.confidence == pytest.approx(0.42)
    assert verdict.rationale == "The model withholds a side."
    assert verdict.evidence_ids == ["ev_1"]
    assert verdict.rubric_scores["relevance"] == pytest.approx(0.4)
    assert rubric_verdict(CLAIM, evidence).label is VerdictLabel.refute


def test_llm_omitted_scores_fall_back_to_the_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    evidence = _pack(SUPPORT_SNIPPETS, (0.8, 0.6))

    def fake_chat(config: object, user_prompt: str) -> str:
        return json.dumps(
            {
                "label": "insufficient",
                "confidence": 0.33,
                "rationale": "Model label only.",
            }
        )

    monkeypatch.setattr("claimforge.judge._chat_completion", fake_chat)
    verdict = judge_claim(CLAIM, evidence)
    assert verdict.label is VerdictLabel.insufficient
    assert verdict.evidence_ids == ["ev_0", "ev_1"]
    assert verdict.rubric_scores == rubric_verdict(CLAIM, evidence).rubric_scores


def test_llm_failure_falls_back_to_the_rubric(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from claimforge.judge import LlmJudgeError

    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    evidence = _pack(REFUTE_SNIPPETS, (0.8, 0.6))

    def failing_chat(config: object, user_prompt: str) -> str:
        raise LlmJudgeError("down")

    monkeypatch.setattr("claimforge.judge._chat_completion", failing_chat)
    with caplog.at_level("WARNING"):
        verdict = judge_claim(CLAIM, evidence)
    assert verdict == rubric_verdict(CLAIM, evidence)
    assert verdict.label is VerdictLabel.refute
    assert "test-key" not in caplog.text


def test_invalid_llm_payload_falls_back_to_the_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    evidence = _pack(REFUTE_SNIPPETS, (0.8, 0.6))

    def fake_chat(config: object, user_prompt: str) -> str:
        return json.dumps(
            {
                "label": "support",
                "confidence": 0.9,
                "rationale": "Too sure.",
                "rubric_scores": {"relevance": 5},
            }
        )

    monkeypatch.setattr("claimforge.judge._chat_completion", fake_chat)
    assert judge_claim(CLAIM, evidence) == rubric_verdict(CLAIM, evidence)


def test_chat_completion_posts_openai_compatible_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from claimforge.extract import LlmConfig
    from claimforge.judge import _chat_completion

    captured: dict[str, object] = {}

    class _Response:
        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "{\"label\": \"insufficient\"}"}}]}
            ).encode()

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

    monkeypatch.setattr("claimforge.judge.urllib.request.urlopen", fake_urlopen)
    text = _chat_completion(
        LlmConfig(
            provider="openai",
            model="gpt-test",
            api_key="test-key",
            base_url="https://llm.example/v1",
        ),
        "Claim and evidence",
    )
    assert text == '{"label": "insufficient"}'
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-test"
    assert body["temperature"] == 0
    assert captured["timeout"] == 30
    messages = body["messages"]
    assert isinstance(messages, list)
    assert "insufficient" in messages[0]["content"]


def test_verify_cli_retrieves_then_prints_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(
        client: object,
        claim: Claim | str,
        *,
        per_source: int,
        top_k: int,
        mailto: str | None,
        s2_api_key: str | None,
        ranker: str = "auto",
        embedding_cache_dir: object = None,
    ) -> list[Evidence]:
        assert isinstance(claim, Claim)
        seen["claim_id"] = claim.id
        seen["text"] = claim.text
        seen["per_source"] = per_source
        seen["top_k"] = top_k
        seen["mailto"] = mailto
        seen["s2_api_key"] = s2_api_key
        seen["ranker"] = ranker
        seen["embedding_cache_dir"] = embedding_cache_dir
        return _pack(SUPPORT_SNIPPETS, (0.8, 0.6))

    monkeypatch.setenv("CLAIMFORGE_OPENALEX_MAILTO", "dev@example.com")
    monkeypatch.delenv("CLAIMFORGE_S2_API_KEY", raising=False)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)

    assert main(["verify", "--text", CLAIM_TEXT, "--top-k", "3", "--per-source", "4", "--ranker", "lexical"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["label"] == "support"
    assert payload["claim_id"] == claim_from_text(CLAIM_TEXT).id
    assert payload["evidence_ids"] == ["ev_0", "ev_1"]
    assert payload["rubric_scores"]["stance_lexical"] == 1.0
    assert seen == {
        "claim_id": claim_from_text(CLAIM_TEXT).id,
        "text": CLAIM_TEXT,
        "per_source": 4,
        "top_k": 3,
        "mailto": "dev@example.com",
        "s2_api_key": None,
        "ranker": "lexical",
        "embedding_cache_dir": Path("data/cache/embeddings"),
    }
    assert captured.err.strip() == "ranker: lexical"


def test_verify_cli_returns_one_when_retrieval_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_retrieve(
        client: object,
        claim: Claim | str,
        *,
        per_source: int,
        top_k: int,
        mailto: str | None,
        s2_api_key: str | None,
        ranker: str = "auto",
        embedding_cache_dir: object = None,
    ) -> list[Evidence]:
        raise EvidenceRetrievalError(["openalex: down", "arxiv: down", "semantic_scholar: down"])

    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)
    assert main(["verify", "--text", CLAIM_TEXT]) == 1
    assert "openalex: down" in capsys.readouterr().err


def test_verify_cli_prints_one_verdict_per_claim(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    other = Claim(
        id="clm_other",
        text="Transformers improve machine translation quality.",
        source_work_id="https://openalex.org/W9",
        source_title="Translation",
    )
    path = tmp_path / "claims.json"
    path.write_text(json.dumps([CLAIM.to_json_dict(), other.to_json_dict()]), encoding="utf-8")

    def fake_retrieve(
        client: object,
        claim: Claim | str,
        *,
        per_source: int,
        top_k: int,
        mailto: str | None,
        s2_api_key: str | None,
        ranker: str = "auto",
        embedding_cache_dir: object = None,
    ) -> list[Evidence]:
        assert isinstance(claim, Claim)
        if claim.id == CLAIM.id:
            return _pack(SUPPORT_SNIPPETS, (0.8, 0.6))
        return []

    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    assert main(["verify", "--claim-json", str(path), "--ranker", "lexical"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [item["claim_id"] for item in payload] == [CLAIM.id, other.id]
    assert [item["label"] for item in payload] == ["support", "insufficient"]
    assert captured.err.strip() == "ranker: lexical"


def test_judge_cli_reads_claim_and_evidence_json(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    claim_path = tmp_path / "claim.json"
    evidence_path = tmp_path / "evidence.json"
    claim_path.write_text(CLAIM.model_dump_json(), encoding="utf-8")
    evidence_path.write_text(
        json.dumps([item.to_json_dict() for item in _pack(REFUTE_SNIPPETS, (0.8, 0.6))]),
        encoding="utf-8",
    )
    assert main(["judge", "--claim-json", str(claim_path), "--evidence-json", str(evidence_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["claim_id"] == CLAIM.id
    assert payload["label"] == "refute"
    assert payload["rubric_scores"]["stance_lexical"] == -1.0


def test_judge_cli_accepts_one_evidence_object(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    claim_path = tmp_path / "claim.json"
    evidence_path = tmp_path / "evidence.json"
    claim_path.write_text(CLAIM.model_dump_json(), encoding="utf-8")
    evidence_path.write_text(
        json.dumps(_pack(REFUTE_SNIPPETS[:1], (0.8,))[0].to_json_dict()),
        encoding="utf-8",
    )
    assert main(["judge", "--claim-json", str(claim_path), "--evidence-json", str(evidence_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "refute"
    assert payload["evidence_ids"] == ["ev_0"]


def test_judge_cli_reads_a_claim_id_map(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    other = Claim(
        id="clm_other",
        text="Transformers improve machine translation quality.",
        source_work_id="https://openalex.org/W9",
        source_title="Translation",
    )
    claim_path = tmp_path / "claims.json"
    evidence_path = tmp_path / "packs.json"
    claim_path.write_text(
        json.dumps([CLAIM.to_json_dict(), other.to_json_dict()]),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                CLAIM.id: [item.to_json_dict() for item in _pack(SUPPORT_SNIPPETS, (0.8, 0.6))],
                other.id: [],
            }
        ),
        encoding="utf-8",
    )
    assert main(["judge", "--claim-json", str(claim_path), "--evidence-json", str(evidence_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["label"] for item in payload] == ["support", "insufficient"]


def test_judge_cli_pairs_packs_for_several_claims(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    other = Claim(
        id="clm_other",
        text="Transformers improve machine translation quality.",
        source_work_id="https://openalex.org/W9",
        source_title="Translation",
    )
    claim_path = tmp_path / "claims.json"
    evidence_path = tmp_path / "packs.json"
    claim_path.write_text(
        json.dumps([CLAIM.to_json_dict(), other.to_json_dict()]),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            [
                {
                    "claim_id": CLAIM.id,
                    "evidence": [item.to_json_dict() for item in _pack(SUPPORT_SNIPPETS, (0.8, 0.6))],
                },
                {"claim_id": other.id, "evidence": []},
            ]
        ),
        encoding="utf-8",
    )
    assert main(["judge", "--claim-json", str(claim_path), "--evidence-json", str(evidence_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["claim_id"] for item in payload] == [CLAIM.id, other.id]
    assert [item["label"] for item in payload] == ["support", "insufficient"]


@pytest.mark.parametrize(
    "argv",
    [
        ["verify"],
        ["verify", "--text", "   "],
        ["verify", "--text", "neural", "--top-k", "0"],
        ["judge"],
        ["judge", "--claim-json", "missing.json"],
    ],
)
def test_verify_and_judge_reject_bad_arguments(argv: list[str], tmp_path: Path) -> None:
    if argv and argv[-1] == "missing.json":
        argv = ["judge", "--claim-json", str(tmp_path / "missing.json"), "--evidence-json", str(tmp_path / "e.json")]
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
