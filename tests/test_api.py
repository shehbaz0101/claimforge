"""HTTP API. TestClient only. No catalog calls and no model download."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimforge import __version__
from claimforge.api import RANKER_HEADER, VERIFY_LIMITER, VERIFY_RATE_DETAIL, app
from claimforge.cli import main
from claimforge.judge import claim_from_text
from claimforge.models import Claim, Evidence, EvidenceSource
from claimforge.retrieve import EvidenceRetrievalError

FIXTURE = Path(__file__).parent / "fixtures" / "gold_claims.json"
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


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("API test must stay offline")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.setattr("claimforge.judge._chat_completion", explode)
    monkeypatch.setattr("claimforge.extract._chat_completion", explode)
    VERIFY_LIMITER.reset()
    for name in (
        "CLAIMFORGE_LLM_API_KEY",
        "CLAIMFORGE_LLM_MODEL",
        "CLAIMFORGE_LLM_PROVIDER",
        "CLAIMFORGE_LLM_BASE_URL",
        "CLAIMFORGE_OPENALEX_MAILTO",
        "CLAIMFORGE_S2_API_KEY",
        "CLAIMFORGE_OFFLINE",
        "CLAIMFORGE_FIXTURE_DIR",
        "CLAIMFORGE_VERIFY_RATE_LIMIT",
        "CLAIMFORGE_VERIFY_RATE_WINDOW_S",
        "CLAIMFORGE_HTTP_MIN_INTERVAL_S",
        "CLAIMFORGE_SOURCE_FAILURE_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _evidence(evidence_id: str, snippet: str, *, score: float, source: EvidenceSource) -> Evidence:
    return Evidence(
        id=evidence_id,
        title="Related work",
        snippet=snippet,
        source=source,
        url=f"https://example.test/{evidence_id}",
        score=score,
    )


def _pack(snippets: tuple[str, ...], scores: tuple[float, ...]) -> list[Evidence]:
    sources = (EvidenceSource.openalex, EvidenceSource.arxiv)
    return [
        _evidence(f"ev_{index}", snippet, score=score, source=sources[index])
        for index, (snippet, score) in enumerate(zip(snippets, scores, strict=True))
    ]


def test_health_reports_the_package_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_openapi_lists_the_day7_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) >= {"/health", "/verify", "/judge", "/eval"}
    assert "get" in paths["/health"]
    assert "post" in paths["/verify"]
    assert "post" in paths["/judge"]
    assert "post" in paths["/eval"]


def test_verify_retrieves_with_the_optional_ranker_and_returns_a_verdict(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(
        http: object,
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
        seen["cache_dir"] = getattr(http, "cache_dir", None)
        seen["embedding_cache_dir"] = embedding_cache_dir
        return _pack(SUPPORT_SNIPPETS, (0.8, 0.6))

    monkeypatch.setenv("CLAIMFORGE_OPENALEX_MAILTO", "dev@example.com")
    monkeypatch.setenv("CLAIMFORGE_S2_API_KEY", "test-s2")
    monkeypatch.setattr("claimforge.api.retrieve_evidence", fake_retrieve)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)

    response = client.post(
        "/verify",
        json={
            "text": f"  {CLAIM_TEXT}  ",
            "ranker": "lexical",
            "top_k": 3,
            "per_source": 4,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    assert response.status_code == 200
    assert response.headers[RANKER_HEADER] == "lexical"
    body = response.json()
    assert body["label"] == "support"
    assert body["claim_id"] == claim_from_text(CLAIM_TEXT).id
    assert body["evidence_ids"] == ["ev_0", "ev_1"]
    assert body["rubric_scores"]["stance_lexical"] == 1.0
    assert seen == {
        "claim_id": claim_from_text(CLAIM_TEXT).id,
        "text": CLAIM_TEXT,
        "per_source": 4,
        "top_k": 3,
        "mailto": "dev@example.com",
        "s2_api_key": "test-s2",
        "ranker": "lexical",
        "cache_dir": tmp_path / "cache",
        "embedding_cache_dir": tmp_path / "cache" / "embeddings",
    }


def test_verify_defaults_the_ranker_to_auto(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(
        http: object,
        claim: Claim | str,
        *,
        per_source: int,
        top_k: int,
        mailto: str | None,
        s2_api_key: str | None,
        ranker: str = "auto",
        embedding_cache_dir: object = None,
    ) -> list[Evidence]:
        seen["ranker"] = ranker
        seen["top_k"] = top_k
        seen["per_source"] = per_source
        seen["mailto"] = mailto
        seen["s2_api_key"] = s2_api_key
        return []

    monkeypatch.setattr("claimforge.api.retrieve_evidence", fake_retrieve)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    response = client.post("/verify", json={"text": CLAIM_TEXT})
    assert response.status_code == 200
    assert response.json()["label"] == "insufficient"
    assert seen["ranker"] == "auto"
    assert seen["top_k"] == 8
    assert seen["per_source"] == 8
    assert seen["mailto"] is None
    assert seen["s2_api_key"] is None
    assert response.headers[RANKER_HEADER] == "lexical"


def test_verify_maps_retrieval_failure_to_502(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_retrieve(*_args: object, **_kwargs: object) -> list[Evidence]:
        raise EvidenceRetrievalError(["openalex: down", "arxiv: down", "semantic_scholar: down"])

    monkeypatch.setattr("claimforge.api.retrieve_evidence", fake_retrieve)
    response = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})
    assert response.status_code == 502
    assert "openalex: down" in response.json()["detail"]
    assert RANKER_HEADER not in response.headers


def test_verify_unexpected_failure_is_json_502(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_retrieve(*_args: object, **_kwargs: object) -> list[Evidence]:
        raise RuntimeError("provider blew up")

    monkeypatch.setattr("claimforge.api.retrieve_evidence", fake_retrieve)
    response = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})
    assert response.status_code == 502
    assert response.json() == {"detail": "evidence retrieval failed"}


def test_verify_rate_limit_returns_429_json(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake_retrieve(*_args: object, **_kwargs: object) -> list[Evidence]:
        calls["n"] += 1
        return []

    clock = {"now": 1_000.0}
    monkeypatch.setattr(VERIFY_LIMITER, "_clock", lambda: clock["now"])
    monkeypatch.setenv("CLAIMFORGE_VERIFY_RATE_LIMIT", "2")
    monkeypatch.setenv("CLAIMFORGE_VERIFY_RATE_WINDOW_S", "60")
    monkeypatch.setattr("claimforge.api.retrieve_evidence", fake_retrieve)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    VERIFY_LIMITER.reset()

    first = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})
    second = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})
    blocked = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == VERIFY_RATE_DETAIL
    assert blocked.json()["retry_after_s"] == 60.0
    assert blocked.headers["retry-after"] == "60"
    assert calls["n"] == 2

    health = client.get("/health")
    judge = client.post("/judge", json={"claim": CLAIM.to_json_dict(), "evidence": []})
    assert health.status_code == 200
    assert judge.status_code == 200
    assert calls["n"] == 2

    clock["now"] += 60
    again = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})
    assert again.status_code == 200
    assert calls["n"] == 3


def test_verify_rate_limit_can_be_disabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAIMFORGE_VERIFY_RATE_LIMIT", "0")
    monkeypatch.setattr("claimforge.api.retrieve_evidence", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    VERIFY_LIMITER.reset()
    for _ in range(3):
        response = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "lexical"})
        assert response.status_code == 200


def test_verify_offline_env_does_not_use_the_network(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CLAIMFORGE_OFFLINE", "1")
    monkeypatch.setenv("CLAIMFORGE_FIXTURE_DIR", str(tmp_path))
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    response = client.post(
        "/verify",
        json={
            "text": "Glaciers in this fixture were not recorded.",
            "ranker": "lexical",
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "insufficient"
    assert body["evidence_ids"] == []
    assert response.headers[RANKER_HEADER] == "lexical"


def test_verify_offline_env_uses_the_committed_cassette(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAIMFORGE_OFFLINE", "1")
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    response = client.post(
        "/verify",
        json={"text": CLAIM_TEXT, "ranker": "lexical"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["evidence_ids"]) == {"ev_burgers_oa", "ev_burgers_ax"}
    assert response.headers[RANKER_HEADER] == "lexical"


def test_verify_rejects_a_missing_embedding_ranker_before_catalog_calls(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)
    response = client.post("/verify", json={"text": CLAIM_TEXT, "ranker": "embeddings"})
    assert response.status_code == 422
    assert "sentence-transformers" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "   "},
        {"text": CLAIM_TEXT, "ranker": "bm25"},
        {"text": CLAIM_TEXT, "top_k": 0},
        {"text": CLAIM_TEXT, "per_source": 26},
        {"text": CLAIM_TEXT, "cache_dir": "  "},
        {"text": CLAIM_TEXT, "extra": True},
    ],
)
def test_verify_rejects_bad_bodies(client: TestClient, payload: dict[str, object]) -> None:
    response = client.post("/verify", json=payload)
    assert response.status_code == 422


def test_judge_scores_a_packed_claim_without_retrieval(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> list[Evidence]:
        raise AssertionError("judge must not retrieve")

    monkeypatch.setattr("claimforge.api.retrieve_evidence", explode)
    evidence = _pack(REFUTE_SNIPPETS, (0.8, 0.6))
    response = client.post(
        "/judge",
        json={
            "claim": CLAIM.to_json_dict(),
            "evidence": [item.to_json_dict() for item in evidence],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == CLAIM.id
    assert body["label"] == "refute"
    assert body["evidence_ids"] == ["ev_0", "ev_1"]
    assert body["rubric_scores"]["stance_lexical"] == -1.0


def test_judge_empty_pack_is_insufficient(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> list[Evidence]:
        raise AssertionError("judge must not retrieve")

    monkeypatch.setattr("claimforge.api.retrieve_evidence", explode)
    response = client.post("/judge", json={"claim": CLAIM.to_json_dict(), "evidence": []})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "insufficient"
    assert body["evidence_ids"] == []


def test_judge_rejects_a_bad_claim(client: TestClient) -> None:
    response = client.post("/judge", json={"claim": {"text": CLAIM_TEXT}, "evidence": []})
    assert response.status_code == 422
    missing = client.post("/judge", json={"evidence": []})
    assert missing.status_code == 422


def test_eval_scores_a_fixture_path_offline(client: TestClient) -> None:
    response = client.post("/eval", json={"fixture": str(FIXTURE)})
    assert response.status_code == 200
    body = response.json()
    assert body["n"] == 13
    assert body["accuracy"] == 1.0
    assert body["agreement_rate"] == 1.0
    assert body["judge"] == "rubric"
    assert body["fixture"] == str(FIXTURE)
    assert body["meets_threshold"] is True


def test_eval_scores_inline_items_and_a_directory(
    client: TestClient,
    tmp_path: Path,
) -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    items = raw["items"]
    assert isinstance(items, list)
    one = client.post("/eval", json={"items": [items[0]], "min_accuracy": 0.0})
    assert one.status_code == 200
    assert one.json()["n"] == 1
    assert one.json()["fixture"] == "inline"
    assert one.json()["accuracy"] == 1.0
    assert one.json()["meets_threshold"] is True

    (tmp_path / "a.json").write_text(json.dumps({"items": items[:6]}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"items": items[6:]}), encoding="utf-8")
    (tmp_path / "skip.txt").write_text("not json", encoding="utf-8")
    directory = client.post("/eval", json={"fixture": str(tmp_path)})
    assert directory.status_code == 200
    assert directory.json()["n"] == len(items)
    assert directory.json()["accuracy"] == 1.0
    assert directory.json()["fixture"] == str(tmp_path)


def test_eval_reports_a_miss_without_failing_the_http_call(
    client: TestClient,
    tmp_path: Path,
) -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    items = raw["items"]
    assert isinstance(items, list)
    first = json.loads(json.dumps(items[0]))
    first["expected_label"] = "refute" if first["expected_label"] != "refute" else "support"
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"items": [first]}), encoding="utf-8")
    response = client.post("/eval", json={"fixture": str(path), "min_accuracy": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["meets_threshold"] is False
    assert body["accuracy"] == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"fixture": str(FIXTURE), "items": []},
        {"fixture": "   "},
        {"items": []},
        {"fixture": str(FIXTURE), "min_accuracy": 1.5},
        {"items": [{"id": "x"}]},
    ],
)
def test_eval_rejects_bad_bodies(client: TestClient, payload: dict[str, object]) -> None:
    response = client.post("/eval", json=payload)
    assert response.status_code == 422


def test_eval_rejects_a_missing_fixture(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/eval", json={"fixture": str(tmp_path / "missing.json")})
    assert response.status_code == 422
    assert "could not read" in response.json()["detail"]


def test_serve_invokes_uvicorn_without_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, object] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        seen["target"] = target
        seen["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert main(["serve"]) == 0
    assert seen["target"] == "claimforge.api:app"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is False

    assert main(["serve", "--host", "0.0.0.0", "--port", "9001", "--reload"]) == 0
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9001
    assert kwargs["reload"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--port", "0"],
        ["serve", "--port", "70000"],
        ["serve", "--host", " "],
    ],
)
def test_serve_rejects_a_bad_bind(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
