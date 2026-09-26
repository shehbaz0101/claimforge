"""Offline verify and retrieve. No live catalog calls."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import pytest

from claimforge.cli import main
from claimforge.http_cache import CachedHttpClient, TransportResponse
from claimforge.judge import claim_from_text
from claimforge.models import Evidence, EvidenceSource
from claimforge.retrieve import retrieve_evidence

CLAIM_TEXT = "Physics-informed neural networks reduce the error on the Burgers equation."
OTHER_TEXT = "Glaciers in this fixture were not recorded."
CASSETTE = Path("data/fixtures/cassettes/burgers.json")


class BoomTransport:
    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        raise AssertionError(f"offline mode contacted the network: {url}")


class ScriptedTransport:
    def __init__(self, body_for_host: dict[str, bytes]) -> None:
        self.body_for_host = body_for_host
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        self.calls.append(url)
        for host, body in self.body_for_host.items():
            if host in url:
                return TransportResponse(status_code=200, headers={}, body=body)
        raise AssertionError(url)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline test must stay offline")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.delenv("CLAIMFORGE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("CLAIMFORGE_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLAIMFORGE_LLM_PROVIDER", raising=False)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)


def test_offline_cassette_is_ranked_without_a_transport(tmp_path: Path) -> None:
    client = CachedHttpClient(tmp_path, transport=BoomTransport(), offline=True)
    evidence = retrieve_evidence(
        client,
        CLAIM_TEXT,
        per_source=2,
        top_k=5,
        ranker="lexical",
        offline=True,
    )
    assert {item.id for item in evidence} == {"ev_burgers_oa", "ev_burgers_ax"}
    assert evidence[0].score is not None and evidence[0].score > 0


def test_offline_env_without_a_cassette_returns_an_empty_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CLAIMFORGE_OFFLINE", "1")
    monkeypatch.setenv("CLAIMFORGE_FIXTURE_DIR", str(tmp_path))
    client = CachedHttpClient(tmp_path / "cache", transport=BoomTransport())
    evidence = retrieve_evidence(client, OTHER_TEXT, per_source=2, top_k=5, ranker="lexical")
    assert evidence == []


def test_offline_false_ignores_a_matching_cassette(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        {
            "api.openalex.org": b'{"results":[]}',
            "export.arxiv.org": b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
""",
            "api.semanticscholar.org": b'{"data":[]}',
        }
    )
    client = CachedHttpClient(tmp_path, transport=transport, offline=False)
    evidence = retrieve_evidence(
        client,
        CLAIM_TEXT,
        per_source=1,
        top_k=3,
        ranker="lexical",
        offline=False,
    )
    assert evidence == []
    assert len(transport.calls) == 3


def test_offline_uses_a_warm_cache_when_no_cassette_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CLAIMFORGE_FIXTURE_DIR", str(tmp_path / "empty"))
    body = json.dumps(
        {
            "results": [
                {
                    "id": "https://openalex.org/W9",
                    "display_name": "Glacier fixture",
                    "abstract_inverted_index": {"Glaciers": [0], "retreat": [1]},
                }
            ]
        }
    ).encode()
    warm = ScriptedTransport(
        {
            "api.openalex.org": body,
            "export.arxiv.org": b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
""",
            "api.semanticscholar.org": b'{"data":[]}',
        }
    )
    cache = tmp_path / "cache"
    online = CachedHttpClient(cache, transport=warm, offline=False)
    retrieve_evidence(online, OTHER_TEXT, per_source=1, top_k=3, ranker="lexical", offline=False)
    calls_after_warm = len(warm.calls)

    offline = CachedHttpClient(cache, transport=BoomTransport(), offline=True)
    evidence = retrieve_evidence(
        offline,
        OTHER_TEXT,
        per_source=1,
        top_k=3,
        ranker="lexical",
        offline=True,
    )
    assert calls_after_warm == len(warm.calls)
    assert any(item.work_id == "https://openalex.org/W9" for item in evidence)


def test_cassette_matches_a_claim_id(tmp_path: Path) -> None:
    path = tmp_path / "pack.json"
    path.write_text(
        json.dumps(
            {
                "id": "clm_custom",
                "text": "some other sentence",
                "evidence": [
                    {
                        "id": "ev_custom",
                        "title": "Custom",
                        "snippet": "A stored snippet.",
                        "source": "semantic_scholar",
                        "url": "https://example.test/custom",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from claimforge.cassettes import load_matching_cassette
    from claimforge.models import Claim

    claim = Claim(
        id="clm_custom",
        text="text that does not match the cassette sentence",
        source_work_id="claimforge:text",
        source_title="",
    )
    matched = load_matching_cassette(claim, tmp_path)
    assert matched is not None
    assert matched[0].id == "ev_custom"
    assert load_matching_cassette("no such claim", tmp_path) is None


def test_verify_offline_cli_uses_the_cassette_and_stays_offline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_network(monkeypatch)
    assert CASSETTE.is_file()
    code = main(
        [
            "verify",
            "--offline",
            "--ranker",
            "lexical",
            "--text",
            CLAIM_TEXT,
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["claim_id"] == claim_from_text(CLAIM_TEXT).id
    assert set(payload["evidence_ids"]) == {"ev_burgers_oa", "ev_burgers_ax"}
    assert payload["label"] in {"support", "refute", "insufficient"}
    assert captured.err.strip() == "ranker: lexical"


def test_verify_offline_without_a_pack_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _block_network(monkeypatch)
    monkeypatch.setenv("CLAIMFORGE_FIXTURE_DIR", str(tmp_path))
    code = main(
        [
            "verify",
            "--offline",
            "--ranker",
            "lexical",
            "--text",
            OTHER_TEXT,
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "insufficient"
    assert payload["evidence_ids"] == []
    assert "No evidence was packed" in payload["rationale"]


def test_retrieve_offline_cli_prints_the_cassette(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_network(monkeypatch)
    assert main(["retrieve-evidence", "--offline", "--ranker", "lexical", "--text", CLAIM_TEXT]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["id"] for item in payload} == {"ev_burgers_oa", "ev_burgers_ax"}
    assert {item["source"] for item in payload} == {
        EvidenceSource.openalex.value,
        EvidenceSource.arxiv.value,
    }


def test_offline_env_flag_on_verify(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _block_network(monkeypatch)
    monkeypatch.setenv("CLAIMFORGE_OFFLINE", "1")
    monkeypatch.setenv("CLAIMFORGE_FIXTURE_DIR", str(tmp_path))
    assert main(["verify", "--ranker", "lexical", "--text", OTHER_TEXT, "--cache-dir", str(tmp_path / "cache")]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "insufficient"


def test_bad_cassette_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    from claimforge.cassettes import load_matching_cassette

    assert load_matching_cassette(CLAIM_TEXT, tmp_path) is None


def test_committed_cassette_evidence_validates() -> None:
    payload = json.loads(CASSETTE.read_text(encoding="utf-8"))
    evidence = [Evidence.model_validate(item) for item in payload["evidence"]]
    assert {item.source for item in evidence} == {EvidenceSource.openalex, EvidenceSource.arxiv}
