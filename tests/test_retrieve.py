"""Evidence retrieval, dedupe, and the retrieve-evidence CLI. HTTP is mocked."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from claimforge.cli import main
from claimforge.http_cache import (
    CachedHttpClient,
    CachedResponse,
    HttpRequestError,
    TransportResponse,
)
from claimforge.literature import lexical_score, normalize_arxiv_id, normalize_doi
from claimforge.models import Claim, Evidence, EvidenceSource
from claimforge.openalex import EVIDENCE_SELECT
from claimforge.retrieve import (
    EvidenceRetrievalError,
    evidence_from_openalex_record,
    merge_evidence,
    retrieve_evidence,
)

CLAIM_TEXT = "Physics-informed neural networks reduce the error on the Burgers equation."

OPENALEX_BODY = json.dumps(
    {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "display_name": "Physics-informed neural networks",
                "doi": "https://doi.org/10.1000/pinn",
                "abstract_inverted_index": {
                    "Physics-informed": [0],
                    "networks": [1],
                    "reduce": [2],
                    "Burgers": [3],
                    "error": [4],
                },
            },
            {"display_name": "missing id"},
            {
                "id": "https://openalex.org/W2",
                "display_name": "   ",
                "abstract_inverted_index": None,
            },
        ]
    }
).encode()

ARXIV_BODY = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Physics-informed neural networks</title>
    <summary>Physics-informed neural networks reduce the Burgers error with a longer abstract than the catalog record.</summary>
    <arxiv:doi>https://doi.org/10.1000/PINN</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <title>Unrelated quantum chromodynamics review article</title>
    <summary>This note discusses lattice gauge theory.</summary>
  </entry>
</feed>
"""

S2_BODY = json.dumps(
    {
        "data": [
            {
                "paperId": "s2paper",
                "title": "Physics-informed neural networks",
                "abstract": "Short.",
                "url": "https://www.semanticscholar.org/paper/s2paper",
                "externalIds": {"DOI": "10.1000/pinn", "ArXiv": "1706.03762"},
            }
        ]
    }
).encode()


class FakeCatalog:
    def __init__(self, routes: list[tuple[str, CachedResponse | Exception]]) -> None:
        self.routes = routes
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> CachedResponse:
        self.urls.append(url)
        self.headers.append(dict(headers or {}))
        for needle, item in self.routes:
            if needle in url:
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(url)


def _response(status: int, body: bytes) -> CachedResponse:
    return CachedResponse(
        url="https://example.test",
        status_code=status,
        headers={},
        body=body,
        from_cache=False,
    )


def _routes(
    *,
    openalex: CachedResponse | Exception | None = None,
    arxiv: CachedResponse | Exception | None = None,
    s2: CachedResponse | Exception | None = None,
) -> list[tuple[str, CachedResponse | Exception]]:
    return [
        ("api.openalex.org", openalex if openalex is not None else _response(200, OPENALEX_BODY)),
        ("export.arxiv.org", arxiv if arxiv is not None else _response(200, ARXIV_BODY)),
        (
            "api.semanticscholar.org",
            s2 if s2 is not None else _response(200, S2_BODY),
        ),
    ]


def _evidence(
    *,
    source: EvidenceSource,
    title: str,
    snippet: str = "",
    work_id: str | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str = "https://example.test/paper",
) -> Evidence:
    return Evidence(
        id=f"ev_{source.value}_{title[:8]}",
        title=title,
        snippet=snippet,
        source=source,
        work_id=work_id,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
    )


def test_identifier_normalizers() -> None:
    assert normalize_doi("https://doi.org/10.1000/AbC") == "10.1000/AbC"
    assert normalize_doi("http://dx.doi.org/10.1000/AbC/") == "10.1000/AbC"
    assert normalize_doi("doi:10.1000/AbC") == "10.1000/AbC"
    assert normalize_doi("not-a-doi") is None
    assert normalize_arxiv_id("http://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert normalize_arxiv_id("https://arxiv.org/pdf/1706.03762v7.pdf") == "1706.03762"
    assert normalize_arxiv_id("arxiv:hep-th/9901001v1") == "hep-th/9901001"
    assert normalize_arxiv_id("Attention Is All You Need") is None


def test_openalex_record_maps_abstract_doi_and_drops_empty_works() -> None:
    record = json.loads(OPENALEX_BODY)["results"][0]
    evidence = evidence_from_openalex_record(record)
    assert evidence is not None
    assert evidence.source is EvidenceSource.openalex
    assert evidence.work_id == "https://openalex.org/W1"
    assert evidence.doi == "10.1000/pinn"
    assert evidence.snippet == "Physics-informed networks reduce Burgers error"
    assert evidence.url == "https://openalex.org/W1"
    assert evidence_from_openalex_record(json.loads(OPENALEX_BODY)["results"][2]) is None
    from_ids = evidence_from_openalex_record(
        {
            "id": "W9",
            "display_name": "From the ids block",
            "ids": {"doi": "https://doi.org/10.1000/ids", "arxiv": "https://arxiv.org/abs/2101.00001v2"},
        }
    )
    assert from_ids is not None
    assert from_ids.doi == "10.1000/ids"
    assert from_ids.arxiv_id == "2101.00001"
    assert from_ids.url == "https://openalex.org/W9"


def test_retrieve_evidence_merges_sources_from_mocked_http() -> None:
    client = FakeCatalog(_routes())
    evidence = retrieve_evidence(
        client,
        CLAIM_TEXT,
        per_source=2,
        top_k=5,
        mailto="dev@example.com",
    )

    assert len(evidence) == 2
    top = evidence[0]
    assert top.source is EvidenceSource.openalex
    assert top.doi == "10.1000/pinn"
    assert top.arxiv_id == "1706.03762"
    assert top.work_id == "https://openalex.org/W1"
    assert top.snippet.startswith("Physics-informed neural networks reduce")
    assert top.url == "https://openalex.org/W1"
    assert top.score is not None and top.score > evidence[1].score
    assert evidence[1].arxiv_id == "2101.00001"
    assert evidence[1].source is EvidenceSource.arxiv

    hosts = " ".join(client.urls)
    assert "api.openalex.org" in hosts
    assert "export.arxiv.org" in hosts
    assert "api.semanticscholar.org" in hosts
    openalex_url = next(url for url in client.urls if "api.openalex.org" in url)
    params = parse_qs(urlsplit(openalex_url).query)
    assert params["select"] == [EVIDENCE_SELECT]
    assert params["mailto"] == ["dev@example.com"]
    assert params["per_page"] == ["2"]
    assert all("dev@example.com" not in url for url in client.urls if "openalex" not in url)
    assert all("x-api-key" not in headers for headers in client.headers)


def test_retrieve_evidence_accepts_a_claim_and_limits_top_k() -> None:
    client = FakeCatalog(_routes())
    claim = Claim(
        id="clm_abc",
        text=CLAIM_TEXT,
        source_work_id="https://openalex.org/W9",
        source_title="PINN",
    )
    evidence = retrieve_evidence(client, claim, per_source=2, top_k=1)
    assert len(evidence) == 1
    assert "Burgers" in evidence[0].snippet or "Burgers" in evidence[0].title


def test_arxiv_406_keeps_other_sources() -> None:
    client = FakeCatalog(_routes(arxiv=_response(406, b"")))
    evidence = retrieve_evidence(client, CLAIM_TEXT, per_source=2, top_k=5)
    assert len(evidence) == 1
    assert evidence[0].source is EvidenceSource.openalex
    assert evidence[0].doi == "10.1000/pinn"


def test_semantic_scholar_401_does_not_drop_other_sources() -> None:
    client = FakeCatalog(_routes(s2=_response(401, b'{"message":"unauthorized"}')))
    evidence = retrieve_evidence(client, CLAIM_TEXT, per_source=2, top_k=5)
    assert evidence
    assert all(item.source is not EvidenceSource.semantic_scholar for item in evidence)


def test_one_provider_outage_keeps_the_rest() -> None:
    client = FakeCatalog(
        _routes(
            openalex=HttpRequestError("down", status_code=503, url="https://api.openalex.org/works"),
            s2=HttpRequestError("limited", status_code=500, url="https://api.semanticscholar.org"),
        )
    )
    evidence = retrieve_evidence(client, CLAIM_TEXT, per_source=2, top_k=5)
    assert evidence
    assert all(item.source is EvidenceSource.arxiv for item in evidence)


def test_all_provider_failures_raise() -> None:
    error = HttpRequestError("down", status_code=503, url="https://example.test")
    client = FakeCatalog(_routes(openalex=error, arxiv=error, s2=error))
    with pytest.raises(EvidenceRetrievalError) as caught:
        retrieve_evidence(client, CLAIM_TEXT, per_source=1, top_k=3)
    assert len(caught.value.errors) == 3


def test_merge_keeps_distinct_works_that_share_a_title() -> None:
    shared = "Physics-informed neural networks"
    packs = [
        [
            _evidence(
                source=EvidenceSource.openalex,
                title=shared,
                work_id="https://openalex.org/W1",
                url="https://openalex.org/W1",
            ),
            _evidence(
                source=EvidenceSource.openalex,
                title=shared,
                work_id="https://openalex.org/W2",
                url="https://openalex.org/W2",
            ),
        ]
    ]
    merged = merge_evidence(packs, query=CLAIM_TEXT, top_k=5)
    assert len(merged) == 2
    assert {item.work_id for item in merged} == {
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    }


def test_merge_title_fills_a_record_that_has_no_identifier() -> None:
    title = "Physics-informed neural networks"
    packs = [
        [
            _evidence(
                source=EvidenceSource.openalex,
                title=title,
                snippet="short",
                work_id="https://openalex.org/W1",
                doi="10.1000/pinn",
                url="https://openalex.org/W1",
            )
        ],
        [
            _evidence(
                source=EvidenceSource.arxiv,
                title=title,
                snippet="a much longer snippet about Burgers error",
                url="https://example.test/no-id",
            )
        ],
    ]
    merged = merge_evidence(packs, query=CLAIM_TEXT, top_k=5)
    assert len(merged) == 1
    assert merged[0].doi == "10.1000/pinn"
    assert merged[0].snippet.startswith("a much longer")
    assert merged[0].source is EvidenceSource.openalex


def test_lexical_score_prefers_title_matches() -> None:
    query = "Burgers equation neural networks"
    titled = lexical_score(query, "Neural networks for the Burgers equation", "")
    buried = lexical_score(query, "A survey", "neural networks and the Burgers equation appear here")
    assert titled > buried


def test_disk_cache_serves_every_provider(tmp_path: Path) -> None:
    class RoutingTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, headers: Mapping[str, str], timeout: float) -> TransportResponse:
            self.calls.append(url)
            if "api.openalex.org" in url:
                body = OPENALEX_BODY
            elif "export.arxiv.org" in url:
                body = ARXIV_BODY
            elif "api.semanticscholar.org" in url:
                body = S2_BODY
            else:
                raise AssertionError(url)
            return TransportResponse(status_code=200, headers={}, body=body)

    transport = RoutingTransport()
    client = CachedHttpClient(
        tmp_path,
        transport=transport,
        sleep=lambda _seconds: None,
        max_retries=0,
        jitter=0.0,
    )
    first = retrieve_evidence(client, CLAIM_TEXT, per_source=2, top_k=5)
    second = retrieve_evidence(client, CLAIM_TEXT, per_source=2, top_k=5)

    assert first == second
    assert first
    assert len(transport.calls) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_retrieve_evidence_cli_prints_json(
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
    ) -> list[Evidence]:
        seen["text"] = claim.text if isinstance(claim, Claim) else claim
        seen["per_source"] = per_source
        seen["top_k"] = top_k
        seen["mailto"] = mailto
        seen["s2_api_key"] = s2_api_key
        return [
            Evidence(
                id="ev_demo",
                title="Physics-informed neural networks",
                snippet="Burgers error falls.",
                source=EvidenceSource.openalex,
                work_id="https://openalex.org/W1",
                url="https://openalex.org/W1",
                score=0.75,
            )
        ]

    monkeypatch.setenv("CLAIMFORGE_OPENALEX_MAILTO", "dev@example.com")
    monkeypatch.delenv("CLAIMFORGE_S2_API_KEY", raising=False)
    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)

    assert main(["retrieve-evidence", "--text", CLAIM_TEXT, "--top-k", "3", "--per-source", "4"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "ev_demo"
    assert payload[0]["source"] == "openalex"
    assert payload[0]["score"] == 0.75
    assert seen == {
        "text": CLAIM_TEXT,
        "per_source": 4,
        "top_k": 3,
        "mailto": "dev@example.com",
        "s2_api_key": None,
    }


def test_retrieve_evidence_cli_reads_claim_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    claims = [
        {
            "id": "clm_one",
            "text": "We show that message passing helps.",
            "source_work_id": "https://openalex.org/W1",
            "source_title": "One",
        },
        {
            "id": "clm_two",
            "text": "Our method uses a residual loss.",
            "source_work_id": "https://openalex.org/W2",
            "source_title": "Two",
        },
    ]
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(claims), encoding="utf-8")
    seen: list[str] = []

    def fake_retrieve(
        client: object,
        claim: Claim | str,
        *,
        per_source: int,
        top_k: int,
        mailto: str | None,
        s2_api_key: str | None,
    ) -> list[Evidence]:
        assert isinstance(claim, Claim)
        seen.append(claim.id)
        return []

    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)
    assert main(["retrieve-evidence", "--claim-json", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert seen == ["clm_one", "clm_two"]
    assert payload == [
        {"claim_id": "clm_one", "evidence": []},
        {"claim_id": "clm_two", "evidence": []},
    ]


def test_retrieve_evidence_cli_single_claim_object_prints_evidence_array(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "claim.json"
    path.write_text(
        json.dumps(
            {
                "id": "clm_one",
                "text": CLAIM_TEXT,
                "source_work_id": "https://openalex.org/W1",
                "source_title": "PINN",
            }
        ),
        encoding="utf-8",
    )

    def fake_retrieve(
        client: object,
        claim: Claim | str,
        *,
        per_source: int,
        top_k: int,
        mailto: str | None,
        s2_api_key: str | None,
    ) -> list[Evidence]:
        return [
            Evidence(
                id="ev_one",
                title="PINN",
                source=EvidenceSource.arxiv,
                arxiv_id="2101.00001",
                url="https://arxiv.org/abs/2101.00001",
            )
        ]

    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)
    assert main(["retrieve-evidence", "--claim-json", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["arxiv_id"] == "2101.00001"
    assert "claim_id" not in payload[0]


def test_retrieve_evidence_cli_returns_one_when_every_source_fails(
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
    ) -> list[Evidence]:
        raise EvidenceRetrievalError(["openalex: down", "arxiv: down", "semantic_scholar: down"])

    monkeypatch.setattr("claimforge.cli.retrieve_evidence", fake_retrieve)
    assert main(["retrieve-evidence", "--text", "neural networks"]) == 1
    assert "openalex: down" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["retrieve-evidence"],
        ["retrieve-evidence", "--text", "neural", "--claim-json", "claims.json"],
        ["retrieve-evidence", "--text", "   "],
        ["retrieve-evidence", "--text", "neural", "--top-k", "0"],
        ["retrieve-evidence", "--text", "neural", "--per-source", "26"],
    ],
)
def test_retrieve_evidence_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
