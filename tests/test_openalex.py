"""Unit tests for OpenAlex URL building, parsing, and the smoke CLI."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from claimforge.cli import main
from claimforge.http_cache import CachedResponse
from claimforge.openalex import (
    DEFAULT_QUERY,
    EVIDENCE_SELECT,
    WORK_RECORD_SELECT,
    OpenAlexError,
    Work,
    build_works_search_url,
    format_works,
    search_work_records,
    search_works,
)


class FakeClient:
    def __init__(self, response: CachedResponse) -> None:
        self.response = response
        self.urls: list[str] = []

    def get(self, url: str) -> CachedResponse:
        self.urls.append(url)
        return self.response


def _cached(status: int, payload: object) -> CachedResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return CachedResponse(
        url="https://api.openalex.org/works",
        status_code=status,
        headers={},
        body=body,
        from_cache=False,
    )


def test_search_url_uses_default_smoke_query() -> None:
    url = build_works_search_url(DEFAULT_QUERY, per_page=3)
    parts = urlsplit(url)
    params = parse_qs(parts.query)
    assert parts.scheme == "https"
    assert parts.netloc == "api.openalex.org"
    assert parts.path == "/works"
    assert params["search"] == ["physics informed neural network"]
    assert params["per_page"] == ["3"]
    assert params["select"] == ["id,display_name"]
    assert "mailto" not in params


def test_search_url_includes_mailto_when_set() -> None:
    url = build_works_search_url("pin", per_page=1, mailto="dev@example.com")
    assert parse_qs(urlsplit(url).query)["mailto"] == ["dev@example.com"]


def test_search_url_rejects_empty_query_and_bad_page_size() -> None:
    with pytest.raises(ValueError):
        build_works_search_url("  ")
    with pytest.raises(ValueError):
        build_works_search_url("pin", per_page=0)


def test_search_works_parses_titles_and_ids() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "display_name": "Physics-informed neural networks",
            },
            {"id": "https://openalex.org/W2", "display_name": "  "},
            {"display_name": "missing id"},
            "not-an-object",
        ]
    }
    client = FakeClient(_cached(200, payload))

    works = search_works(client, "physics informed neural network", per_page=3)

    assert works == [
        Work(id="https://openalex.org/W1", title="Physics-informed neural networks"),
        Work(id="https://openalex.org/W2", title=""),
    ]
    assert "per_page=3" in client.urls[0]


def test_search_works_rejects_error_status_invalid_json_and_missing_results() -> None:
    with pytest.raises(OpenAlexError, match="HTTP 404"):
        search_works(FakeClient(_cached(404, {"results": []})), "pin")
    with pytest.raises(OpenAlexError, match="invalid JSON"):
        search_works(FakeClient(_cached(200, b"not-json")), "pin")
    with pytest.raises(OpenAlexError, match="missing results"):
        search_works(FakeClient(_cached(200, {"meta": {}})), "pin")


def test_search_work_records_requests_abstracts_and_keeps_indexes() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "display_name": "A title",
                "abstract_inverted_index": {"We": [0], "show": [1]},
            },
            {"display_name": "missing id"},
            "not-an-object",
        ]
    }
    client = FakeClient(_cached(200, payload))

    records = search_work_records(client, "pin", per_page=2)

    assert records == [
        {
            "id": "https://openalex.org/W1",
            "display_name": "A title",
            "abstract_inverted_index": {"We": [0], "show": [1]},
        }
    ]
    assert parse_qs(urlsplit(client.urls[0]).query)["select"] == [WORK_RECORD_SELECT]


def test_search_work_records_can_request_evidence_fields() -> None:
    client = FakeClient(_cached(200, {"results": []}))
    assert search_work_records(client, "pin", per_page=2, select=EVIDENCE_SELECT) == []
    assert parse_qs(urlsplit(client.urls[0]).query)["select"] == [EVIDENCE_SELECT]


def test_format_works_lists_titles_and_ids() -> None:
    text = format_works(
        "pin",
        [Work(id="https://openalex.org/W1", title="A title"), Work(id="W2", title="")],
    )
    assert text.splitlines() == [
        "OpenAlex works for: pin",
        "1. A title",
        "   id: https://openalex.org/W1",
        "2. (untitled)",
        "   id: W2",
    ]
    assert format_works("pin", []) == "OpenAlex works for: pin\nNo works returned."


def test_smoke_cli_prints_titles_and_exits_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_search(client: object, query: str, *, per_page: int, mailto: str | None) -> list[Work]:
        assert query == DEFAULT_QUERY
        assert per_page == 3
        assert mailto is None
        return [Work(id="https://openalex.org/W9", title="PINN review")]

    monkeypatch.setattr("claimforge.cli.search_works", fake_search)

    assert main(["smoke-openalex"]) == 0
    output = capsys.readouterr().out
    assert "PINN review" in output
    assert "https://openalex.org/W9" in output


def test_smoke_cli_passes_mailto_and_custom_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_search(client: object, query: str, *, per_page: int, mailto: str | None) -> list[Work]:
        seen["query"] = query
        seen["per_page"] = per_page
        seen["mailto"] = mailto
        return []

    monkeypatch.setenv("CLAIMFORGE_OPENALEX_MAILTO", "dev@example.com")
    monkeypatch.setattr("claimforge.cli.search_works", fake_search)

    assert main(["smoke-openalex", "--query", "graphs", "--per-page", "2"]) == 0
    assert seen == {"query": "graphs", "per_page": 2, "mailto": "dev@example.com"}
    assert "No works returned." in capsys.readouterr().out


def test_smoke_cli_returns_one_on_openalex_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_search(client: object, query: str, *, per_page: int, mailto: str | None) -> list[Work]:
        raise OpenAlexError("down")

    monkeypatch.setattr("claimforge.cli.search_works", fake_search)
    assert main(["smoke-openalex"]) == 1
    assert "down" in capsys.readouterr().err


def test_extract_claims_cli_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_records(
        client: object,
        query: str,
        *,
        per_page: int,
        mailto: str | None,
    ) -> list[dict[str, object]]:
        assert query == "graph neural network"
        assert per_page == 2
        assert mailto is None
        return [
            {
                "id": "https://openalex.org/W9",
                "display_name": "GNN paper",
                "abstract": "We show that message passing improves node accuracy on this benchmark.",
            }
        ]

    monkeypatch.setattr("claimforge.cli.search_work_records", fake_records)
    assert main(["extract-claims", "--query", "graph neural network", "--per-page", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["source_work_id"] == "https://openalex.org/W9"
    assert payload[0]["source_title"] == "GNN paper"
    assert payload[0]["claim_type"] == "result"
    assert "message passing" in payload[0]["text"]


def test_extract_claims_cli_returns_one_on_openalex_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_records(
        client: object,
        query: str,
        *,
        per_page: int,
        mailto: str | None,
    ) -> list[dict[str, object]]:
        raise OpenAlexError("down")

    monkeypatch.setattr("claimforge.cli.search_work_records", fake_records)
    assert main(["extract-claims", "--query", "pin"]) == 1
    assert "down" in capsys.readouterr().err


def test_smoke_cli_rejects_page_size_out_of_range(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["smoke-openalex", "--per-page", "0"])
    assert caught.value.code == 2
    assert "between 1 and 25" in capsys.readouterr().err
