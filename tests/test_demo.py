"""Offline demo command. No live catalog calls."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

from claimforge.cli import main
from claimforge.demo import BURGERS_TEXT, DEFAULT_GOLD_FIXTURE

VERIFY_SAMPLE = Path("docs/samples/verify.json")
EVAL_SAMPLE = Path("docs/samples/eval.json")


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("demo must stay offline")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.delenv("CLAIMFORGE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("CLAIMFORGE_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLAIMFORGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CLAIMFORGE_LLM_BASE_URL", raising=False)
    monkeypatch.setattr("claimforge.rank.embeddings_available", lambda: False)


def test_demo_prints_support_and_a_perfect_eval(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_network(monkeypatch)
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    text = captured.out
    assert "ClaimForge 0.1.0 offline demo" in text
    assert "Ranker: lexical" in text
    assert BURGERS_TEXT in text
    assert "label       support" in text
    assert "ev_burgers_oa" in text
    assert "ev_burgers_ax" in text
    assert "accuracy: 1.0000" in text
    assert "mismatches: none" in text
    assert str(DEFAULT_GOLD_FIXTURE) in text


def test_demo_ignores_llm_settings_and_restores_them(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_network(monkeypatch)
    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    assert main(["demo"]) == 0
    assert "label       support" in capsys.readouterr().out
    assert os.environ["CLAIMFORGE_LLM_API_KEY"] == "sk-test"
    assert os.environ["CLAIMFORGE_LLM_MODEL"] == "gpt-test"


def test_demo_exits_when_the_cassette_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _block_network(monkeypatch)
    monkeypatch.setenv("CLAIMFORGE_FIXTURE_DIR", str(tmp_path))
    code = main(["demo"])
    captured = capsys.readouterr()
    assert code == 1
    assert "label       insufficient" in captured.out
    assert "no evidence" in captured.err


def test_demo_rejects_a_missing_fixture(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--fixture", str(tmp_path / "missing.json")])
    assert exc.value.code == 2


def test_committed_samples_match_offline_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_network(monkeypatch)
    assert (
        main(
            [
                "verify",
                "--offline",
                "--ranker",
                "lexical",
                "--text",
                BURGERS_TEXT,
            ]
        )
        == 0
    )
    verify_out = capsys.readouterr().out
    payload = json.loads(verify_out)
    assert payload["label"] == "support"
    assert VERIFY_SAMPLE.read_text(encoding="utf-8") == verify_out

    assert main(["eval", "--fixture", str(DEFAULT_GOLD_FIXTURE), "--format", "json"]) == 0
    eval_out = capsys.readouterr().out
    report = json.loads(eval_out)
    assert report["accuracy"] == 1.0
    assert report["n"] == 13
    assert EVAL_SAMPLE.read_text(encoding="utf-8") == eval_out
