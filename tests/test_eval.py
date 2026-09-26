"""Offline gold-claim eval. No network."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pytest

from claimforge.cli import main
from claimforge.eval import (
    DEFAULT_MIN_ACCURACY,
    GoldFixtureError,
    GoldItem,
    evaluate_gold,
    format_eval_table,
    load_gold_fixture,
)
from claimforge.models import Claim, Evidence, Verdict, VerdictLabel

FIXTURE = Path(__file__).parent / "fixtures" / "gold_claims.json"

KNOWN_LABELS = {
    "gold_support_burgers": VerdictLabel.support,
    "gold_support_statins_same_catalog": VerdictLabel.support,
    "gold_refute_aspirin": VerdictLabel.refute,
    "gold_refute_burgers_opposite": VerdictLabel.refute,
    "gold_insufficient_empty": VerdictLabel.insufficient,
    "gold_insufficient_low_coverage": VerdictLabel.insufficient,
    "gold_insufficient_weak_refute": VerdictLabel.insufficient,
    "gold_insufficient_below_support_bar": VerdictLabel.insufficient,
}


def _claim(claim_id: str) -> Claim:
    return Claim(
        id=claim_id,
        text="Example claim text for the metrics test.",
        source_work_id="claimforge:gold",
        source_title="Metrics",
    )


def _gold(item_id: str, label: VerdictLabel) -> GoldItem:
    return GoldItem(
        id=item_id,
        expected_label=label,
        claim=_claim(f"clm_{item_id}"),
        evidence=(),
    )


def _stub_verdict(claim: Claim, label: VerdictLabel) -> Verdict:
    return Verdict(
        claim_id=claim.id,
        label=label,
        confidence=0.5,
        rationale="Stub verdict for the metrics test.",
        evidence_ids=[],
        rubric_scores={"relevance": 0.0, "coverage": 0.0, "stance_lexical": 0.0},
    )


def _flip_first_label(raw: dict[str, object]) -> dict[str, object]:
    items = raw["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    current = first["expected_label"]
    first["expected_label"] = "refute" if current != "refute" else "support"
    return raw


def test_gold_fixture_is_a_small_offline_set() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    items = raw["items"]
    assert 8 <= len(items) <= 15
    labels = {item["expected_label"] for item in items}
    assert labels == {"support", "refute", "insufficient"}
    assert "OpenAlex" in raw["description"]
    for item in items:
        assert isinstance(item["evidence"], list)
        assert item["note"].strip()
        for row in item["evidence"]:
            assert row["url"].startswith("https://example.test/gold/")
    text = FIXTURE.read_text(encoding="utf-8")
    assert "api_key" not in text.casefold()
    assert "CLAIMFORGE_" not in text


def test_eval_matches_known_gold_cases_and_reports_metrics() -> None:
    items = load_gold_fixture(FIXTURE)
    report = evaluate_gold(items, fixture=str(FIXTURE))

    assert report.n == len(items)
    assert report.judge == "rubric"
    assert report.correct == report.n
    assert report.accuracy == 1.0
    assert report.agreement_rate == report.accuracy
    assert report.macro_f1 == 1.0
    assert report.min_accuracy == DEFAULT_MIN_ACCURACY
    assert report.meets_threshold is True
    by_id = {item.id: item for item in report.items}
    assert set(KNOWN_LABELS) <= set(by_id)
    for item_id, label in KNOWN_LABELS.items():
        assert by_id[item_id].expected is label
        assert by_id[item_id].predicted is label
        assert by_id[item_id].match is True
        assert 0.0 <= by_id[item_id].confidence <= 1.0
    assert report.per_label["support"].f1 == 1.0
    assert report.per_label["refute"].f1 == 1.0
    assert report.per_label["insufficient"].f1 == 1.0
    assert report.confusion["support"]["support"] == report.per_label["support"].gold
    assert report.confusion["refute"]["refute"] == report.per_label["refute"].gold
    assert report.confusion["insufficient"]["insufficient"] == report.per_label["insufficient"].gold
    table = format_eval_table(report)
    assert "accuracy: 1.0000" in table
    assert "agreement: 1.0000" in table
    assert "mismatches: none" in table
    assert "support" in table and "refute" in table and "insufficient" in table


def test_partial_predictions_set_accuracy_f1_and_agreement() -> None:
    items = (
        _gold("a", VerdictLabel.support),
        _gold("b", VerdictLabel.support),
        _gold("c", VerdictLabel.refute),
        _gold("d", VerdictLabel.insufficient),
    )
    predicted = {
        "clm_a": VerdictLabel.support,
        "clm_b": VerdictLabel.refute,
        "clm_c": VerdictLabel.refute,
        "clm_d": VerdictLabel.insufficient,
    }

    def judge(claim: Claim, evidence: Sequence[Evidence]) -> Verdict:
        assert evidence == ()
        return _stub_verdict(claim, predicted[claim.id])

    report = evaluate_gold(items, judge=judge, judge_name="stub", min_accuracy=1.0)
    assert report.n == 4
    assert report.correct == 3
    assert report.accuracy == pytest.approx(0.75)
    assert report.agreement_rate == report.accuracy
    assert report.meets_threshold is False
    assert report.per_label["support"].precision == pytest.approx(1.0)
    assert report.per_label["support"].recall == pytest.approx(0.5)
    assert report.per_label["support"].f1 == pytest.approx(2.0 / 3.0)
    assert report.per_label["refute"].precision == pytest.approx(0.5)
    assert report.per_label["refute"].recall == pytest.approx(1.0)
    assert report.per_label["refute"].f1 == pytest.approx(2.0 / 3.0)
    assert report.per_label["insufficient"].f1 == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(7.0 / 9.0)
    assert report.confusion["support"]["refute"] == 1
    payload = report.to_json_dict()
    assert payload["accuracy"] == 0.75
    assert payload["agreement_rate"] == 0.75
    assert payload["per_label"]["support"]["f1"] == 0.6667
    assert payload["macro_f1"] == 0.7778
    assert payload["judge"] == "stub"
    missed = [item for item in report.items if not item.match]
    assert [item.id for item in missed] == ["b"]
    assert "expected support predicted refute" in format_eval_table(report)


def test_swapped_labels_score_zero_f1() -> None:
    items = (
        _gold("a", VerdictLabel.support),
        _gold("b", VerdictLabel.refute),
    )
    predicted = {
        "clm_a": VerdictLabel.refute,
        "clm_b": VerdictLabel.support,
    }

    def judge(claim: Claim, evidence: Sequence[Evidence]) -> Verdict:
        return _stub_verdict(claim, predicted[claim.id])

    report = evaluate_gold(items, judge=judge, judge_name="stub")
    assert report.accuracy == 0.0
    assert report.agreement_rate == 0.0
    assert report.per_label["support"].precision == 0.0
    assert report.per_label["support"].recall == 0.0
    assert report.per_label["support"].f1 == 0.0
    assert report.per_label["refute"].f1 == 0.0
    assert report.macro_f1 == 0.0


def test_unused_labels_are_left_out_of_macro_f1() -> None:
    items = (_gold("a", VerdictLabel.support), _gold("b", VerdictLabel.support))

    def judge(claim: Claim, evidence: Sequence[Evidence]) -> Verdict:
        return _stub_verdict(claim, VerdictLabel.support)

    report = evaluate_gold(items, judge=judge, judge_name="stub")
    assert report.per_label["refute"].gold == 0
    assert report.per_label["refute"].predicted == 0
    assert report.per_label["refute"].f1 is None
    assert report.per_label["insufficient"].f1 is None
    assert report.macro_f1 == 1.0
    assert "n/a" in format_eval_table(report)


def test_threshold_is_inclusive() -> None:
    items = (_gold("a", VerdictLabel.support), _gold("b", VerdictLabel.refute))

    def judge(claim: Claim, evidence: Sequence[Evidence]) -> Verdict:
        return _stub_verdict(claim, VerdictLabel.support)

    low = evaluate_gold(items, judge=judge, judge_name="stub", min_accuracy=0.5)
    assert low.accuracy == 0.5
    assert low.meets_threshold is True
    strict = evaluate_gold(items, judge=judge, judge_name="stub", min_accuracy=0.5 + 1e-9)
    assert strict.meets_threshold is False


def test_eval_ignores_llm_settings_and_does_not_use_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("eval must stay offline")

    monkeypatch.setenv("CLAIMFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMFORGE_LLM_MODEL", "gpt-test")
    monkeypatch.setattr("claimforge.judge._chat_completion", explode)
    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.setattr("claimforge.retrieve.retrieve_evidence", explode)
    report = evaluate_gold(load_gold_fixture(FIXTURE), fixture="gold_claims.json")
    assert report.accuracy == 1.0
    assert report.judge == "rubric"


def test_loader_rejects_bad_fixtures(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(GoldFixtureError, match="could not read"):
        load_gold_fixture(missing)

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(GoldFixtureError, match="not valid JSON"):
        load_gold_fixture(broken)

    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    items = raw["items"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"items": [items[0], items[0]]}), encoding="utf-8")
    with pytest.raises(GoldFixtureError, match="duplicate gold id"):
        load_gold_fixture(duplicate)

    bad_label = tmp_path / "label.json"
    labeled = json.loads(json.dumps(items[0]))
    labeled["expected_label"] = "maybe"
    bad_label.write_text(json.dumps([labeled]), encoding="utf-8")
    with pytest.raises(GoldFixtureError, match="expected_label"):
        load_gold_fixture(bad_label)

    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"items": items, "threshold": 0.5}), encoding="utf-8")
    with pytest.raises(GoldFixtureError, match="unknown fields"):
        load_gold_fixture(extra)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(GoldFixtureError, match="no items"):
        load_gold_fixture(empty)

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(items), encoding="utf-8")
    loaded = load_gold_fixture(bare)
    assert len(loaded) == len(items)
    assert loaded[0].id == items[0]["id"]


def test_eval_cli_prints_json_and_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--fixture", str(FIXTURE)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["n"] == 13
    assert payload["accuracy"] == 1.0
    assert payload["agreement_rate"] == 1.0
    assert payload["macro_f1"] == 1.0
    assert payload["meets_threshold"] is True
    assert payload["min_accuracy"] == 1.0
    assert payload["judge"] == "rubric"
    assert {row["id"] for row in payload["items"]} >= set(KNOWN_LABELS)
    assert "accuracy: 1.0000" in captured.err
    assert "agreement: 1.0000" in captured.err
    assert "macro_f1: 1.0000" in captured.err
    assert captured.out.strip().startswith("{")


def test_eval_cli_format_json_and_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--fixture", str(FIXTURE), "--format", "json"]) == 0
    json_only = capsys.readouterr()
    assert json.loads(json_only.out)["accuracy"] == 1.0
    assert json_only.err == ""

    assert main(["eval", "--fixture", str(FIXTURE), "--format", "table"]) == 0
    table_only = capsys.readouterr()
    assert table_only.err == ""
    assert "accuracy: 1.0000" in table_only.out
    assert "label" in table_only.out
    with pytest.raises(json.JSONDecodeError):
        json.loads(table_only.out)


def test_eval_cli_strict_exits_only_below_the_threshold(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert main(["eval", "--fixture", str(FIXTURE), "--strict", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["meets_threshold"] is True

    raw = _flip_first_label(json.loads(FIXTURE.read_text(encoding="utf-8")))
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert main(["eval", "--fixture", str(path), "--format", "json"]) == 0
    soft = capsys.readouterr()
    payload = json.loads(soft.out)
    assert payload["meets_threshold"] is False
    assert payload["accuracy"] < 1.0
    assert soft.err == ""

    assert main(["eval", "--fixture", str(path), "--strict", "--format", "json"]) == 1
    failed = capsys.readouterr()
    assert "below" in failed.err
    assert json.loads(failed.out)["accuracy"] == payload["accuracy"]

    assert (
        main(
            [
                "eval",
                "--fixture",
                str(path),
                "--strict",
                "--min-accuracy",
                "0",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["meets_threshold"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["eval"],
        ["eval", "--fixture", "missing-gold.json", "--min-accuracy", "1.5"],
        ["eval", "--fixture", "missing-gold.json", "--min-accuracy", "nan"],
    ],
)
def test_eval_cli_rejects_bad_arguments(argv: list[str], tmp_path: Path) -> None:
    if argv[-1] in {"1.5", "nan"} or (len(argv) > 2 and argv[2] == "missing-gold.json"):
        argv = ["eval", "--fixture", str(tmp_path / "missing-gold.json"), *argv[3:]]
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2


def test_eval_cli_rejects_a_bad_fixture(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["eval", "--fixture", str(path)])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as missing:
        main(["eval", "--fixture", str(tmp_path / "missing.json")])
    assert missing.value.code == 2
