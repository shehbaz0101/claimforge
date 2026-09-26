"""Score frozen gold fixtures with the rubric judge.

The harness reads claims and inline evidence packs from JSON. A path may be
one file, several files, or a directory of ``*.json`` fixtures. Directories
are not recursive. It does not retrieve literature and it does not call a
model. ``CLAIMFORGE_LLM_*`` is ignored. Metrics are exact label match against
``expected_label``.

* ``accuracy`` is correct labels divided by the number of items.
* ``agreement_rate`` is that same fraction. With one judge and one gold label
  per item, raw agreement and accuracy are equal. Both are reported so the
  record names the agreement explicitly.
* Per-label precision, recall, and F1 cover ``support``, ``refute``, and
  ``insufficient``. A rate is null when its denominator is zero. F1 is null
  unless both precision and recall are defined. Macro-F1 averages the defined
  F1 values and ignores labels that were neither gold nor predicted.

``--strict`` is the only accuracy failure. The default minimum is 1.0 because
``tests/fixtures/gold_claims.json`` is written so the rubric matches every
gold label. The comparison is inclusive.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from claimforge.judge import rubric_verdict
from claimforge.models import Claim, Evidence, Verdict, VerdictLabel

# Inclusive bar for ``claimforge eval --strict``. The committed fixture matches
# the rubric on every item, so the default is 1.0. Override with --min-accuracy.
DEFAULT_MIN_ACCURACY = 1.0

_LABELS: tuple[VerdictLabel, ...] = (
    VerdictLabel.support,
    VerdictLabel.refute,
    VerdictLabel.insufficient,
)
_ITEM_KEYS = frozenset({"id", "expected_label", "claim", "evidence", "note"})
_DOCUMENT_KEYS = frozenset({"name", "description", "items"})

JudgeFn = Callable[[Claim, Sequence[Evidence]], Verdict]


class GoldFixtureError(ValueError):
    """The gold fixture could not be read or did not match the schema."""


@dataclass(frozen=True)
class GoldItem:
    """One frozen claim, its pack, and the label the judge should return."""

    id: str
    expected_label: VerdictLabel
    claim: Claim
    evidence: tuple[Evidence, ...]
    note: str = ""


@dataclass(frozen=True)
class LabelScore:
    """Precision, recall, and F1 for one verdict label."""

    label: str
    precision: float | None
    recall: float | None
    f1: float | None
    gold: int
    predicted: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "precision": _round_metric(self.precision),
            "recall": _round_metric(self.recall),
            "f1": _round_metric(self.f1),
            "gold": self.gold,
            "predicted": self.predicted,
        }


@dataclass(frozen=True)
class ItemResult:
    """Gold versus predicted label for one fixture item."""

    id: str
    claim_id: str
    expected: VerdictLabel
    predicted: VerdictLabel
    match: bool
    confidence: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "expected": self.expected.value,
            "predicted": self.predicted.value,
            "match": self.match,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class EvalReport:
    """Accuracy, agreement, and per-label F1 for one fixture run."""

    fixture: str
    judge: str
    n: int
    correct: int
    accuracy: float
    agreement_rate: float
    macro_f1: float | None
    per_label: dict[str, LabelScore]
    confusion: dict[str, dict[str, int]]
    items: tuple[ItemResult, ...]
    min_accuracy: float
    meets_threshold: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fixture": self.fixture,
            "judge": self.judge,
            "n": self.n,
            "correct": self.correct,
            "accuracy": _round_metric(self.accuracy),
            "agreement_rate": _round_metric(self.agreement_rate),
            "macro_f1": _round_metric(self.macro_f1),
            "min_accuracy": _round_metric(self.min_accuracy),
            "meets_threshold": self.meets_threshold,
            "per_label": {name: score.to_json_dict() for name, score in self.per_label.items()},
            "confusion": self.confusion,
            "items": [item.to_json_dict() for item in self.items],
        }


def parse_min_accuracy(value: float) -> float:
    """Return ``value`` when it is a finite accuracy threshold in ``[0, 1]``."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GoldFixtureError("min accuracy must be between 0 and 1") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise GoldFixtureError("min accuracy must be between 0 and 1")
    return number


def load_gold_fixture(path: Path | str) -> tuple[GoldItem, ...]:
    """Load one gold-claims file. Packs are inline. No I/O except the file read."""

    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GoldFixtureError(f"could not read gold fixture: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GoldFixtureError(f"gold fixture is not valid JSON: {exc}") from exc
    return parse_gold_document(payload)


def parse_gold_document(payload: object) -> tuple[GoldItem, ...]:
    """Parse a gold document or a bare list of items. Does not touch the network."""

    if isinstance(payload, list):
        raw_items: object = payload
    elif isinstance(payload, dict):
        unknown = set(payload) - _DOCUMENT_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise GoldFixtureError(f"gold fixture has unknown fields: {names}")
        raw_items = payload.get("items")
    else:
        raise GoldFixtureError("gold fixture must be an object with items, or a list of items")
    if not isinstance(raw_items, list):
        raise GoldFixtureError("gold fixture items must be a list")
    if not raw_items:
        raise GoldFixtureError("gold fixture contained no items")

    items: list[GoldItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        item = _parse_item(raw, index)
        if item.id in seen:
            raise GoldFixtureError(f"duplicate gold id: {item.id}")
        seen.add(item.id)
        items.append(item)
    return tuple(items)


def expand_fixture_paths(paths: Sequence[Path | str]) -> tuple[Path, ...]:
    """Turn files and directories into the JSON files a batch should score.

    A directory contributes its ``*.json`` files, sorted by name, and nothing
    nested below it. Other files in that directory are skipped. A path that
    is not a directory is kept as given, including a missing file, so the
    loader can report the read error.
    """

    if not paths:
        raise GoldFixtureError("no gold fixtures were given")
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = _json_files_in(path)
            if not found:
                raise GoldFixtureError(f"gold fixture directory has no JSON files: {path}")
            files.extend(found)
            continue
        files.append(path)
    return tuple(files)


def load_gold_fixtures(paths: Sequence[Path | str]) -> tuple[GoldItem, ...]:
    """Load one file, several files, or every JSON file in a directory.

    Item ids must be unique across the batch. One file matches
    :func:`load_gold_fixture`.
    """

    items: list[GoldItem] = []
    seen: set[str] = set()
    for path in expand_fixture_paths(paths):
        for item in load_gold_fixture(path):
            if item.id in seen:
                raise GoldFixtureError(f"duplicate gold id: {item.id}")
            seen.add(item.id)
            items.append(item)
    return tuple(items)


def describe_fixtures(paths: Sequence[Path | str]) -> str:
    """Label a batch for the metrics report. One path stays a single string."""

    labels = [str(path) for path in paths]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels)


def _json_files_in(directory: Path) -> list[Path]:
    files = [
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() == ".json" and not item.name.startswith(".")
    ]
    return sorted(files, key=lambda item: item.name)


def evaluate_gold(
    items: Sequence[GoldItem],
    *,
    fixture: str = "",
    judge: JudgeFn | None = None,
    judge_name: str = "rubric",
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
) -> EvalReport:
    """Judge each gold item and return accuracy, F1, and agreement.

    The default judge is :func:`claimforge.judge.rubric_verdict`. A custom
    ``judge`` is for tests. The CLI does not pass one, so a configured LLM
    cannot change the metrics or open a connection.
    """

    if not items:
        raise GoldFixtureError("gold fixture contained no items")
    threshold = parse_min_accuracy(min_accuracy)
    scorer = rubric_verdict if judge is None else judge
    results: list[ItemResult] = []
    for item in items:
        verdict = scorer(item.claim, item.evidence)
        predicted = verdict.label
        results.append(
            ItemResult(
                id=item.id,
                claim_id=item.claim.id,
                expected=item.expected_label,
                predicted=predicted,
                match=predicted is item.expected_label,
                confidence=verdict.confidence,
            )
        )
    return _report_from_results(
        results,
        fixture=fixture,
        judge_name=judge_name,
        min_accuracy=threshold,
    )


def format_eval_table(report: EvalReport) -> str:
    """Return a compact text table of the metrics report."""

    macro = "n/a" if report.macro_f1 is None else f"{report.macro_f1:.4f}"
    lines = [
        f"fixture: {report.fixture}",
        f"judge: {report.judge}",
        (
            f"items: {report.n}  correct: {report.correct}  "
            f"accuracy: {report.accuracy:.4f}  "
            f"agreement: {report.agreement_rate:.4f}  "
            f"macro_f1: {macro}"
        ),
        "",
        f"{'label':<14}{'precision':>11}{'recall':>11}{'f1':>11}{'gold':>8}{'predicted':>12}",
    ]
    for label in _LABELS:
        score = report.per_label[label.value]
        lines.append(
            f"{label.value:<14}"
            f"{_cell(score.precision):>11}"
            f"{_cell(score.recall):>11}"
            f"{_cell(score.f1):>11}"
            f"{score.gold:>8}"
            f"{score.predicted:>12}"
        )
    missed = [item for item in report.items if not item.match]
    if not missed:
        lines.append("")
        lines.append("mismatches: none")
    else:
        lines.append("")
        lines.append("mismatches:")
        for item in missed:
            lines.append(
                f"  {item.id} expected {item.expected.value} predicted {item.predicted.value}"
            )
    return "\n".join(lines)


def _report_from_results(
    results: Sequence[ItemResult],
    *,
    fixture: str,
    judge_name: str,
    min_accuracy: float,
) -> EvalReport:
    expected = [item.expected for item in results]
    predicted = [item.predicted for item in results]
    correct = sum(1 for item in results if item.match)
    n = len(results)
    accuracy = correct / n
    per_label = _per_label_scores(expected, predicted)
    defined = [score.f1 for score in per_label.values() if score.f1 is not None]
    macro_f1 = sum(defined) / len(defined) if defined else None
    return EvalReport(
        fixture=fixture,
        judge=judge_name,
        n=n,
        correct=correct,
        accuracy=accuracy,
        agreement_rate=accuracy,
        macro_f1=macro_f1,
        per_label=per_label,
        confusion=_confusion(expected, predicted),
        items=tuple(results),
        min_accuracy=min_accuracy,
        meets_threshold=accuracy >= min_accuracy,
    )


def _per_label_scores(
    expected: Sequence[VerdictLabel],
    predicted: Sequence[VerdictLabel],
) -> dict[str, LabelScore]:
    scores: dict[str, LabelScore] = {}
    for label in _LABELS:
        tp = sum(
            1
            for gold, pred in zip(expected, predicted, strict=True)
            if gold is label and pred is label
        )
        gold_count = sum(1 for gold in expected if gold is label)
        predicted_count = sum(1 for pred in predicted if pred is label)
        fp = predicted_count - tp
        fn = gold_count - tp
        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        scores[label.value] = LabelScore(
            label=label.value,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            gold=gold_count,
            predicted=predicted_count,
        )
    return scores


def _confusion(
    expected: Sequence[VerdictLabel],
    predicted: Sequence[VerdictLabel],
) -> dict[str, dict[str, int]]:
    matrix = {gold.value: {pred.value: 0 for pred in _LABELS} for gold in _LABELS}
    for gold, pred in zip(expected, predicted, strict=True):
        matrix[gold.value][pred.value] += 1
    return matrix


def _parse_item(raw: object, index: int) -> GoldItem:
    if not isinstance(raw, Mapping):
        raise GoldFixtureError(f"gold item {index} must be an object")
    unknown = set(raw) - _ITEM_KEYS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise GoldFixtureError(f"gold item {index} has unknown fields: {names}")
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise GoldFixtureError(f"gold item {index} is missing an id")
    label_raw = raw.get("expected_label")
    if not isinstance(label_raw, str):
        raise GoldFixtureError(f"gold item {item_id.strip()} is missing expected_label")
    try:
        expected = VerdictLabel(label_raw.strip().lower())
    except ValueError as exc:
        raise GoldFixtureError(
            f"gold item {item_id.strip()} expected_label must be support, refute, or insufficient"
        ) from exc
    note = raw.get("note", "")
    if not isinstance(note, str):
        raise GoldFixtureError(f"gold item {item_id.strip()} note must be a string")
    try:
        claim = Claim.model_validate(raw.get("claim"))
    except ValidationError as exc:
        raise GoldFixtureError(f"gold item {item_id.strip()} has an invalid claim: {exc}") from exc
    evidence_raw = raw.get("evidence")
    if not isinstance(evidence_raw, list):
        raise GoldFixtureError(f"gold item {item_id.strip()} evidence must be a list")
    evidence = tuple(_parse_evidence(row, item_id.strip()) for row in evidence_raw)
    return GoldItem(
        id=item_id.strip(),
        expected_label=expected,
        claim=claim,
        evidence=evidence,
        note=" ".join(note.split()),
    )


def _parse_evidence(raw: object, item_id: str) -> Evidence:
    try:
        return Evidence.model_validate(raw)
    except ValidationError as exc:
        raise GoldFixtureError(f"gold item {item_id} has invalid evidence: {exc}") from exc


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _cell(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
