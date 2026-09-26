"""Offline demo for ``claimforge demo`` and ``scripts/demo.sh``.

The run does not open a catalog connection and does not call a model.
It verifies the Burgers claim against ``data/fixtures/cassettes/burgers.json``,
judges that ranked pack, and scores the gold fixture with the rubric.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from claimforge import __version__
from claimforge.eval import EvalReport, evaluate_gold, format_eval_table, load_gold_fixtures
from claimforge.http_cache import CachedHttpClient
from claimforge.judge import claim_from_text, judge_claim
from claimforge.models import Verdict
from claimforge.retrieve import retrieve_evidence
from claimforge.settings import pop_offline, push_offline

BURGERS_TEXT = "Physics-informed neural networks reduce the error on the Burgers equation."
DEFAULT_GOLD_FIXTURE = Path("tests/fixtures/gold_claims.json")
DEMO_RANKER = "lexical"

_LLM_ENV = (
    "CLAIMFORGE_LLM_API_KEY",
    "CLAIMFORGE_LLM_MODEL",
    "CLAIMFORGE_LLM_PROVIDER",
    "CLAIMFORGE_LLM_BASE_URL",
)


@dataclass(frozen=True)
class DemoReport:
    """Verify, judge, and eval results from one offline demo run."""

    claim_text: str
    verify: Verdict
    judge: Verdict
    evaluation: EvalReport


@contextmanager
def _rubric_only() -> Iterator[None]:
    """Hide LLM settings so the demo cannot call a chat endpoint."""

    saved = {key: os.environ.get(key) for key in _LLM_ENV}
    for key in _LLM_ENV:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_offline_demo(fixture: Path | str = DEFAULT_GOLD_FIXTURE) -> DemoReport:
    """Verify the Burgers cassette, judge that pack, and score ``fixture``.

    Retrieval uses an empty cache directory, so a warm ``data/cache`` cannot
    replace the cassette. ``CLAIMFORGE_LLM_*`` is ignored for this call and
    restored afterward.
    """

    fixture_path = Path(fixture)
    claim = claim_from_text(BURGERS_TEXT)
    with _rubric_only():
        token = push_offline(True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cache_dir = Path(tmp)
                evidence = retrieve_evidence(
                    CachedHttpClient(cache_dir=cache_dir),
                    claim,
                    ranker=DEMO_RANKER,
                    embedding_cache_dir=cache_dir / "embeddings",
                    offline=True,
                )
        finally:
            pop_offline(token)
        verified = judge_claim(claim, evidence)
        judged = judge_claim(claim, evidence)
        evaluation = evaluate_gold(
            load_gold_fixtures([fixture_path]),
            fixture=str(fixture_path),
        )
    return DemoReport(
        claim_text=claim.text,
        verify=verified,
        judge=judged,
        evaluation=evaluation,
    )


def format_demo_report(report: DemoReport) -> str:
    """Return the short report ``claimforge demo`` prints."""

    evidence = ", ".join(report.verify.evidence_ids) or "(none)"
    lines = [
        f"ClaimForge {__version__} offline demo",
        "No catalog requests. Ranker: lexical. Judge: rubric.",
        "",
        "Verify  burgers cassette",
        f"  text        {report.claim_text}",
        f"  label       {report.verify.label.value}",
        f"  confidence  {report.verify.confidence:.4f}",
        f"  evidence    {evidence}",
        f"  rationale   {report.verify.rationale}",
        "",
        "Judge   ranked cassette, no retrieval",
        f"  label       {report.judge.label.value}",
        f"  confidence  {report.judge.confidence:.4f}",
        "",
        "Eval    gold fixture",
        format_eval_table(report.evaluation),
    ]
    return "\n".join(lines)
