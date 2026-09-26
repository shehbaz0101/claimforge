"""Offline evidence cassettes.

A cassette is a JSON file with a claim id and/or claim text plus an evidence
list. ``verify`` and ``retrieve-evidence`` read these files only when offline
mode is on. They are not gold labels. ``claimforge eval`` does not read them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from claimforge.models import Claim, Evidence
from claimforge.settings import fixture_dir

logger = logging.getLogger(__name__)


def normalize_claim_text(text: str) -> str:
    """Collapse whitespace and case so cassette text matches a claim."""

    return " ".join(text.casefold().split())


def load_matching_cassette(
    claim: Claim | str,
    directory: Path | str | None = None,
) -> list[Evidence] | None:
    """Return the evidence list from a matching cassette, or None.

    Match the claim id first, then the normalized claim text. ``None`` means
    nothing matched, so the caller may still use a warm HTTP cache. An empty
    list is a cassette that matched and stored no evidence. A missing
    directory, a bad file, or an invalid record is skipped. Offline mode
    does not raise for a bad cassette.
    """

    root = Path(directory) if directory is not None else Path(fixture_dir())
    if not root.is_dir():
        return None
    claim_id = claim.id if isinstance(claim, Claim) else None
    text_key = normalize_claim_text(claim.text if isinstance(claim, Claim) else claim)
    by_text: list[Evidence] | None = None
    for path in sorted(root.glob("*.json")):
        parsed = _read_cassette(path)
        if parsed is None:
            continue
        file_id, file_text, evidence = parsed
        if claim_id is not None and file_id == claim_id:
            return evidence
        if text_key and file_text == text_key and by_text is None:
            by_text = evidence
    return by_text


def _read_cassette(
    path: Path,
) -> tuple[str | None, str, list[Evidence]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable cassette %s (%s)", path, exc)
        return None
    if not isinstance(payload, Mapping):
        logger.warning("ignoring cassette %s: expected an object", path)
        return None
    raw_id = payload.get("id")
    if raw_id is not None and not isinstance(raw_id, str):
        logger.warning("ignoring cassette %s: id must be a string", path)
        return None
    raw_text = payload.get("text")
    if raw_text is None:
        raw_text = ""
    if not isinstance(raw_text, str):
        logger.warning("ignoring cassette %s: text must be a string", path)
        return None
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list):
        logger.warning("ignoring cassette %s: evidence must be a list", path)
        return None
    evidence: list[Evidence] = []
    for item in raw_evidence:
        try:
            evidence.append(Evidence.model_validate(item))
        except ValidationError as exc:
            logger.warning("ignoring cassette %s: %s", path, exc)
            return None
    claim_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else None
    return claim_id, normalize_claim_text(raw_text), evidence
