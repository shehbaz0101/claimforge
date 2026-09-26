"""Command line interface.

``smoke-openalex`` prints work titles and ids.
``extract-claims`` fetches a few works and prints claims as JSON.
``retrieve-evidence`` searches OpenAlex, arXiv, and Semantic Scholar,
re-scores the pack with the active ranker, and prints evidence as JSON.
The ranker name is written to stderr. Scores on stdout are that ranker's
cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from claimforge import __version__
from claimforge.extract import extract_from_openalex_work
from claimforge.http_cache import CachedHttpClient, ClaimForgeError
from claimforge.models import Claim, dump_claims, dump_evidence
from claimforge.openalex import (
    DEFAULT_PER_PAGE,
    DEFAULT_QUERY,
    format_works,
    search_work_records,
    search_works,
)
from claimforge.rank import active_ranker_name
from claimforge.retrieve import DEFAULT_PER_SOURCE, DEFAULT_TOP_K, retrieve_evidence

DEFAULT_CACHE_DIR = Path("data/cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claimforge",
        description="ClaimForge scientific claim verifier.",
    )
    parser.add_argument("--version", action="version", version=f"claimforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke-openalex",
        help="Search OpenAlex and print a few work titles and ids",
    )
    _add_search_arguments(smoke)

    extract = subparsers.add_parser(
        "extract-claims",
        help="Fetch OpenAlex works and print claims extracted from abstracts as JSON",
    )
    _add_search_arguments(extract)

    retrieve = subparsers.add_parser(
        "retrieve-evidence",
        help="Search literature for a claim and print evidence as JSON",
    )
    source = retrieve.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="claim text to search for")
    source.add_argument(
        "--claim-json",
        type=Path,
        help="path to one Claim JSON object or a list of claims",
    )
    retrieve.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"evidence records to keep after dedupe, 1-25 (default: {DEFAULT_TOP_K})",
    )
    retrieve.add_argument(
        "--per-source",
        type=int,
        default=DEFAULT_PER_SOURCE,
        help=f"records to request from each catalog, 1-25 (default: {DEFAULT_PER_SOURCE})",
    )
    retrieve.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"disk cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    retrieve.add_argument(
        "--ranker",
        choices=("auto", "lexical", "embeddings"),
        default="auto",
        help=(
            "auto uses a local embedding model when sentence-transformers is "
            "installed, otherwise TF-IDF cosine (default: auto)"
        ),
    )
    return parser


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"works search text (default: {DEFAULT_QUERY!r})",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help=f"number of works to request, 1-25 (default: {DEFAULT_PER_PAGE})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"disk cache directory (default: {DEFAULT_CACHE_DIR})",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "smoke-openalex":
        return _smoke_openalex(parser, args)
    if args.command == "extract-claims":
        return _extract_claims(parser, args)
    if args.command == "retrieve-evidence":
        return _retrieve_evidence(parser, args)
    parser.error(f"unknown command {args.command}")
    return 2


def _search_inputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[str, int, str | None]:
    if args.per_page < 1 or args.per_page > 25:
        parser.error("--per-page must be between 1 and 25")
    query = str(args.query).strip()
    if not query:
        parser.error("--query must not be empty")
    mailto = os.environ.get("CLAIMFORGE_OPENALEX_MAILTO", "").strip() or None
    return query, args.per_page, mailto


def _smoke_openalex(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    query, per_page, mailto = _search_inputs(parser, args)
    client = CachedHttpClient(cache_dir=args.cache_dir)
    try:
        works = search_works(
            client,
            query,
            per_page=per_page,
            mailto=mailto,
        )
    except ClaimForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_works(query, works))
    return 0


def _extract_claims(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    query, per_page, mailto = _search_inputs(parser, args)
    client = CachedHttpClient(cache_dir=args.cache_dir)
    try:
        records = search_work_records(
            client,
            query,
            per_page=per_page,
            mailto=mailto,
        )
    except ClaimForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    claims = []
    for record in records:
        claims.extend(extract_from_openalex_work(record))
    print(json.dumps(dump_claims(claims), indent=2, ensure_ascii=False))
    return 0


def _retrieve_evidence(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.top_k < 1 or args.top_k > 25:
        parser.error("--top-k must be between 1 and 25")
    if args.per_source < 1 or args.per_source > 25:
        parser.error("--per-source must be between 1 and 25")
    mailto = os.environ.get("CLAIMFORGE_OPENALEX_MAILTO", "").strip() or None
    s2_api_key = os.environ.get("CLAIMFORGE_S2_API_KEY", "").strip() or None
    try:
        targets = _retrieval_targets(args)
    except ValueError as exc:
        parser.error(str(exc))
    client = CachedHttpClient(cache_dir=args.cache_dir)
    packs: list[tuple[Claim | None, list[dict[str, object]]]] = []
    for claim in targets:
        try:
            evidence = retrieve_evidence(
                client,
                claim if claim is not None else str(args.text).strip(),
                per_source=args.per_source,
                top_k=args.top_k,
                mailto=mailto,
                s2_api_key=s2_api_key,
                ranker=args.ranker,
                embedding_cache_dir=args.cache_dir / "embeddings",
            )
        except ClaimForgeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        packs.append((claim, dump_evidence(evidence)))
    print(f"ranker: {active_ranker_name(args.ranker)}", file=sys.stderr)
    if len(packs) == 1:
        print(json.dumps(packs[0][1], indent=2, ensure_ascii=False))
        return 0
    body = [
        {"claim_id": claim.id, "evidence": evidence}
        for claim, evidence in packs
        if claim is not None
    ]
    print(json.dumps(body, indent=2, ensure_ascii=False))
    return 0


def _retrieval_targets(args: argparse.Namespace) -> list[Claim | None]:
    """Return claims from ``--claim-json``, or ``[None]`` when ``--text`` is set."""

    if args.text is not None:
        if not str(args.text).strip():
            raise ValueError("--text must not be empty")
        return [None]
    return _load_claims(args.claim_json)


def _load_claims(path: Path) -> list[Claim]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read claim JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"claim JSON is not valid JSON: {exc}") from exc
    if isinstance(payload, dict):
        items: list[object] = [payload]
    elif isinstance(payload, list):
        items = list(payload)
    else:
        raise ValueError("claim JSON must be an object or a list of objects")
    if not items:
        raise ValueError("claim JSON contained no claims")
    claims: list[Claim] = []
    for item in items:
        try:
            claims.append(Claim.model_validate(item))
        except ValidationError as exc:
            raise ValueError(f"invalid claim: {exc}") from exc
    return claims
