"""Command line interface.

``smoke-openalex`` prints work titles and ids.
``extract-claims`` fetches a few works and prints claims as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from claimforge import __version__
from claimforge.extract import extract_from_openalex_work
from claimforge.http_cache import CachedHttpClient, ClaimForgeError
from claimforge.models import dump_claims
from claimforge.openalex import (
    DEFAULT_PER_PAGE,
    DEFAULT_QUERY,
    format_works,
    search_work_records,
    search_works,
)

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
