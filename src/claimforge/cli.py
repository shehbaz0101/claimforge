"""Command line interface. Day 1 exposes the OpenAlex smoke command only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from claimforge import __version__
from claimforge.http_cache import CachedHttpClient, ClaimForgeError
from claimforge.openalex import DEFAULT_PER_PAGE, DEFAULT_QUERY, format_works, search_works

DEFAULT_CACHE_DIR = Path("data/cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claimforge",
        description="ClaimForge scientific claim verifier (Day 1: OpenAlex smoke).",
    )
    parser.add_argument("--version", action="version", version=f"claimforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke-openalex",
        help="Search OpenAlex and print a few work titles and ids",
    )
    smoke.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"works search text (default: {DEFAULT_QUERY!r})",
    )
    smoke.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help=f"number of works to request, 1-25 (default: {DEFAULT_PER_PAGE})",
    )
    smoke.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"disk cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "smoke-openalex":
        return _smoke_openalex(parser, args)
    parser.error(f"unknown command {args.command}")
    return 2


def _smoke_openalex(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.per_page < 1 or args.per_page > 25:
        parser.error("--per-page must be between 1 and 25")
    if not str(args.query).strip():
        parser.error("--query must not be empty")

    mailto = os.environ.get("CLAIMFORGE_OPENALEX_MAILTO", "").strip() or None
    client = CachedHttpClient(cache_dir=args.cache_dir)
    try:
        works = search_works(
            client,
            args.query,
            per_page=args.per_page,
            mailto=mailto,
        )
    except ClaimForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_works(args.query, works))
    return 0
