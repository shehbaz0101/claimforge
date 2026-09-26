"""Command line interface.

``smoke-openalex`` prints work titles and ids.
``extract-claims`` fetches a few works and prints claims as JSON.
``retrieve-evidence`` searches OpenAlex, arXiv, and Semantic Scholar,
re-scores the pack with the active ranker, and prints evidence as JSON.
The ranker name is written to stderr. Scores on stdout are that ranker's
cosine similarity.
``verify`` runs that retrieval and then the judge, and prints a verdict.
``--offline`` on ``verify`` and ``retrieve-evidence`` does not use the
network. A cassette under ``data/fixtures/cassettes`` or a warm disk cache
is the pack. If neither has the claim, retrieve prints an empty list and
verify prints an insufficient verdict.
``judge`` scores a claim against a saved evidence file and does not use
the network.
``eval`` scores gold fixtures with the rubric and prints accuracy,
per-label F1, and agreement. ``--fixture`` takes one JSON file, several
files, or a directory of JSON fixtures. It does not use the network.
Exit code 0 unless ``--strict`` and accuracy is below ``--min-accuracy``.
``serve`` runs the HTTP API (``GET /health``, ``POST /verify``,
``POST /judge``, ``POST /eval``). The same app is ``uvicorn claimforge.api:app``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from claimforge import __version__
from claimforge.eval import (
    DEFAULT_MIN_ACCURACY,
    GoldFixtureError,
    describe_fixtures,
    evaluate_gold,
    format_eval_table,
    load_gold_fixtures,
    parse_min_accuracy,
)
from claimforge.extract import extract_from_openalex_work
from claimforge.http_cache import CachedHttpClient, ClaimForgeError
from claimforge.judge import claim_from_text, judge_claim
from claimforge.models import Claim, Evidence, Verdict, dump_claims, dump_evidence, dump_verdicts
from claimforge.openalex import (
    DEFAULT_PER_PAGE,
    DEFAULT_QUERY,
    format_works,
    search_work_records,
    search_works,
)
from claimforge.rank import active_ranker_name
from claimforge.retrieve import DEFAULT_PER_SOURCE, DEFAULT_TOP_K, retrieve_evidence
from claimforge.settings import pop_offline, push_offline

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
    retrieve.add_argument(
        "--offline",
        action="store_true",
        help=(
            "do not use the network; read a cassette pack or the disk cache, "
            "or return an empty pack"
        ),
    )

    verify = subparsers.add_parser(
        "verify",
        help="Retrieve evidence for a claim and print a verdict as JSON",
    )
    verify_source = verify.add_mutually_exclusive_group(required=True)
    verify_source.add_argument("--text", help="claim text to retrieve and judge")
    verify_source.add_argument(
        "--claim-json",
        type=Path,
        help="path to one Claim JSON object or a list of claims",
    )
    verify.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"evidence records to keep after dedupe, 1-25 (default: {DEFAULT_TOP_K})",
    )
    verify.add_argument(
        "--per-source",
        type=int,
        default=DEFAULT_PER_SOURCE,
        help=f"records to request from each catalog, 1-25 (default: {DEFAULT_PER_SOURCE})",
    )
    verify.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"disk cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    verify.add_argument(
        "--ranker",
        choices=("auto", "lexical", "embeddings"),
        default="auto",
        help=(
            "auto uses a local embedding model when sentence-transformers is "
            "installed, otherwise TF-IDF cosine (default: auto)"
        ),
    )
    verify.add_argument(
        "--offline",
        action="store_true",
        help=(
            "do not use the network; judge a cassette pack or a cached pack, "
            "or record insufficient when neither exists"
        ),
    )

    judge = subparsers.add_parser(
        "judge",
        help="Score a saved claim and evidence pack and print a verdict as JSON",
    )
    judge.add_argument(
        "--claim-json",
        type=Path,
        required=True,
        help="path to one Claim JSON object or a list of claims",
    )
    judge.add_argument(
        "--evidence-json",
        type=Path,
        required=True,
        help="evidence list, one evidence object, or packs paired by claim_id",
    )

    evaluate = subparsers.add_parser(
        "eval",
        help="Score gold fixtures with the rubric and print agreement metrics",
    )
    evaluate.add_argument(
        "--fixture",
        type=Path,
        nargs="+",
        action=_CollectPaths,
        required=True,
        metavar="PATH",
        help=(
            "gold claims JSON file, several files, or a directory of JSON fixtures "
            "(example: tests/fixtures/gold_claims.json)"
        ),
    )
    evaluate.add_argument(
        "--format",
        dest="output_format",
        choices=("both", "json", "table"),
        default="both",
        help="both writes a table to stderr and JSON to stdout (default: both)",
    )
    evaluate.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_MIN_ACCURACY,
        help=(
            "inclusive accuracy bar for --strict, from 0 to 1 "
            f"(default: {DEFAULT_MIN_ACCURACY:.1f})"
        ),
    )
    evaluate.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when accuracy is below --min-accuracy; otherwise exit 0",
    )

    serve = subparsers.add_parser(
        "serve",
        help="Run the HTTP API (health, verify, judge, eval)",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: 127.0.0.1)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="bind port (default: 8000)",
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        help="reload when source files change",
    )
    serve.add_argument(
        "--offline",
        action="store_true",
        help="set CLAIMFORGE_OFFLINE=1 for this process so /verify does not use the network",
    )
    return parser


class _CollectPaths(argparse.Action):
    """Append every path from one or more ``--fixture`` flags."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        current = list(getattr(namespace, self.dest) or [])
        if isinstance(values, list):
            current.extend(values)
        else:
            current.append(values)
        setattr(namespace, self.dest, current)


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
    if args.command == "verify":
        return _verify(parser, args)
    if args.command == "judge":
        return _judge(parser, args)
    if args.command == "eval":
        return _eval(parser, args)
    if args.command == "serve":
        return _serve(parser, args)
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
    token = push_offline(True) if args.offline else None
    try:
        return _retrieve_evidence_online(args, targets, mailto, s2_api_key)
    finally:
        if token is not None:
            pop_offline(token)


def _retrieve_evidence_online(
    args: argparse.Namespace,
    targets: list[Claim | None],
    mailto: str | None,
    s2_api_key: str | None,
) -> int:
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
        except Exception as exc:
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


def _verify(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.top_k < 1 or args.top_k > 25:
        parser.error("--top-k must be between 1 and 25")
    if args.per_source < 1 or args.per_source > 25:
        parser.error("--per-source must be between 1 and 25")
    mailto = os.environ.get("CLAIMFORGE_OPENALEX_MAILTO", "").strip() or None
    s2_api_key = os.environ.get("CLAIMFORGE_S2_API_KEY", "").strip() or None
    try:
        claims = _verify_claims(args)
    except ValueError as exc:
        parser.error(str(exc))
    token = push_offline(True) if args.offline else None
    try:
        client = CachedHttpClient(cache_dir=args.cache_dir)
        verdicts: list[Verdict] = []
        for claim in claims:
            try:
                evidence = retrieve_evidence(
                    client,
                    claim,
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
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            verdicts.append(judge_claim(claim, evidence))
        print(f"ranker: {active_ranker_name(args.ranker)}", file=sys.stderr)
        _print_verdicts(verdicts)
        return 0
    finally:
        if token is not None:
            pop_offline(token)


def _judge(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        claims = _load_claims(args.claim_json)
        packs = _load_evidence_packs(args.evidence_json, claims)
    except ValueError as exc:
        parser.error(str(exc))
    verdicts = [judge_claim(claim, evidence) for claim, evidence in zip(claims, packs, strict=True)]
    _print_verdicts(verdicts)
    return 0


def _eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        minimum = parse_min_accuracy(args.min_accuracy)
        items = load_gold_fixtures(args.fixture)
    except GoldFixtureError as exc:
        parser.error(str(exc))
    report = evaluate_gold(
        items,
        fixture=describe_fixtures(args.fixture),
        min_accuracy=minimum,
    )
    table = format_eval_table(report)
    if args.output_format == "table":
        print(table)
    elif args.output_format == "both":
        print(table, file=sys.stderr)
    if args.output_format in ("json", "both"):
        print(json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False))
    if args.strict and not report.meets_threshold:
        print(
            f"accuracy {report.accuracy:.4f} is below {report.min_accuracy:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


def _serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    host = str(args.host).strip()
    if not host:
        parser.error("--host must not be empty")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.offline:
        os.environ["CLAIMFORGE_OFFLINE"] = "1"
    try:
        import uvicorn
    except ImportError:
        print(
            "error: uvicorn is not installed. Reinstall ClaimForge to run the API.",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(
        "claimforge.api:app",
        host=host,
        port=args.port,
        reload=bool(args.reload),
    )
    return 0


def _verify_claims(args: argparse.Namespace) -> list[Claim]:
    """Claims to verify. ``--text`` becomes one synthetic claim."""

    if args.text is not None:
        if not str(args.text).strip():
            raise ValueError("--text must not be empty")
        return [claim_from_text(str(args.text))]
    return _load_claims(args.claim_json)


def _print_verdicts(verdicts: list[Verdict]) -> None:
    if len(verdicts) == 1:
        print(json.dumps(verdicts[0].to_json_dict(), indent=2, ensure_ascii=False))
        return
    print(json.dumps(dump_verdicts(verdicts), indent=2, ensure_ascii=False))


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


def _load_evidence_packs(path: Path, claims: list[Claim]) -> list[list[Evidence]]:
    """Read evidence for ``claims``.

    A bare list or one evidence object is the pack for a single claim.
    Several claims need a ``claim_id`` to list map, or a list of
    ``{"claim_id", "evidence"}`` objects.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read evidence JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"evidence JSON is not valid JSON: {exc}") from exc

    if isinstance(payload, dict) and _is_evidence_object(payload):
        if len(claims) != 1:
            raise ValueError("a single evidence object requires one claim")
        return [[_validate_evidence(payload)]]
    if isinstance(payload, dict):
        return [_pack_from_map(payload, claim) for claim in claims]
    if isinstance(payload, list):
        if payload and all(_is_paired_pack(item) for item in payload):
            by_id = {str(item["claim_id"]): item["evidence"] for item in payload}
            return [_validate_evidence_list(by_id.get(claim.id), claim.id) for claim in claims]
        if len(claims) != 1:
            raise ValueError(
                "a bare evidence list requires one claim; pair each pack with claim_id"
            )
        return [_validate_evidence_list(payload, claims[0].id)]
    raise ValueError(
        "evidence JSON must be an evidence list, one evidence object, or claim_id packs"
    )


def _is_evidence_object(item: object) -> bool:
    return isinstance(item, dict) and "source" in item and "url" in item and "id" in item


def _is_paired_pack(item: object) -> bool:
    return (
        isinstance(item, dict)
        and "claim_id" in item
        and "evidence" in item
        and "source" not in item
    )


def _pack_from_map(payload: dict[str, object], claim: Claim) -> list[Evidence]:
    if claim.id not in payload:
        raise ValueError(f"no evidence for claim {claim.id}")
    return _validate_evidence_list(payload[claim.id], claim.id)


def _validate_evidence_list(raw: object, claim_id: str) -> list[Evidence]:
    if not isinstance(raw, list):
        raise ValueError(f"evidence for {claim_id} must be a list")
    return [_validate_evidence(item) for item in raw]


def _validate_evidence(item: object) -> Evidence:
    try:
        return Evidence.model_validate(item)
    except ValidationError as exc:
        raise ValueError(f"invalid evidence: {exc}") from exc
