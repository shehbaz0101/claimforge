"""HTTP API for verify, judge, and offline eval.

``GET /health`` reports the package version.
``POST /verify`` retrieves evidence for claim text and returns a verdict.
``ranker`` is optional: ``auto`` (default), ``lexical``, or ``embeddings``.
``POST /judge`` scores a claim against a packed evidence list. It does not
retrieve. ``POST /eval`` scores inline gold items, one JSON fixture, or a
directory of JSON fixtures with the rubric. It does not retrieve and it
does not call a model.

``POST /verify`` is the only rate-limited route. The limit is a fixed window
in this process: 60 requests per 60 seconds unless
``CLAIMFORGE_VERIFY_RATE_LIMIT`` and ``CLAIMFORGE_VERIFY_RATE_WINDOW_S``
say otherwise. ``0`` disables it. A rejected call is HTTP 429 with a JSON
body and a ``Retry-After`` header. ``/health``, ``/judge``, and ``/eval``
are not limited.

A catalog failure does not crash the process. One failed catalog is dropped.
If every catalog fails, ``/verify`` returns HTTP 502 JSON. Offline mode
(``CLAIMFORGE_OFFLINE=1`` or ``claimforge serve --offline``) never opens a
catalog connection: a cassette or a warm cache is judged, and a miss is an
insufficient verdict with HTTP 200.

Run ``uvicorn claimforge.api:app`` or ``claimforge serve``. The server
binds to ``127.0.0.1:8000`` from the CLI. ``/verify`` uses the same catalog
clients as ``claimforge verify``. ``/judge`` and ``/eval`` do not open a
catalog connection.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import Response as StarletteResponse

from claimforge import __version__
from claimforge.eval import (
    DEFAULT_MIN_ACCURACY,
    GoldFixtureError,
    describe_fixtures,
    evaluate_gold,
    load_gold_fixtures,
    parse_gold_document,
    parse_min_accuracy,
)
from claimforge.http_cache import CachedHttpClient, ClaimForgeError
from claimforge.judge import claim_from_text, judge_claim
from claimforge.models import Claim, Evidence, Verdict
from claimforge.rank import RankerError, active_ranker_name
from claimforge.retrieve import DEFAULT_PER_SOURCE, DEFAULT_TOP_K, retrieve_evidence
from claimforge.settings import verify_rate_limit, verify_rate_window_s

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")
RANKER_HEADER = "X-ClaimForge-Ranker"
VERIFY_RATE_DETAIL = "rate limit exceeded for /verify"


class FixedWindowLimiter:
    """Count ``POST /verify`` calls inside one time window for this process.

    The window and the limit are read from the environment on each check.
    Defaults are 60 requests per 60 seconds. A limit of 0 allows every call.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()
        self._window_start = 0.0
        self._count = 0

    def reset(self) -> None:
        """Start a fresh window. Tests call this so cases do not share a count."""

        with self._lock:
            self._window_start = 0.0
            self._count = 0

    def decide(self) -> tuple[bool, float]:
        """Return whether the call is allowed and, if not, seconds to wait."""

        limit = verify_rate_limit()
        window = verify_rate_window_s()
        if limit <= 0:
            return True, 0.0
        now = self._clock()
        with self._lock:
            if self._window_start <= 0.0 or now - self._window_start >= window:
                self._window_start = now
                self._count = 0
            if self._count >= limit:
                retry = max(0.0, window - (now - self._window_start))
                return False, retry
            self._count += 1
            return True, 0.0


VERIFY_LIMITER = FixedWindowLimiter()

app = FastAPI(
    title="ClaimForge",
    version=__version__,
    description="Verify a scientific claim and score frozen gold fixtures.",
)


@app.middleware("http")
async def limit_verify_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[StarletteResponse]],
) -> StarletteResponse:
    """Reject ``POST /verify`` once the in-process window is full.

    Other routes are not counted. The body is JSON, including
    ``retry_after_s``. ``Retry-After`` is the same wait in whole seconds.
    """

    if request.method == "POST" and request.url.path.rstrip("/") == "/verify":
        allowed, retry_after = VERIFY_LIMITER.decide()
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": VERIFY_RATE_DETAIL,
                    "retry_after_s": round(retry_after, 3),
                },
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )
    return await call_next(request)


class HealthResponse(BaseModel):
    """Liveness payload. No catalog calls."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class VerifyRequest(BaseModel):
    """Claim text plus the same retrieval knobs as ``claimforge verify``."""

    model_config = ConfigDict(extra="forbid")

    text: str
    ranker: Literal["auto", "lexical", "embeddings"] = "auto"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=25)
    per_source: int = Field(default=DEFAULT_PER_SOURCE, ge=1, le=25)
    cache_dir: str | None = None

    @field_validator("text")
    @classmethod
    def _collapse_text(cls, value: str) -> str:
        stripped = " ".join(value.split())
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped

    @field_validator("cache_dir")
    @classmethod
    def _cache_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("cache_dir must not be empty")
        return stripped


class JudgeRequest(BaseModel):
    """One claim and the evidence the judge is allowed to see."""

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    evidence: list[Evidence]


class EvalRequest(BaseModel):
    """A local fixture path or inline gold items. Not both."""

    model_config = ConfigDict(extra="forbid")

    fixture: str | None = None
    items: list[dict[str, Any]] | None = None
    min_accuracy: float = DEFAULT_MIN_ACCURACY

    @field_validator("fixture")
    @classmethod
    def _fixture(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("fixture must not be empty")
        return stripped


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the process is up. Does not touch the network."""

    return HealthResponse(status="ok", version=__version__)


@app.post("/verify", response_model=Verdict)
def verify(body: VerifyRequest, response: Response) -> dict[str, object]:
    """Retrieve evidence for ``text`` and return one verdict.

    The body is the verdict JSON from the judge. The ranker name is the
    ``X-ClaimForge-Ranker`` response header.
    """

    claim = claim_from_text(body.text)
    cache_dir = Path(body.cache_dir) if body.cache_dir is not None else DEFAULT_CACHE_DIR
    client = CachedHttpClient(cache_dir=cache_dir)
    try:
        evidence = retrieve_evidence(
            client,
            claim,
            per_source=body.per_source,
            top_k=body.top_k,
            mailto=_mailto(),
            s2_api_key=_s2_api_key(),
            ranker=body.ranker,
            embedding_cache_dir=cache_dir / "embeddings",
        )
    except RankerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClaimForgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("verify failed")
        raise HTTPException(status_code=502, detail="evidence retrieval failed") from exc
    verdict = judge_claim(claim, evidence)
    response.headers[RANKER_HEADER] = active_ranker_name(body.ranker)
    return verdict.to_json_dict()


@app.post("/judge", response_model=Verdict)
def judge(body: JudgeRequest) -> dict[str, object]:
    """Score ``claim`` against ``evidence``. Does not retrieve literature."""

    return judge_claim(body.claim, body.evidence).to_json_dict()


@app.post("/eval")
def evaluate(body: EvalRequest) -> dict[str, object]:
    """Score gold items with the rubric and return the metrics report.

    Provide ``fixture`` (one JSON file or a directory of JSON fixtures) or
    ``items`` (the same objects a fixture file holds). The rubric ignores
    ``CLAIMFORGE_LLM_*``. A low accuracy is still HTTP 200; ``meets_threshold``
    is in the body. ``--strict`` is a CLI exit code, not an HTTP error.
    """

    has_fixture = body.fixture is not None
    has_items = body.items is not None
    if has_fixture == has_items:
        raise HTTPException(status_code=422, detail="provide a fixture path or inline items")
    try:
        minimum = parse_min_accuracy(body.min_accuracy)
        if body.items is not None:
            items = parse_gold_document(body.items)
            fixture = "inline"
        else:
            fixture_path = body.fixture
            if fixture_path is None:
                raise HTTPException(status_code=422, detail="provide a fixture path or inline items")
            items = load_gold_fixtures([fixture_path])
            fixture = describe_fixtures([fixture_path])
        report = evaluate_gold(items, fixture=fixture, min_accuracy=minimum)
    except GoldFixtureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.to_json_dict()


def _mailto() -> str | None:
    return os.environ.get("CLAIMFORGE_OPENALEX_MAILTO", "").strip() or None


def _s2_api_key() -> str | None:
    return os.environ.get("CLAIMFORGE_S2_API_KEY", "").strip() or None
