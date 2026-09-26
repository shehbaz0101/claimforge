"""Disk-cached HTTP GET with retries for rate limits and transient server errors."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

USER_AGENT = "ClaimForge/0.1 (+https://github.com/shehbaz0101/claimforge)"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_CACHE_VERSION = 1


class ClaimForgeError(Exception):
    """Base error for ClaimForge operations."""


class TransientNetworkError(ClaimForgeError):
    """A connection failure that may succeed if tried again."""


class HttpRequestError(ClaimForgeError):
    """A retryable HTTP status persisted until the retry budget was spent."""

    def __init__(self, message: str, *, status_code: int, url: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class CachedResponse:
    url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    from_cache: bool

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding)

    def json(self) -> object:
        return json.loads(self.body)


class HttpTransport(Protocol):
    """Minimal GET transport. HTTP error statuses are returned, not raised."""

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        """Fetch ``url`` and return the status, headers, and body."""


class UrllibTransport:
    """GET via the standard library. HTTP error statuses are returned, not raised."""

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return TransportResponse(
                    status_code=response.status,
                    headers={key: value for key, value in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            header_items = exc.headers.items() if exc.headers is not None else ()
            return TransportResponse(
                status_code=exc.code,
                headers={key: value for key, value in header_items},
                body=exc.read(),
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise TransientNetworkError(str(reason)) from exc


def canonical_url(url: str) -> str:
    """Normalize a URL so equivalent queries share one cache key.

    Scheme and host are lowercased, the query is sorted, and the fragment is
    dropped. The path is preserved as written aside from defaulting to ``/``.
    """

    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"expected an absolute http(s) URL, got {url!r}")
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    query = urlencode(sorted(pairs))
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def make_cache_key(url: str, method: str = "GET") -> str:
    """Return the SHA-256 hex digest of ``METHOD canonical-url``."""

    canonical = canonical_url(url)
    material = f"{method.upper()} {canonical}".encode()
    return hashlib.sha256(material).hexdigest()


def parse_retry_after(value: str, now: datetime) -> float | None:
    """Parse a Retry-After value into seconds, or None if it is not understood."""

    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - now).total_seconds())
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


class CachedHttpClient:
    """GET a URL, retry transient failures, and store 2xx bodies on disk.

    Cache files live in ``cache_dir`` and are named ``<sha256>.json``. Only
    successful responses are stored. 429 and transient 5xx responses are
    retried. A ``Retry-After`` header replaces exponential backoff when it
    parses; the wait is still capped by ``max_retry_after``.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        max_retry_after: float = 120.0,
        jitter: float = 0.0,
        user_agent: str = USER_AGENT,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_base < 0 or backoff_max < 0 or jitter < 0 or max_retry_after < 0:
            raise ValueError("backoff settings must be >= 0")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_retry_after = max_retry_after
        self.jitter = jitter
        self.user_agent = user_agent
        self.transport = transport if transport is not None else UrllibTransport()
        self._sleep = sleep
        self._now = now if now is not None else _utcnow
        self._rng = rng if rng is not None else random.random

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> CachedResponse:
        request_url = _merge_url(url, params)
        canonical = canonical_url(request_url)
        key = make_cache_key(canonical)
        if use_cache:
            cached = self._read_cache(key, canonical)
            if cached is not None:
                return cached

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if headers:
            request_headers.update(headers)

        last_network_error: TransientNetworkError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                fetched = self.transport.get(request_url, request_headers, self.timeout)
            except TransientNetworkError as exc:
                last_network_error = exc
                if attempt >= self.max_retries:
                    raise
                delay = self._backoff_delay(attempt)
                logger.info("network error for %s; retrying in %.2fs", canonical, delay)
                self._sleep(delay)
                continue

            if fetched.status_code not in RETRYABLE_STATUS:
                response = CachedResponse(
                    url=canonical,
                    status_code=fetched.status_code,
                    headers=dict(fetched.headers),
                    body=fetched.body,
                    from_cache=False,
                )
                if use_cache and 200 <= fetched.status_code < 300:
                    self._write_cache(key, response)
                return response

            if attempt >= self.max_retries:
                snippet = fetched.body[:200].decode("utf-8", errors="replace")
                raise HttpRequestError(
                    f"GET {canonical} failed with HTTP {fetched.status_code} "
                    f"after {attempt + 1} attempts: {snippet}",
                    status_code=fetched.status_code,
                    url=canonical,
                )

            delay = self._retry_delay(fetched.headers, attempt)
            logger.info(
                "HTTP %s for %s; retrying in %.2fs",
                fetched.status_code,
                canonical,
                delay,
            )
            self._sleep(delay)

        # The loop either returns or raises. This guards a future logic slip.
        if last_network_error is not None:
            raise last_network_error
        raise ClaimForgeError(f"GET {canonical} failed without a response")

    def _retry_delay(self, headers: Mapping[str, str], attempt: int) -> float:
        raw = _header(headers, "Retry-After")
        if raw is not None:
            parsed = parse_retry_after(raw, self._now())
            if parsed is not None:
                return min(parsed, self.max_retry_after)
        return self._backoff_delay(attempt)

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.backoff_max, self.backoff_base * (2**attempt))
        if self.jitter:
            delay += self._rng() * self.jitter
        return delay

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str, canonical: str) -> CachedResponse | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("v") != _CACHE_VERSION or payload.get("url") != canonical:
                raise ValueError("cache record does not match this request")
            status_code = payload["status_code"]
            if not isinstance(status_code, int) or not 200 <= status_code < 300:
                raise ValueError("cache record is not a stored success")
            body = base64.b64decode(payload["body_b64"], validate=True)
            raw_headers = payload.get("headers") or {}
            if not isinstance(raw_headers, dict):
                raise ValueError("cache headers must be an object")
            headers = {
                str(name): str(value) for name, value in raw_headers.items()
            }
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("ignoring unreadable cache file %s (%s)", path, exc)
            return None
        return CachedResponse(
            url=canonical,
            status_code=status_code,
            headers=headers,
            body=body,
            from_cache=True,
        )

    def _write_cache(self, key: str, response: CachedResponse) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "v": _CACHE_VERSION,
            "url": response.url,
            "status_code": response.status_code,
            "headers": response.headers,
            "body_b64": base64.b64encode(response.body).decode("ascii"),
            "cached_at": self._now().isoformat(),
        }
        path = self._cache_path(key)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _merge_url(url: str, params: Mapping[str, str] | None) -> str:
    if not params:
        return url
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    pairs.extend((str(key), str(value)) for key, value in params.items())
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(pairs), parts.fragment)
    )
