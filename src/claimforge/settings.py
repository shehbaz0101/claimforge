"""Environment settings.

The package does not load a dotenv file. Each helper reads the current
environment, so a command or request sees values that were exported before
it started. Invalid values are ignored and the default is used.
"""

from __future__ import annotations

import logging
import math
import os
from contextvars import ContextVar, Token

logger = logging.getLogger(__name__)

# ``None`` means "read CLAIMFORGE_OFFLINE". The CLI sets this for ``--offline``.
_offline_override: ContextVar[bool | None] = ContextVar(
    "claimforge_offline_override",
    default=None,
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def push_offline(enabled: bool) -> Token[bool | None]:
    """Force offline mode until :func:`pop_offline`."""

    return _offline_override.set(enabled)


def pop_offline(token: Token[bool | None]) -> None:
    """Restore the offline override from :func:`push_offline`."""

    _offline_override.reset(token)


def offline_enabled() -> bool:
    """True when this call must not use the network.

    ``--offline`` wins over the environment. Otherwise
    ``CLAIMFORGE_OFFLINE`` is true for ``1``, ``true``, ``yes``, or ``on``.
    """

    override = _offline_override.get()
    if override is not None:
        return override
    return os.environ.get("CLAIMFORGE_OFFLINE", "").strip().lower() in _TRUTHY


def http_min_interval_s() -> float:
    """Minimum seconds between outbound GETs to one host.

    Unset means ``0`` (no extra delay). ``CLAIMFORGE_HTTP_MIN_INTERVAL_S``
    is optional. Cache hits do not wait.
    """

    return _float_env("CLAIMFORGE_HTTP_MIN_INTERVAL_S", default=0.0)


def source_failure_limit() -> int:
    """Consecutive hard failures before a catalog is skipped.

    The default is 3. ``0`` disables the skip. The counter is process memory
    only. See ``claimforge.retrieve`` for how long the skip lasts.
    """

    return _int_env("CLAIMFORGE_SOURCE_FAILURE_LIMIT", default=3)


def verify_rate_limit() -> int:
    """Maximum ``POST /verify`` requests per window. ``0`` disables the limit."""

    return _int_env("CLAIMFORGE_VERIFY_RATE_LIMIT", default=60)


def verify_rate_window_s() -> float:
    """Length of the ``POST /verify`` window in seconds. Default 60."""

    value = _float_env("CLAIMFORGE_VERIFY_RATE_WINDOW_S", default=60.0)
    if value <= 0:
        logger.warning("ignoring non-positive CLAIMFORGE_VERIFY_RATE_WINDOW_S; using 60")
        return 60.0
    return value


def fixture_dir() -> str:
    """Directory of offline evidence cassettes.

    Unset means ``data/fixtures/cassettes``, relative to the working directory.
    """

    raw = os.environ.get("CLAIMFORGE_FIXTURE_DIR", "").strip()
    return raw or "data/fixtures/cassettes"


def _int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r", name, raw)
        return default
    if value < 0:
        logger.warning("ignoring invalid %s=%r", name, raw)
        return default
    return value


def _float_env(name: str, *, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r", name, raw)
        return default
    if not math.isfinite(value) or value < 0:
        logger.warning("ignoring invalid %s=%r", name, raw)
        return default
    return value
