"""Reset process-local hardening state so tests do not share streaks or host timers."""

from __future__ import annotations

import pytest

from claimforge.circuit import reset_process_source_circuit
from claimforge.http_cache import reset_shared_host_interval_limiter


@pytest.fixture(autouse=True)
def _reset_hardening_state(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_process_source_circuit()
    reset_shared_host_interval_limiter()
    for name in (
        "CLAIMFORGE_OFFLINE",
        "CLAIMFORGE_HTTP_MIN_INTERVAL_S",
        "CLAIMFORGE_SOURCE_FAILURE_LIMIT",
        "CLAIMFORGE_VERIFY_RATE_LIMIT",
        "CLAIMFORGE_VERIFY_RATE_WINDOW_S",
    ):
        monkeypatch.delenv(name, raising=False)
