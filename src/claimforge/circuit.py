"""Process-local skip after consecutive catalog failures.

The counter lives in memory for this process. It is not written to disk.
A successful search resets that source. After ``N`` consecutive hard
failures, later retrievals in the same process do not call that source.
In a multi-claim ``verify`` or ``retrieve-evidence`` command, those later
retrievals are the rest of the call. In ``claimforge serve``, the skip
lasts until the process exits. Restart the process to try the source again.

``N`` defaults to 3 (``CLAIMFORGE_SOURCE_FAILURE_LIMIT``). ``0`` disables
the skip. Semantic Scholar HTTP 401, 403, and 429 still return an empty
contribution from that client. They are not hard failures, so they do not
open the skip. Timeouts, connection errors, retry-exhausted 429 and 5xx,
and unexpected exceptions do.
"""

from __future__ import annotations

import threading

from claimforge.settings import source_failure_limit


class SourceCircuit:
    """Count consecutive hard failures and refuse a source once the limit is hit."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        self.limit = limit
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()

    def allow(self, source: str) -> bool:
        """False when this source has already reached the failure limit."""

        with self._lock:
            if self.limit <= 0:
                return True
            return self._failures.get(source, 0) < self.limit

    def record_failure(self, source: str) -> int:
        """Increment the streak. Return the new count. ``0`` when the skip is disabled."""

        with self._lock:
            if self.limit <= 0:
                return 0
            count = self._failures.get(source, 0) + 1
            self._failures[source] = count
            return count

    def record_success(self, source: str) -> None:
        """Clear the streak so the next failure starts again at one."""

        with self._lock:
            self._failures[source] = 0

    def reset(self) -> None:
        """Clear every streak. Tests use this between cases."""

        with self._lock:
            self._failures.clear()

    def failures(self, source: str) -> int:
        with self._lock:
            return self._failures.get(source, 0)


_process_circuit = SourceCircuit(limit=3)
_process_lock = threading.Lock()


def process_source_circuit() -> SourceCircuit:
    """The circuit shared by every retrieval in this process.

    The limit is refreshed from the environment on each lookup so a command
    sees ``CLAIMFORGE_SOURCE_FAILURE_LIMIT`` without a restart of the
    counter itself.
    """

    with _process_lock:
        _process_circuit.limit = source_failure_limit()
        return _process_circuit


def reset_process_source_circuit() -> None:
    """Drop process streaks. Does not change the configured limit."""

    process_source_circuit().reset()
