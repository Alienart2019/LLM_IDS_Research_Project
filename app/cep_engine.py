"""
cep_engine.py — Complex Event Processing (CEP) engine.

Traditional IDS rule engines match individual events against a static
database.  CEP instead detects *patterns across a stream* of events in
real time — for example "three failed logins from the same IP within 60
seconds followed by a successful login" or "five different destination
ports scanned within 10 seconds".

Architecture
------------
The engine is built around three primitives:

* **Pattern** — a named sequence of predicates with a time window and a
  minimum match count.
* **PatternWindow** — a sliding buffer of events associated with one
  source IP that the engine inspects for pattern matches.
* **CEPEngine** — the coordinator.  Call ``process(event)`` and it
  returns a list of :class:`CEPAlert` objects for every pattern that
  fired.

Built-in patterns
-----------------
The following patterns are pre-registered and reflect the most common
multi-event attack signatures seen in IoT traffic:

* ``brute_force``       — N failed logins from the same IP within T seconds
* ``port_scan``         — N distinct dest ports visited within T seconds
* ``lateral_movement``  — N distinct dest hosts contacted within T seconds
* ``login_after_scan``  — port scan followed by a successful auth event
* ``data_exfil_burst``  — sustained high-volume outbound traffic bursts

Custom patterns can be added at runtime with :meth:`CEPEngine.register`.

Memory note
-----------
Research shows CEP uses *less* memory than store-and-query approaches
because events are processed and discarded in the sliding window rather
than written to a persistent store.  The trade-off is CPU — pattern
matching runs on every ingested event.

Usage
-----
    from app.cep_engine import CEPEngine
    from app.schemas import LogEvent

    engine = CEPEngine()
    alerts = engine.process(log_event)
    for alert in alerts:
        print(alert.pattern_name, alert.source_ip, alert.matched_count)
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

from app.schemas import LogEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CEPAlert:
    """Fired when a registered pattern matches against the event stream."""
    pattern_name: str
    source_ip: str
    matched_count: int
    window_seconds: float
    severity: str
    description: str
    matched_events: list[LogEvent] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Pattern:
    """
    A CEP pattern definition.

    Attributes
    ----------
    name:
        Unique identifier for this pattern.
    predicate:
        Callable that returns True when a :class:`~app.schemas.LogEvent`
        is relevant to this pattern (not necessarily a match — just a
        candidate that contributes to the count).
    window_seconds:
        Sliding time window in which ``min_count`` events must occur.
    min_count:
        Minimum number of matching events to fire the alert.
    severity:
        Severity string attached to the resulting :class:`CEPAlert`.
    description:
        Human-readable explanation of what the pattern detects.
    distinct_field:
        If set, count distinct values of this field (``"dst_port"``,
        ``"dst_host"``) instead of raw event count.  Used for port-scan
        and lateral-movement patterns.
    sequence:
        Optional list of sub-pattern names that must all fire (in any
        order) before this pattern fires.  Enables compound patterns like
        "scan then auth".
    """
    name: str
    predicate: Callable[[LogEvent], bool]
    window_seconds: float
    min_count: int
    severity: str
    description: str
    distinct_field: str | None = None
    sequence: list[str] | None = None


# ---------------------------------------------------------------------------
# Field extractors for distinct-count patterns
# ---------------------------------------------------------------------------


def _extract_dst_port(event: LogEvent) -> str | None:
    m = re.search(r"dst_port=(\d+)", event.message)
    return m.group(1) if m else None


def _extract_dst_host(event: LogEvent) -> str | None:
    m = re.search(r"dst_ip=(\S+)", event.message)
    return m.group(1) if m else None


_FIELD_EXTRACTORS: dict[str, Callable[[LogEvent], str | None]] = {
    "dst_port": _extract_dst_port,
    "dst_host": _extract_dst_host,
}


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------


def _is_failed_login(e: LogEvent) -> bool:
    msg = e.message.lower()
    return "failed password" in msg or "login failed" in msg or "authentication failure" in msg


def _is_successful_login(e: LogEvent) -> bool:
    msg = e.message.lower()
    return "accepted password" in msg or "session opened" in msg or "login successful" in msg


def _has_dst_port(e: LogEvent) -> bool:
    return bool(re.search(r"dst_port=\d+", e.message))


def _has_dst_ip(e: LogEvent) -> bool:
    return bool(re.search(r"dst_ip=\d", e.message))


def _is_high_volume(e: LogEvent) -> bool:
    # Rough proxy: packet present in service=tcp/udp with large payload indicator
    msg = e.message.lower()
    return (e.service in ("tcp", "udp")) and ("payload=" in msg or "bytes=" in msg)


_BUILTIN_PATTERNS: list[Pattern] = [
    Pattern(
        name="brute_force",
        predicate=_is_failed_login,
        window_seconds=60.0,
        min_count=5,
        severity="high",
        description="5+ failed login attempts from the same IP within 60 seconds",
    ),
    Pattern(
        name="port_scan",
        predicate=_has_dst_port,
        window_seconds=10.0,
        min_count=5,
        severity="high",
        description="5+ distinct destination ports contacted within 10 seconds",
        distinct_field="dst_port",
    ),
    Pattern(
        name="lateral_movement",
        predicate=_has_dst_ip,
        window_seconds=30.0,
        min_count=4,
        severity="critical",
        description="4+ distinct destination hosts contacted within 30 seconds",
        distinct_field="dst_host",
    ),
    Pattern(
        name="login_after_scan",
        predicate=_is_successful_login,
        window_seconds=120.0,
        min_count=1,
        severity="critical",
        description="Successful login following a port scan from the same IP",
        sequence=["port_scan"],
    ),
    Pattern(
        name="data_exfil_burst",
        predicate=_is_high_volume,
        window_seconds=5.0,
        min_count=20,
        severity="high",
        description="20+ high-volume TCP/UDP events within 5 seconds (possible data exfiltration)",
    ),
]


# ---------------------------------------------------------------------------
# Per-IP sliding window
# ---------------------------------------------------------------------------


@dataclass
class _IPWindow:
    """Sliding event buffer for one source IP."""
    events: deque = field(default_factory=lambda: deque(maxlen=500))
    # pattern_name -> timestamp of last alert (for cooldown)
    last_alert: dict[str, float] = field(default_factory=dict)
    # pattern_name -> True if this pattern has fired (for sequence tracking)
    fired_patterns: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CEP Engine
# ---------------------------------------------------------------------------


class CEPEngine:
    """
    Real-time Complex Event Processing engine.

    Call :meth:`process` for every :class:`~app.schemas.LogEvent` arriving
    from the detection pipeline.  The engine maintains per-IP sliding windows
    and fires :class:`CEPAlert` objects whenever a pattern threshold is
    crossed.

    A 30-second per-pattern cooldown per IP prevents alert storms during
    sustained attacks.

    Thread safety: uses a per-instance lock.  Safe for concurrent callers.
    """

    ALERT_COOLDOWN_SECONDS = 30.0

    def __init__(self) -> None:
        self._patterns: dict[str, Pattern] = {}
        self._windows: dict[str, _IPWindow] = defaultdict(lambda: _IPWindow())
        import threading
        self._lock = threading.Lock()

        for pat in _BUILTIN_PATTERNS:
            self.register(pat)

        logger.info(
            "cep_engine_initialized",
            extra={"pattern_count": len(self._patterns)},
        )

    # ── Pattern registration ────────────────────────────────────────────

    def register(self, pattern: Pattern) -> None:
        """Add a pattern to the engine."""
        with self._lock:
            self._patterns[pattern.name] = pattern
            logger.debug(
                "cep_pattern_registered",
                extra={"name": pattern.name, "window_s": pattern.window_seconds},
            )

    def unregister(self, name: str) -> None:
        with self._lock:
            self._patterns.pop(name, None)

    # ── Main entry point ────────────────────────────────────────────────

    def process(self, event: LogEvent) -> list[CEPAlert]:
        """
        Ingest one event and return any :class:`CEPAlert` objects that fired.

        This is O(P × W) where P is the number of patterns and W is the
        average window size — typically a few microseconds per event.
        """
        with self._lock:
            ip = event.source_ip
            window = self._windows[ip]
            window.events.append(event)
            alerts = self._evaluate_all(ip, window, event)

        return alerts

    # ── Internal evaluation ─────────────────────────────────────────────

    def _evaluate_all(
        self,
        ip: str,
        window: _IPWindow,
        trigger_event: LogEvent,
    ) -> list[CEPAlert]:
        alerts: list[CEPAlert] = []
        now = trigger_event.timestamp

        for name, pattern in self._patterns.items():
            # Cooldown guard.
            last = window.last_alert.get(name, 0.0)
            if now - last < self.ALERT_COOLDOWN_SECONDS:
                continue

            alert = self._match_pattern(ip, window, pattern, now)
            if alert:
                window.last_alert[name] = now
                window.fired_patterns[name] = now
                alerts.append(alert)

                logger.warning(
                    "cep_pattern_fired",
                    extra={
                        "pattern": name,
                        "source_ip": ip,
                        "matched_count": alert.matched_count,
                        "severity": alert.severity,
                    },
                )

        return alerts

    def _match_pattern(
        self,
        ip: str,
        window: _IPWindow,
        pattern: Pattern,
        now: float,
    ) -> CEPAlert | None:
        # Sequence prerequisite check.
        if pattern.sequence:
            for prereq in pattern.sequence:
                fired_at = window.fired_patterns.get(prereq, 0.0)
                # Prerequisite must have fired within the pattern's window.
                if now - fired_at > pattern.window_seconds:
                    return None

        # Collect candidate events within the window.
        candidates = [
            e for e in window.events
            if (now - e.timestamp) <= pattern.window_seconds
            and pattern.predicate(e)
        ]

        if pattern.distinct_field:
            extractor = _FIELD_EXTRACTORS.get(pattern.distinct_field)
            if extractor:
                values = set(
                    v for e in candidates
                    if (v := extractor(e)) is not None
                )
                count = len(values)
            else:
                count = len(candidates)
        else:
            count = len(candidates)

        if count < pattern.min_count:
            return None

        return CEPAlert(
            pattern_name=pattern.name,
            source_ip=ip,
            matched_count=count,
            window_seconds=pattern.window_seconds,
            severity=pattern.severity,
            description=pattern.description,
            matched_events=candidates[-10:],   # keep the 10 most recent
        )

    # ── Utility ─────────────────────────────────────────────────────────

    def active_patterns(self) -> list[str]:
        with self._lock:
            return list(self._patterns.keys())

    def window_stats(self, source_ip: str) -> dict:
        with self._lock:
            w = self._windows.get(source_ip)
            if not w:
                return {}
            return {
                "buffered_events": len(w.events),
                "fired_patterns": {
                    k: round(v, 1) for k, v in w.fired_patterns.items()
                },
            }

    def flush_expired_windows(self, max_age_seconds: float = 600.0) -> int:
        """
        Remove per-IP windows that have not seen any events recently.

        Returns the number of windows pruned.  Call periodically to bound
        memory usage in long-running deployments.
        """
        now = time.time()
        with self._lock:
            stale = [
                ip for ip, w in self._windows.items()
                if w.events and now - w.events[-1].timestamp > max_age_seconds
            ]
            for ip in stale:
                del self._windows[ip]
        return len(stale)
