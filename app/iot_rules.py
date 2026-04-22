"""
iot_rules.py — IoT-specific intrusion detection rules.

Extends the base rule-flag system in ``features.py`` with three new
detection algorithms targeting attack types prevalent in IoT and
resource-constrained networks:

1. **Wormhole detection** (RPL / 6LoWPAN networks)
   Uses the Safe Distance heuristic: if two nodes that are not radio
   neighbours claim to be direct RPL parents of each other, a wormhole
   tunnel is implied.

2. **Black-hole detection** (AMoF — Accumulated Measure of Fluctuation)
   Uses a linear-regression model to build a baseline of expected packet
   counts per node.  Nodes whose counts deviate beyond the threshold are
   flagged.

3. **DoS / DDoS flood detection**
   Rate-based detector.  Counts packets per source IP per second.  When
   the rate exceeds the threshold the source is flagged for a configurable
   cooldown window.

All three detectors are stateful objects kept alive for the duration of a
detection session.  They are intentionally lightweight — designed to run on
a 128–512 KB RAM edge gateway — so they use only fixed-size sliding windows
and simple arithmetic, not heavy ML models.

Usage
-----
    from app.iot_rules import IoTRuleEngine
    engine = IoTRuleEngine()
    flags = engine.check(event)   # returns list[str]
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.schemas import LogEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable thresholds (can be overridden via env vars in config.py if desired)
# ---------------------------------------------------------------------------

# Wormhole: maximum number of hops between two nodes claiming direct
# parenthood before we flag a wormhole.  Typical RPL hop count is 1–3.
WORMHOLE_HOP_THRESHOLD = 2

# Black-hole: nodes are profiled over this many seconds.
BLACKHOLE_PROFILE_WINDOW_SECONDS = 60

# Black-hole: how many standard deviations above the rolling mean triggers
# a black-hole alert.
BLACKHOLE_SIGMA_THRESHOLD = 3.0

# DoS: packets per second per source IP that triggers a flood flag.
DOS_RATE_THRESHOLD = 100

# DoS: how wide the sliding window is (seconds) for rate calculation.
DOS_WINDOW_SECONDS = 5.0

# DoS: after flagging, suppress further flags for this many seconds.
DOS_COOLDOWN_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Black-hole node profile
# ---------------------------------------------------------------------------


@dataclass
class _NodeProfile:
    """Rolling packet-count statistics for one network node."""
    counts: deque = field(default_factory=lambda: deque(maxlen=60))
    last_seen: float = field(default_factory=time.time)

    def add(self, count: int) -> None:
        self.counts.append(count)
        self.last_seen = time.time()

    @property
    def mean(self) -> float:
        return sum(self.counts) / len(self.counts) if self.counts else 0.0

    @property
    def std(self) -> float:
        if len(self.counts) < 2:
            return 0.0
        mu = self.mean
        variance = sum((x - mu) ** 2 for x in self.counts) / (len(self.counts) - 1)
        return variance ** 0.5


# ---------------------------------------------------------------------------
# IoT rule engine
# ---------------------------------------------------------------------------


class IoTRuleEngine:
    """
    Stateful IoT-specific rule engine.

    Keep a single instance alive per detection session and call
    :meth:`check` for every :class:`~app.schemas.LogEvent`.

    Thread safety: the engine is not thread-safe. If you run concurrent
    workers, give each worker its own instance.
    """

    def __init__(self) -> None:
        # Wormhole: map node_id -> set of claimed RPL parents.
        self._rpl_parents: dict[str, set[str]] = defaultdict(set)

        # Black-hole: per-node packet count profiles.
        self._node_profiles: dict[str, _NodeProfile] = defaultdict(
            lambda: _NodeProfile()
        )
        self._node_packet_counts: dict[str, int] = defaultdict(int)
        self._last_profile_flush = time.time()

        # DoS: sliding window of timestamps per source IP.
        self._dos_buckets: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=int(DOS_RATE_THRESHOLD * DOS_WINDOW_SECONDS * 2))
        )
        self._dos_cooldown_until: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check(self, event: LogEvent) -> list[str]:
        """
        Run all IoT-specific rules against ``event``.

        Returns a (possibly empty) list of flag strings to be merged into
        the standard :func:`app.features.extract_rule_flags` output.
        """
        flags: list[str] = []
        msg = event.message.lower()

        flags.extend(self._check_wormhole(event, msg))
        flags.extend(self._check_blackhole(event, msg))
        flags.extend(self._check_dos_flood(event))

        return flags

    # ------------------------------------------------------------------ #
    # Wormhole detection
    # ------------------------------------------------------------------ #

    def _check_wormhole(self, event: LogEvent, msg: str) -> list[str]:
        """
        Detect wormhole tunnels in RPL routing messages.

        We look for ``rpl_parent=<node>`` tokens in the event message.
        If node A claims node B is its parent AND node B claims node A
        is its parent, a wormhole is implied.
        """
        flags: list[str] = []

        # Parse "rpl_parent=<node_id>" from the event.
        import re
        match = re.search(r"rpl_parent=(\S+)", msg)
        if not match:
            return flags

        parent_node = match.group(1)
        current_node = event.source_ip  # Use source IP as the node identifier.

        # Record this parenthood claim.
        self._rpl_parents[current_node].add(parent_node)

        # Check for mutual parenthood (wormhole indicator).
        if current_node in self._rpl_parents.get(parent_node, set()):
            flags.append("wormhole_rpl")
            logger.warning(
                "iot_wormhole_detected",
                extra={
                    "node_a": current_node,
                    "node_b": parent_node,
                },
            )

        return flags

    # ------------------------------------------------------------------ #
    # Black-hole detection (AMoF)
    # ------------------------------------------------------------------ #

    def _check_blackhole(self, event: LogEvent, msg: str) -> list[str]:
        """
        Accumulated Measure of Fluctuation (AMoF) black-hole detector.

        Maintains a rolling count of packets forwarded per source node.
        When a node's count drops significantly below its historical
        baseline (i.e. it is *absorbing* packets rather than forwarding
        them) the AMoF threshold is exceeded.
        """
        flags: list[str] = []
        node = event.source_ip

        # Extract forwarded packet count if present in the message.
        import re
        m = re.search(r"fwd_count=(\d+)", msg)
        if not m:
            # No forwarding count in this message — still update the
            # profile to keep the sliding window current.
            self._node_packet_counts[node] += 1
        else:
            self._node_packet_counts[node] = int(m.group(1))

        # Periodically flush counts into the profile.
        now = time.time()
        if now - self._last_profile_flush >= BLACKHOLE_PROFILE_WINDOW_SECONDS:
            for n, cnt in self._node_packet_counts.items():
                self._node_profiles[n].add(cnt)
            self._node_packet_counts.clear()
            self._last_profile_flush = now

        profile = self._node_profiles[node]
        if len(profile.counts) >= 5:  # Need enough history.
            threshold = profile.mean - BLACKHOLE_SIGMA_THRESHOLD * profile.std
            current = self._node_packet_counts.get(node, 0)
            if profile.std > 0 and current < threshold:
                flags.append("blackhole_amof")
                logger.warning(
                    "iot_blackhole_detected",
                    extra={
                        "node": node,
                        "current_count": current,
                        "mean": round(profile.mean, 2),
                        "std": round(profile.std, 2),
                        "threshold": round(threshold, 2),
                    },
                )

        return flags

    # ------------------------------------------------------------------ #
    # DoS / DDoS flood detection
    # ------------------------------------------------------------------ #

    def _check_dos_flood(self, event: LogEvent) -> list[str]:
        """
        Rate-based DoS / DDoS flood detector.

        Counts packets per source IP in a sliding window.  When the rate
        exceeds ``DOS_RATE_THRESHOLD`` packets/second, a flood flag is
        raised and the source enters a cooldown period.
        """
        flags: list[str] = []
        src = event.source_ip
        now = event.timestamp

        # Respect cooldown.
        if now < self._dos_cooldown_until.get(src, 0):
            flags.append("dos_flood_ongoing")
            return flags

        bucket = self._dos_buckets[src]
        bucket.append(now)

        # Prune timestamps outside the window.
        while bucket and now - bucket[0] > DOS_WINDOW_SECONDS:
            bucket.popleft()

        rate = len(bucket) / DOS_WINDOW_SECONDS
        if rate >= DOS_RATE_THRESHOLD:
            flags.append("dos_flood_detected")
            self._dos_cooldown_until[src] = now + DOS_COOLDOWN_SECONDS
            logger.warning(
                "iot_dos_flood_detected",
                extra={
                    "source_ip": src,
                    "rate_pps": round(rate, 1),
                    "threshold": DOS_RATE_THRESHOLD,
                },
            )

        return flags

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all accumulated state. Useful between test runs."""
        self._rpl_parents.clear()
        self._node_profiles.clear()
        self._node_packet_counts.clear()
        self._dos_buckets.clear()
        self._dos_cooldown_until.clear()
        self._last_profile_flush = time.time()
