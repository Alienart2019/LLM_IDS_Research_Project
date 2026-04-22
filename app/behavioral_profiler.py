"""
behavioral_profiler.py — Global Network Behavioral Profiling.

Two complementary subsystems:

1. **AMoF Profiler** (Accumulated Measure of Fluctuation)
   Profiles every network node over time using iterative linear regression.
   At each tick the profiler fits a slope to the node's recent packet-count
   series and accumulates the *change* in slope (the "fluctuation").  Nodes
   whose accumulated fluctuation exceeds a threshold are flagged as behaving
   anomalously — this is particularly sensitive to Black Hole attacks where
   a node quietly stops forwarding packets without triggering rate-based
   detectors.

2. **Dedicated Sniffer (DS) Hierarchy**
   Models a two-tier architecture:

   * **Leaf sniffers** run in promiscuous mode on network segments.  They
     collect raw events and forward ``CorrectlyClassifiedInstance`` (CCI)
     messages to the super node only when the local SGD model classifies
     an event with confidence above a threshold.  This dramatically reduces
     the volume of traffic the super node has to process.
   * **Super node** aggregates CCIs from all leaf sniffers and applies a
     global voting policy.  If a majority of sniffers agree that an IP is
     malicious, a global alert is raised.

Both systems are self-contained in-process objects suitable for embedding
in the existing ``IDSDetector`` or running as a background service.

Usage
-----
    from app.behavioral_profiler import AMoFProfiler, SuperNode, LeafSniffer

    # AMoF — call tick() every profiling_interval seconds
    profiler = AMoFProfiler()
    profiler.record(source_ip="10.0.0.5", packet_count=42)
    flags = profiler.tick()          # returns list of anomalous node IPs

    # DS hierarchy
    super_node = SuperNode()
    sniffer_a  = LeafSniffer("segment-A", super_node)
    sniffer_a.ingest(detection_result)   # sends CCI to super node if confident
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from app.schemas import DetectionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# AMoF: how many recent slope measurements to keep per node.
AMOF_SLOPE_WINDOW = 10

# AMoF: accumulated fluctuation threshold for anomaly flag.
AMOF_FLUCTUATION_THRESHOLD = 5.0

# AMoF: minimum number of data points before the regression is trusted.
AMOF_MIN_POINTS = 6

# DS: CCI forwarding confidence threshold — only forward if the local
# model is this confident or more.
CCI_CONFIDENCE_THRESHOLD = 0.75

# DS: fraction of sniffers that must agree for a global malicious verdict.
SUPER_NODE_CONSENSUS_FRACTION = 0.6

# DS: how many recent CCI records to keep per (sniffer, ip) pair.
CCI_HISTORY_DEPTH = 50


# ---------------------------------------------------------------------------
# AMoF Profiler
# ---------------------------------------------------------------------------


@dataclass
class _NodeAMoFState:
    """Per-node AMoF bookkeeping."""
    counts: deque = field(default_factory=lambda: deque(maxlen=60))
    slopes: deque = field(default_factory=lambda: deque(maxlen=AMOF_SLOPE_WINDOW))
    accumulated_fluctuation: float = 0.0
    last_slope: float = 0.0


def _linear_regression_slope(ys: list[float]) -> float:
    """Return the slope of the least-squares line through (0..n-1, ys)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


class AMoFProfiler:
    """
    Accumulated Measure of Fluctuation profiler.

    Records packet counts per node and, on each :meth:`tick`, fits a
    regression slope to each node's recent history.  The *change* between
    consecutive slopes is accumulated.  A large accumulated fluctuation
    indicates a node whose forwarding behaviour is irregular — the hallmark
    of a Black Hole or selective-drop attacker.

    Thread safety: uses a per-instance lock; safe for concurrent callers.
    """

    def __init__(
        self,
        fluctuation_threshold: float = AMOF_FLUCTUATION_THRESHOLD,
    ) -> None:
        self._threshold = fluctuation_threshold
        self._nodes: dict[str, _NodeAMoFState] = defaultdict(
            lambda: _NodeAMoFState()
        )
        self._lock = threading.Lock()

    def record(self, source_ip: str, packet_count: int) -> None:
        """
        Record a packet count observation for ``source_ip``.

        Call this every time the node sends a packet or batch of packets.
        The profiler accumulates observations; :meth:`tick` processes them.
        """
        with self._lock:
            self._nodes[source_ip].counts.append(float(packet_count))

    def tick(self) -> list[str]:
        """
        Process all accumulated observations and return a list of
        source IPs whose AMoF has exceeded the threshold.

        This should be called on a regular schedule (e.g. every 30 s).
        After each call the accumulated fluctuation of non-anomalous nodes
        is decayed by 20 % to prevent false positives from stale history.
        """
        anomalous: list[str] = []

        with self._lock:
            for ip, state in self._nodes.items():
                if len(state.counts) < AMOF_MIN_POINTS:
                    continue

                slope = _linear_regression_slope(list(state.counts))
                state.slopes.append(slope)

                if len(state.slopes) >= 2:
                    fluctuation = abs(slope - state.last_slope)
                    state.accumulated_fluctuation += fluctuation

                state.last_slope = slope

                if state.accumulated_fluctuation >= self._threshold:
                    anomalous.append(ip)
                    logger.warning(
                        "amof_anomaly_detected",
                        extra={
                            "source_ip": ip,
                            "accumulated_fluctuation": round(
                                state.accumulated_fluctuation, 3
                            ),
                            "threshold": self._threshold,
                            "current_slope": round(slope, 4),
                        },
                    )
                else:
                    # Decay to prevent indefinite build-up on noisy-but-benign nodes.
                    state.accumulated_fluctuation *= 0.80

        return anomalous

    def reset_node(self, source_ip: str) -> None:
        """Clear all history for a node (e.g. after a confirmed alert)."""
        with self._lock:
            self._nodes.pop(source_ip, None)

    def get_state(self, source_ip: str) -> Optional[dict]:
        """Return a snapshot of a node's AMoF state for diagnostics."""
        with self._lock:
            state = self._nodes.get(source_ip)
            if state is None:
                return None
            return {
                "accumulated_fluctuation": round(state.accumulated_fluctuation, 4),
                "last_slope": round(state.last_slope, 4),
                "count_history_len": len(state.counts),
                "slope_history": [round(s, 4) for s in state.slopes],
            }


# ---------------------------------------------------------------------------
# Dedicated Sniffer (DS) Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class CCI:
    """
    Correctly Classified Instance — the unit of communication from a leaf
    sniffer to the super node.

    Only high-confidence classifications are forwarded to keep inter-node
    bandwidth low.
    """
    sniffer_id: str
    source_ip: str
    predicted_label: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    rule_flags: list[str] = field(default_factory=list)


class SuperNode:
    """
    Aggregates CCIs from multiple leaf sniffers and applies a global
    voting policy.

    A source IP is declared globally malicious when a fraction ≥
    ``SUPER_NODE_CONSENSUS_FRACTION`` of active sniffers have sent a
    malicious CCI for that IP within the last ``window_seconds``.

    The super node can be polled by the API ``/alerts`` endpoint or run
    a background thread to push consensus alerts to the storage layer.
    """

    def __init__(
        self,
        consensus_fraction: float = SUPER_NODE_CONSENSUS_FRACTION,
        window_seconds: float = 300.0,
    ) -> None:
        self._consensus_fraction = consensus_fraction
        self._window_seconds = window_seconds
        # {source_ip: {sniffer_id: deque[CCI]}}
        self._cci_store: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=CCI_HISTORY_DEPTH))
        )
        self._registered_sniffers: set[str] = set()
        self._lock = threading.Lock()

    def register_sniffer(self, sniffer_id: str) -> None:
        with self._lock:
            self._registered_sniffers.add(sniffer_id)

    def receive_cci(self, cci: CCI) -> Optional[str]:
        """
        Accept a CCI from a leaf sniffer.

        Returns a verdict string (``"malicious"`` | ``"suspicious"`` |
        ``None``) if global consensus has been reached for
        ``cci.source_ip``.
        """
        with self._lock:
            self._cci_store[cci.source_ip][cci.sniffer_id].append(cci)
            return self._check_consensus(cci.source_ip)

    def _check_consensus(self, source_ip: str) -> Optional[str]:
        """Internal: evaluate consensus for a source IP (lock must be held)."""
        now = time.time()
        sniffer_votes: dict[str, str] = {}

        for sniffer_id, records in self._cci_store[source_ip].items():
            # Most recent CCI from this sniffer within the window.
            recent = [
                r for r in records
                if now - r.timestamp <= self._window_seconds
            ]
            if recent:
                # Take the most severe recent verdict from this sniffer.
                labels = [r.predicted_label for r in recent]
                if "malicious" in labels:
                    sniffer_votes[sniffer_id] = "malicious"
                elif "suspicious" in labels:
                    sniffer_votes[sniffer_id] = "suspicious"

        total = len(self._registered_sniffers) or len(sniffer_votes)
        if total == 0:
            return None

        mal_count = sum(1 for v in sniffer_votes.values() if v == "malicious")
        sus_count = sum(1 for v in sniffer_votes.values() if v == "suspicious")

        if mal_count / total >= self._consensus_fraction:
            logger.warning(
                "super_node_global_malicious_consensus",
                extra={
                    "source_ip": source_ip,
                    "malicious_sniffers": mal_count,
                    "total_sniffers": total,
                },
            )
            return "malicious"

        if (mal_count + sus_count) / total >= self._consensus_fraction:
            return "suspicious"

        return None

    def active_threats(self, window_seconds: Optional[float] = None) -> list[dict]:
        """
        Return all source IPs that currently have a consensus verdict.

        Useful for the API ``/stats`` or a dedicated ``/threats`` endpoint.
        """
        ws = window_seconds or self._window_seconds
        now = time.time()
        results = []

        with self._lock:
            for ip, sniffer_map in self._cci_store.items():
                verdict = self._check_consensus(ip)
                if verdict:
                    all_ccis = [
                        r
                        for records in sniffer_map.values()
                        for r in records
                        if now - r.timestamp <= ws
                    ]
                    results.append({
                        "source_ip": ip,
                        "verdict": verdict,
                        "sniffer_count": len(sniffer_map),
                        "cci_count": len(all_ccis),
                        "latest_ts": max((r.timestamp for r in all_ccis), default=0.0),
                    })

        return results


class LeafSniffer:
    """
    Simulates a dedicated sniffer node operating in promiscuous mode.

    In a real deployment each ``LeafSniffer`` instance would be a
    separate process or container running close to a network segment.
    Here it is modelled as an in-process object for integration testing.

    Only events whose confidence exceeds ``CCI_CONFIDENCE_THRESHOLD`` are
    forwarded to the super node — this filters out borderline cases and
    keeps inter-node traffic low.
    """

    def __init__(
        self,
        sniffer_id: str,
        super_node: SuperNode,
        confidence_threshold: float = CCI_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.sniffer_id = sniffer_id
        self._super_node = super_node
        self._threshold = confidence_threshold
        super_node.register_sniffer(sniffer_id)
        logger.info(
            "leaf_sniffer_registered",
            extra={
                "sniffer_id": sniffer_id,
                "confidence_threshold": confidence_threshold,
            },
        )

    def ingest(self, result: DetectionResult) -> Optional[str]:
        """
        Process a :class:`~app.schemas.DetectionResult` from the local
        detection pipeline.

        If the result is high-confidence and non-benign, a CCI is forwarded
        to the super node.  Returns the super node's consensus verdict (if
        any), or ``None``.
        """
        if result.predicted_label == "benign":
            return None
        if result.confidence < self._threshold:
            return None

        cci = CCI(
            sniffer_id=self.sniffer_id,
            source_ip=result.source_ip,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            timestamp=result.timestamp,
            rule_flags=list(result.rule_flags),
        )

        verdict = self._super_node.receive_cci(cci)

        if verdict:
            logger.warning(
                "leaf_sniffer_cci_consensus_reached",
                extra={
                    "sniffer_id": self.sniffer_id,
                    "source_ip": result.source_ip,
                    "verdict": verdict,
                },
            )

        return verdict
