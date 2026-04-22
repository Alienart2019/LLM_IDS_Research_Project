"""
trust_manager.py — Trust Management and Protocol Verification.

Two subsystems:

1. **Trust Manager** (NBTD / CNTD)
   Maintains a trust score in [0.0, 1.0] for every network node.  Scores
   are updated based on classification verdicts from the IDS and from
   neighbouring nodes (trust dissemination).  Two dissemination strategies
   are supported:

   * **NBTD** (Neighbour-Based Trust Dissemination) — each node's score is
     a weighted average of its own behaviour and its direct neighbours'
     opinions.
   * **CNTD** (Clustered Neighbour-Based Trust Dissemination) — same idea
     but applied within a cluster, so cluster members weight their
     neighbours' opinions more heavily and ignore distant nodes.

   Trust scores can be used to weight alert severity: a low-trust source
   escalates alerts while a high-trust source dampens them.

2. **Protocol Automaton**
   Uses finite automaton theory to validate that 6LoWPAN / CoAP message
   sequences follow expected protocol state transitions.  Any transition
   that is not in the automaton's edge set is flagged as a protocol
   violation.

   The automaton is extensible: call :meth:`ProtocolAutomaton.add_transition`
   to add new protocols or extend existing ones.

Usage
-----
    from app.trust_manager import TrustManager, ProtocolAutomaton
    from app.schemas import DetectionResult

    tm = TrustManager()
    tm.update_from_detection(result)
    score = tm.get_score("10.0.0.5")    # 0.0 = untrusted, 1.0 = fully trusted

    pa = ProtocolAutomaton()
    violation = pa.transition("10.0.0.5", "coap", "CON", "ACK")
    if violation:
        print("Protocol violation:", violation)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from app.schemas import DetectionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Initial trust score for any new node.
INITIAL_TRUST_SCORE = 0.7

# How much a single malicious verdict degrades trust.
TRUST_PENALTY_MALICIOUS = 0.25
TRUST_PENALTY_SUSPICIOUS = 0.10
TRUST_REWARD_BENIGN = 0.02

# Minimum / maximum trust clamps.
TRUST_MIN = 0.0
TRUST_MAX = 1.0

# Weight of neighbour opinions vs. direct observations (NBTD).
NBTD_NEIGHBOUR_WEIGHT = 0.3

# Clustered NBTD: within-cluster neighbour weight.
CNTD_CLUSTER_WEIGHT = 0.4
CNTD_OUTSIDE_WEIGHT = 0.1

# How long (seconds) a trust score record is kept without updates.
TRUST_TTL_SECONDS = 3600.0


# ---------------------------------------------------------------------------
# Trust score record
# ---------------------------------------------------------------------------


@dataclass
class TrustRecord:
    score: float = INITIAL_TRUST_SCORE
    malicious_count: int = 0
    suspicious_count: int = 0
    benign_count: int = 0
    cluster_id: str = ""
    last_updated: float = field(default_factory=time.time)
    neighbours: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Trust Manager
# ---------------------------------------------------------------------------


class TrustManager:
    """
    Maintains per-node trust scores and propagates them via NBTD or CNTD.

    Thread safety: uses a per-instance lock.

    Integration
    -----------
    Call :meth:`update_from_detection` after each :class:`DetectionResult`
    to adjust the source node's score.  Then call
    :meth:`severity_multiplier` to get a severity modifier (values > 1.0
    escalate; < 1.0 dampen) to apply before storing the alert.
    """

    def __init__(self, strategy: str = "nbtd") -> None:
        """
        Parameters
        ----------
        strategy:
            ``"nbtd"`` for Neighbour-Based Trust Dissemination, or
            ``"cntd"`` for Clustered Neighbour-Based Trust Dissemination.
        """
        if strategy not in ("nbtd", "cntd"):
            raise ValueError("strategy must be 'nbtd' or 'cntd'")
        self._strategy = strategy
        self._records: dict[str, TrustRecord] = {}
        self._lock = threading.Lock()

    # ── Score management ────────────────────────────────────────────────

    def get_score(self, ip: str) -> float:
        with self._lock:
            return self._records.get(ip, TrustRecord()).score

    def _get_or_create(self, ip: str) -> TrustRecord:
        """Return (and create if needed) the TrustRecord for ``ip``."""
        if ip not in self._records:
            self._records[ip] = TrustRecord()
        return self._records[ip]

    def update_from_detection(self, result: DetectionResult) -> float:
        """
        Adjust the trust score of ``result.source_ip`` based on the
        detection outcome.

        Returns the updated score.
        """
        with self._lock:
            rec = self._get_or_create(result.source_ip)
            rec.last_updated = time.time()

            if result.predicted_label == "malicious":
                rec.score -= TRUST_PENALTY_MALICIOUS
                rec.malicious_count += 1
            elif result.predicted_label == "suspicious":
                rec.score -= TRUST_PENALTY_SUSPICIOUS
                rec.suspicious_count += 1
            else:
                rec.score += TRUST_REWARD_BENIGN
                rec.benign_count += 1

            rec.score = max(TRUST_MIN, min(TRUST_MAX, rec.score))

            # Propagate to neighbours.
            self._disseminate(result.source_ip, rec)

            logger.debug(
                "trust_score_updated",
                extra={
                    "ip": result.source_ip,
                    "label": result.predicted_label,
                    "new_score": round(rec.score, 3),
                },
            )
            return rec.score

    def register_neighbour(self, ip: str, neighbour_ip: str) -> None:
        """Declare that ``ip`` and ``neighbour_ip`` are direct neighbours."""
        with self._lock:
            self._get_or_create(ip).neighbours.add(neighbour_ip)
            self._get_or_create(neighbour_ip).neighbours.add(ip)

    def assign_cluster(self, ip: str, cluster_id: str) -> None:
        """Assign ``ip`` to a cluster (for CNTD)."""
        with self._lock:
            self._get_or_create(ip).cluster_id = cluster_id

    # ── Trust dissemination ─────────────────────────────────────────────

    def _disseminate(self, ip: str, rec: TrustRecord) -> None:
        """
        Propagate ``ip``'s trust change to its neighbours.

        The dissemination is intentionally asymmetric: a *penalty* is
        propagated to neighbours (a malicious node taints its peers) but
        a *reward* is not (we don't elevate other nodes just because one
        behaved well).
        """
        if not rec.neighbours:
            return

        penalty = 0.0
        if rec.score < INITIAL_TRUST_SCORE:
            penalty = (INITIAL_TRUST_SCORE - rec.score)

        if penalty == 0.0:
            return

        for neighbour_ip in rec.neighbours:
            if neighbour_ip not in self._records:
                continue
            neighbour = self._records[neighbour_ip]

            if self._strategy == "nbtd":
                influence = penalty * NBTD_NEIGHBOUR_WEIGHT
            else:
                # CNTD: stronger influence within the same cluster.
                same_cluster = (
                    rec.cluster_id
                    and neighbour.cluster_id == rec.cluster_id
                )
                influence = penalty * (
                    CNTD_CLUSTER_WEIGHT if same_cluster else CNTD_OUTSIDE_WEIGHT
                )

            neighbour.score -= influence
            neighbour.score = max(TRUST_MIN, neighbour.score)

    # ── Severity modifier ───────────────────────────────────────────────

    def severity_multiplier(self, ip: str) -> float:
        """
        Return a severity multiplier based on ``ip``'s trust score.

        * Score < 0.3 → 1.5  (low-trust node: escalate)
        * Score 0.3–0.7 → 1.0 (neutral)
        * Score > 0.7 → 0.8  (high-trust node: slight dampening)
        """
        score = self.get_score(ip)
        if score < 0.3:
            return 1.5
        if score > 0.7:
            return 0.8
        return 1.0

    # ── Housekeeping ────────────────────────────────────────────────────

    def evict_stale(self) -> int:
        """Remove records that have not been updated within TTL. Returns count."""
        now = time.time()
        with self._lock:
            stale = [
                ip for ip, rec in self._records.items()
                if now - rec.last_updated > TRUST_TTL_SECONDS
            ]
            for ip in stale:
                del self._records[ip]
        return len(stale)

    def snapshot(self) -> list[dict]:
        """Return all current trust records as a list of dicts."""
        with self._lock:
            return [
                {
                    "ip": ip,
                    "score": round(rec.score, 4),
                    "malicious_count": rec.malicious_count,
                    "suspicious_count": rec.suspicious_count,
                    "benign_count": rec.benign_count,
                    "cluster_id": rec.cluster_id,
                    "last_updated": rec.last_updated,
                }
                for ip, rec in self._records.items()
            ]


# ---------------------------------------------------------------------------
# Protocol Automaton
# ---------------------------------------------------------------------------


@dataclass
class ProtocolViolation:
    source_ip: str
    protocol: str
    from_state: str
    to_state: str
    message: str
    timestamp: float = field(default_factory=time.time)


class ProtocolAutomaton:
    """
    Finite automaton for detecting protocol-specific violations.

    The automaton is defined as a set of valid
    ``(protocol, from_state, to_state)`` triples.  Any observed transition
    that is not in this set is a violation.

    Designed for 6LoWPAN / CoAP environments but fully extensible.

    Pre-loaded transitions
    ----------------------
    CoAP:  CON → ACK, CON → RST, NON → (terminal), ACK → (terminal)
    RPL :  DIS → DIO, DIO → DAO, DAO → DAO_ACK
    """

    def __init__(self) -> None:
        # {protocol: {(from_state, to_state)}}
        self._transitions: dict[str, set[tuple[str, str]]] = defaultdict(set)
        # {source_ip: {protocol: current_state}}
        self._states: dict[str, dict[str, str]] = defaultdict(dict)
        self._lock = threading.Lock()

        self._load_defaults()

    def _load_defaults(self) -> None:
        # CoAP confirmable message flow.
        for t in [
            ("CON", "ACK"), ("CON", "RST"),
            ("NON", "TERMINAL"), ("ACK", "TERMINAL"),
            ("RST", "TERMINAL"),
        ]:
            self.add_transition("coap", *t)

        # RPL routing protocol message flow.
        for t in [
            ("DIS", "DIO"), ("DIO", "DAO"), ("DAO", "DAO_ACK"),
            ("DAO_ACK", "TERMINAL"), ("DIO", "TERMINAL"),
        ]:
            self.add_transition("rpl", *t)

        # Generic TCP handshake.
        for t in [
            ("CLOSED", "SYN_SENT"), ("SYN_SENT", "SYN_RCVD"),
            ("SYN_RCVD", "ESTABLISHED"), ("ESTABLISHED", "FIN_WAIT"),
            ("FIN_WAIT", "CLOSED"),
        ]:
            self.add_transition("tcp", *t)

    def add_transition(
        self, protocol: str, from_state: str, to_state: str
    ) -> None:
        """Register a valid state transition for ``protocol``."""
        with self._lock:
            self._transitions[protocol.lower()].add(
                (from_state.upper(), to_state.upper())
            )

    def transition(
        self,
        source_ip: str,
        protocol: str,
        event_state: str,
        next_state: Optional[str] = None,
    ) -> Optional[ProtocolViolation]:
        """
        Record a protocol state transition for ``source_ip``.

        Parameters
        ----------
        source_ip:
            The node performing the transition.
        protocol:
            Protocol name (``"coap"``, ``"rpl"``, ``"tcp"``).
        event_state:
            The state observed in the current message/packet.
        next_state:
            The state the node claims to be transitioning to.  If ``None``,
            the automaton infers it from the last known state.

        Returns
        -------
        ProtocolViolation or None
            A violation if the transition is not in the allowed set.
        """
        proto = protocol.lower()
        cur = event_state.upper()
        nxt = next_state.upper() if next_state else None

        with self._lock:
            known_state = self._states[source_ip].get(proto)

            # Determine the "from" state for validation.
            from_state = known_state or cur

            if nxt is not None:
                allowed = self._transitions.get(proto, set())
                if allowed and (from_state, nxt) not in allowed:
                    violation = ProtocolViolation(
                        source_ip=source_ip,
                        protocol=proto,
                        from_state=from_state,
                        to_state=nxt,
                        message=(
                            f"Invalid {proto.upper()} transition "
                            f"{from_state} → {nxt} from {source_ip}"
                        ),
                    )
                    logger.warning(
                        "protocol_violation_detected",
                        extra={
                            "source_ip": source_ip,
                            "protocol": proto,
                            "from_state": from_state,
                            "to_state": nxt,
                        },
                    )
                    return violation

                # Update known state.
                self._states[source_ip][proto] = nxt
            else:
                self._states[source_ip][proto] = cur

        return None

    def reset_node(self, source_ip: str, protocol: Optional[str] = None) -> None:
        """Clear state for a node (optionally for a specific protocol only)."""
        with self._lock:
            if protocol:
                self._states[source_ip].pop(protocol.lower(), None)
            else:
                self._states.pop(source_ip, None)
