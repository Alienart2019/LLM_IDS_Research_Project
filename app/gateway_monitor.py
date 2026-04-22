"""
gateway_monitor.py — Specialized Gateway Packet-Drop Probability Monitor.

Gateways are critical single points of failure in IoT networks.  A
compromised gateway can silently drop or corrupt packets without triggering
any endpoint-level alerts.  This module monitors gateway health using a
statistical likelihood ratio test (LRT) on the observed packet-drop rate.

Algorithm
---------
For each gateway the monitor tracks two quantities over a sliding window:

* **forwarded** — packets the gateway reports as successfully relayed.
* **dropped**   — packets logged as dropped (by the gateway itself or by
  downstream nodes that detect missing sequence numbers).

The estimated drop probability ``p̂ = dropped / (forwarded + dropped)``.

Under normal operation ``p̂`` follows a stable distribution.  A sudden
shift is detected using the **Sequential Probability Ratio Test (SPRT)**,
a variant of the likelihood ratio test that decides between:

* H₀: drop probability = ``p0_normal``  (baseline, e.g. 0.01 = 1%)
* H₁: drop probability = ``p1_attack``  (elevated, e.g. 0.15 = 15%)

When the log-likelihood ratio crosses the upper threshold ``B``, H₁ is
accepted and a gateway alert is raised.

Reference: Wald's SPRT — thresholds derived from target error rates α and β
where ``A = log(β / (1-α))`` and ``B = log((1-β) / α)``.

Usage
-----
    from app.gateway_monitor import GatewayMonitor
    gm = GatewayMonitor()

    # Call on every gateway log line:
    alert = gm.observe("gw-01", forwarded=98, dropped=2)
    if alert:
        print(alert.message)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# H₀: expected normal packet drop probability.
PDP_NORMAL = 0.01        # 1% background drop rate

# H₁: drop probability that constitutes an attack.
PDP_ATTACK = 0.15        # 15% = suspicious gateway behaviour

# SPRT error rate targets (false alarm / miss).
SPRT_ALPHA = 0.05        # false alarm rate
SPRT_BETA  = 0.10        # miss rate

# SPRT decision boundaries derived from the above.
_SPRT_A = math.log(SPRT_BETA / (1 - SPRT_ALPHA))        # lower boundary (accept H₀)
_SPRT_B = math.log((1 - SPRT_BETA) / SPRT_ALPHA)        # upper boundary (reject H₀)

# Rolling window of observations per gateway.
OBSERVATION_WINDOW = 200

# Minimum observations before SPRT fires.
MIN_OBSERVATIONS = 20

# After an alert, suppress further alerts for this many seconds.
ALERT_COOLDOWN_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GatewayObservation:
    """One observation from a gateway log line."""
    timestamp: float
    forwarded: int
    dropped: int

    @property
    def total(self) -> int:
        return self.forwarded + self.dropped

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.total if self.total > 0 else 0.0


@dataclass
class GatewayAlert:
    """Raised when a gateway's drop probability crosses the SPRT threshold."""
    gateway_id: str
    sprt_statistic: float
    estimated_pdp: float
    observation_count: int
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "gateway_id": self.gateway_id,
            "sprt_statistic": round(self.sprt_statistic, 4),
            "estimated_pdp": round(self.estimated_pdp, 4),
            "observation_count": self.observation_count,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class _GatewayState:
    observations: deque = field(
        default_factory=lambda: deque(maxlen=OBSERVATION_WINDOW)
    )
    sprt_stat: float = 0.0           # Cumulative log-likelihood ratio
    last_alert_ts: float = 0.0
    total_forwarded: int = 0
    total_dropped: int = 0


# ---------------------------------------------------------------------------
# Gateway Monitor
# ---------------------------------------------------------------------------


class GatewayMonitor:
    """
    Sequential Probability Ratio Test monitor for gateway packet-drop
    probability.

    Each ``observe()`` call updates the SPRT statistic for the named gateway
    and returns a :class:`GatewayAlert` if H₁ is accepted (i.e. the gateway
    appears to be dropping an anomalously high fraction of packets).

    Thread safety: uses a per-instance lock.

    Integration
    -----------
    Parse gateway log lines for forwarded/dropped counts and pass them to
    :meth:`observe`.  The simplest integration point is inside
    ``detector.predict_event`` when the event service is ``"gateway"`` or
    the message contains ``fwd_count=`` / ``drop_count=`` tokens.
    """

    def __init__(
        self,
        p0: float = PDP_NORMAL,
        p1: float = PDP_ATTACK,
        alpha: float = SPRT_ALPHA,
        beta: float = SPRT_BETA,
    ) -> None:
        self._p0 = p0
        self._p1 = p1
        self._log_A = math.log(beta / (1 - alpha))
        self._log_B = math.log((1 - beta) / alpha)

        self._gateways: dict[str, _GatewayState] = defaultdict(
            lambda: _GatewayState()
        )
        self._lock = threading.Lock()

        logger.info(
            "gateway_monitor_initialized",
            extra={
                "p0_normal": p0,
                "p1_attack": p1,
                "threshold_B": round(self._log_B, 3),
                "threshold_A": round(self._log_A, 3),
            },
        )

    # ── Main entry point ────────────────────────────────────────────────

    def observe(
        self,
        gateway_id: str,
        forwarded: int,
        dropped: int,
        timestamp: Optional[float] = None,
    ) -> Optional[GatewayAlert]:
        """
        Record one observation for ``gateway_id`` and return an alert if
        the SPRT upper boundary is crossed.

        Parameters
        ----------
        gateway_id:
            Unique identifier for the gateway (IP or hostname).
        forwarded:
            Packets forwarded in this observation window.
        dropped:
            Packets dropped in this observation window.
        timestamp:
            Observation time.  Defaults to ``time.time()``.
        """
        ts = timestamp or time.time()
        obs = GatewayObservation(
            timestamp=ts,
            forwarded=forwarded,
            dropped=dropped,
        )

        with self._lock:
            state = self._gateways[gateway_id]
            state.observations.append(obs)
            state.total_forwarded += forwarded
            state.total_dropped += dropped

            # Update SPRT statistic with the log-likelihood ratio for this obs.
            if obs.total > 0:
                # Log-likelihood ratio contribution:
                # Σ dropped * log(p1/p0) + forwarded * log((1-p1)/(1-p0))
                llr = (
                    dropped * math.log(self._p1 / self._p0)
                    + forwarded * math.log((1 - self._p1) / (1 - self._p0))
                )
                state.sprt_stat += llr

            # Clamp at lower boundary (reset to zero when H₀ accepted).
            if state.sprt_stat <= self._log_A:
                state.sprt_stat = 0.0  # Reset — H₀ accepted, normal behaviour.

            # Check upper boundary.
            if (
                state.sprt_stat >= self._log_B
                and len(state.observations) >= MIN_OBSERVATIONS
                and ts - state.last_alert_ts > ALERT_COOLDOWN_SECONDS
            ):
                state.last_alert_ts = ts
                state.sprt_stat = 0.0  # Reset after alert.

                total = state.total_forwarded + state.total_dropped
                estimated_pdp = state.total_dropped / total if total > 0 else 0.0

                alert = GatewayAlert(
                    gateway_id=gateway_id,
                    sprt_statistic=state.sprt_stat,
                    estimated_pdp=estimated_pdp,
                    observation_count=len(state.observations),
                    message=(
                        f"Gateway {gateway_id} packet-drop probability "
                        f"{estimated_pdp:.1%} exceeds SPRT threshold "
                        f"(normal={self._p0:.1%}, attack={self._p1:.1%})"
                    ),
                    timestamp=ts,
                )

                logger.warning(
                    "gateway_pdp_alert",
                    extra={
                        "gateway_id": gateway_id,
                        "estimated_pdp": round(estimated_pdp, 4),
                        "sprt_stat": round(state.sprt_stat, 3),
                    },
                )
                return alert

        return None

    # ── Diagnostics ─────────────────────────────────────────────────────

    def gateway_status(self, gateway_id: str) -> dict:
        """Return current SPRT statistics for a gateway."""
        with self._lock:
            state = self._gateways.get(gateway_id)
            if state is None:
                return {"gateway_id": gateway_id, "status": "no_data"}
            total = state.total_forwarded + state.total_dropped
            return {
                "gateway_id": gateway_id,
                "total_forwarded": state.total_forwarded,
                "total_dropped": state.total_dropped,
                "estimated_pdp": round(
                    state.total_dropped / total if total > 0 else 0.0, 4
                ),
                "sprt_stat": round(state.sprt_stat, 4),
                "upper_threshold": round(self._log_B, 4),
                "observation_count": len(state.observations),
            }

    def all_gateway_statuses(self) -> list[dict]:
        with self._lock:
            return [
                self.gateway_status(gid) for gid in self._gateways
            ]

    def reset_gateway(self, gateway_id: str) -> None:
        """Clear all state for a gateway (e.g. after maintenance)."""
        with self._lock:
            self._gateways.pop(gateway_id, None)
