"""
physical_fusion.py — Physical-Cyber Security Fusion layer.

Bridges physical-world sensors with the network IDS so that physical
intrusion signals can corroborate (or escalate) cyber alerts, and vice
versa.

Two sensor types are supported:

1. **PIR (Passive Infrared) sensors** — detect motion in a monitored zone.
   When the IDS is already in a high-risk state *and* a PIR fires, the
   combined event is escalated to ``critical`` and an optional GSM/SMS
   notification is dispatched.

2. **IP cameras** — normally idle.  When the IDS labels an event
   ``malicious`` or ``critical``, the camera gateway is called to begin
   recording.  Recording stops automatically after a configurable cooldown
   window, conserving bandwidth and storage.

Architecture
------------
Both sensor adapters are intentionally thin HTTP/serial wrappers.  The
default implementations talk to:

- A REST-style PIR gateway (e.g. a Raspberry Pi running Flask that
  publishes ``GET /pir/state`` → ``{"motion": true|false}``).
- An ONVIF-lite camera gateway that accepts ``POST /record/start`` and
  ``POST /record/stop`` (replace with your real camera SDK as needed).

Enabling
--------
Set ``IDS_PHYSICAL_FUSION=true`` to activate the module at API startup.
PIR and camera endpoints are configured via env vars (see ``app/config.py``).

Usage
-----
    from app.physical_fusion import PhysicalFusionEngine
    engine = PhysicalFusionEngine()

    # Poll PIR and combine with a DetectionResult:
    fused = engine.fuse(detection_result)
    # fused.severity may be elevated; fused.rule_flags may include
    # "physical_motion_corroborated" or "camera_recording_started".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from app.config import (
    PHYSICAL_FUSION_ENABLED,
    PIR_GATEWAY_URL,
    PIR_POLL_INTERVAL_SECONDS,
    PIR_MOTION_WINDOW_SECONDS,
    CAMERA_GATEWAY_URL,
    CAMERA_RECORD_COOLDOWN_SECONDS,
    GSM_GATEWAY_URL,
    GSM_ALERT_PHONE,
    PHYSICAL_FUSION_TRIGGER_LABELS,
)
from app.schemas import DetectionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fused result wrapper
# ---------------------------------------------------------------------------


@dataclass
class FusedResult:
    """
    A :class:`~app.schemas.DetectionResult` enriched with physical-layer
    context.

    Attributes
    ----------
    base:
        The original network detection result (unmodified).
    physical_motion_detected:
        True if a PIR sensor reported motion within the fusion window.
    camera_recording_started:
        True if this event triggered a camera recording request.
    gsm_alert_sent:
        True if an SMS notification was dispatched.
    effective_severity:
        Possibly elevated severity after physical corroboration.
    extra_flags:
        Additional rule flags added by the fusion layer.
    """
    base: DetectionResult
    physical_motion_detected: bool = False
    camera_recording_started: bool = False
    gsm_alert_sent: bool = False
    effective_severity: str = ""
    extra_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.effective_severity:
            self.effective_severity = self.base.severity


# ---------------------------------------------------------------------------
# PIR sensor adapter
# ---------------------------------------------------------------------------


class PIRAdapter:
    """
    Polls a REST PIR gateway and maintains a short-lived motion window.

    The motion window prevents a single PIR trigger from elevating every
    event for too long — only events arriving within
    ``PIR_MOTION_WINDOW_SECONDS`` of the last detected motion are fused.
    """

    def __init__(self) -> None:
        self._last_motion_ts: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Background polling ──────────────────────────────────────────────

    def start_polling(self) -> None:
        """Start background thread that polls the PIR gateway."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="pir_poller"
        )
        self._thread.start()
        logger.info("pir_polling_started", extra={"url": PIR_GATEWAY_URL})

    def stop_polling(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                resp = requests.get(PIR_GATEWAY_URL, timeout=3.0)
                if resp.ok:
                    data = resp.json()
                    if data.get("motion"):
                        with self._lock:
                            self._last_motion_ts = time.time()
                        logger.warning(
                            "pir_motion_detected",
                            extra={"gateway": PIR_GATEWAY_URL},
                        )
            except Exception as exc:
                logger.debug(
                    "pir_poll_failed",
                    extra={"error": str(exc)},
                )
            time.sleep(PIR_POLL_INTERVAL_SECONDS)

    # ── Query ───────────────────────────────────────────────────────────

    def motion_within_window(self) -> bool:
        """Return True if motion was detected within the fusion window."""
        with self._lock:
            age = time.time() - self._last_motion_ts
        return age <= PIR_MOTION_WINDOW_SECONDS

    def inject_motion(self) -> None:
        """Programmatically inject a motion event (for testing)."""
        with self._lock:
            self._last_motion_ts = time.time()


# ---------------------------------------------------------------------------
# Camera adapter
# ---------------------------------------------------------------------------


class CameraAdapter:
    """
    Controls an IP camera recording gateway.

    Recording is started on demand and stops automatically after
    ``CAMERA_RECORD_COOLDOWN_SECONDS`` of inactivity.  A daemon timer
    thread handles the auto-stop so the detection path is never blocked.
    """

    def __init__(self) -> None:
        self._recording = False
        self._stop_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def start_recording(self) -> bool:
        """
        Request the camera gateway to begin recording.

        Returns True if the request was sent successfully.
        """
        with self._lock:
            # Reset the auto-stop timer if already recording.
            if self._stop_timer is not None:
                self._stop_timer.cancel()

            success = self._post("/record/start")
            if success:
                self._recording = True
                self._stop_timer = threading.Timer(
                    CAMERA_RECORD_COOLDOWN_SECONDS,
                    self._auto_stop,
                )
                self._stop_timer.daemon = True
                self._stop_timer.start()
                logger.warning(
                    "camera_recording_started",
                    extra={
                        "gateway": CAMERA_GATEWAY_URL,
                        "auto_stop_in_s": CAMERA_RECORD_COOLDOWN_SECONDS,
                    },
                )
            return success

    def _auto_stop(self) -> None:
        with self._lock:
            self._post("/record/stop")
            self._recording = False
            self._stop_timer = None
            logger.info(
                "camera_recording_stopped_auto",
                extra={"gateway": CAMERA_GATEWAY_URL},
            )

    def _post(self, path: str) -> bool:
        try:
            resp = requests.post(
                CAMERA_GATEWAY_URL.rstrip("/") + path,
                timeout=5.0,
            )
            return resp.ok
        except Exception as exc:
            logger.warning(
                "camera_request_failed",
                extra={"path": path, "error": str(exc)},
            )
            return False

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording


# ---------------------------------------------------------------------------
# GSM / SMS notifier
# ---------------------------------------------------------------------------


def _send_gsm_alert(message: str) -> bool:
    """
    POST an SMS alert to the configured GSM gateway.

    The gateway is expected to accept JSON: ``{"to": "<phone>",
    "message": "<text>"}``.  Returns True on success.

    For Raspberry Pi deployments with a GSM HAT this can be replaced with
    a direct AT-command call via ``serial.Serial``.
    """
    if not GSM_GATEWAY_URL or not GSM_ALERT_PHONE:
        return False
    try:
        resp = requests.post(
            GSM_GATEWAY_URL,
            json={"to": GSM_ALERT_PHONE, "message": message},
            timeout=10.0,
        )
        if resp.ok:
            logger.warning(
                "gsm_alert_sent",
                extra={"to": GSM_ALERT_PHONE},
            )
            return True
        logger.warning(
            "gsm_alert_failed",
            extra={"status": resp.status_code},
        )
    except Exception as exc:
        logger.error("gsm_alert_error", extra={"error": str(exc)})
    return False


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------


_SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def _escalate_severity(current: str) -> str:
    """Bump severity up by one level, capped at critical."""
    idx = _SEVERITY_ORDER.index(current) if current in _SEVERITY_ORDER else 0
    return _SEVERITY_ORDER[min(idx + 1, len(_SEVERITY_ORDER) - 1)]


class PhysicalFusionEngine:
    """
    Combines network detection results with physical sensor data.

    One instance is created at API startup when
    ``PHYSICAL_FUSION_ENABLED=true``.  Call :meth:`fuse` for every
    :class:`~app.schemas.DetectionResult` that comes off the network
    detection pipeline.

    Thread safety: the PIR adapter uses its own lock; the fusion engine
    itself is safe to call from multiple threads.
    """

    def __init__(self) -> None:
        self.pir = PIRAdapter()
        self.camera = CameraAdapter()

        if PHYSICAL_FUSION_ENABLED:
            self.pir.start_polling()
            logger.info(
                "physical_fusion_engine_ready",
                extra={
                    "pir_url": PIR_GATEWAY_URL,
                    "camera_url": CAMERA_GATEWAY_URL,
                    "gsm_phone": GSM_ALERT_PHONE or "disabled",
                },
            )

    def fuse(self, result: DetectionResult) -> FusedResult:
        """
        Enrich ``result`` with physical sensor context.

        Logic:
        1. If the network label is in ``PHYSICAL_FUSION_TRIGGER_LABELS``
           AND a PIR motion event was recorded within the fusion window,
           the severity is escalated by one tier and the flag
           ``physical_motion_corroborated`` is added.
        2. If the (possibly escalated) severity is ``critical`` or the
           label is ``malicious``, a camera recording is started.
        3. If severity is ``critical`` AND PIR was triggered, an SMS
           alert is sent.
        """
        fused = FusedResult(base=result)

        if not PHYSICAL_FUSION_ENABLED:
            fused.effective_severity = result.severity
            return fused

        is_trigger = result.predicted_label in PHYSICAL_FUSION_TRIGGER_LABELS
        motion = self.pir.motion_within_window()

        # ── Physical corroboration ──────────────────────────────────────
        if is_trigger and motion:
            fused.physical_motion_detected = True
            fused.extra_flags.append("physical_motion_corroborated")
            fused.effective_severity = _escalate_severity(result.severity)
            logger.warning(
                "physical_cyber_fusion_hit",
                extra={
                    "source_ip": result.source_ip,
                    "original_severity": result.severity,
                    "fused_severity": fused.effective_severity,
                    "label": result.predicted_label,
                },
            )

        # ── Camera activation ───────────────────────────────────────────
        if (
            fused.effective_severity == "critical"
            or result.predicted_label == "malicious"
        ):
            if self.camera.start_recording():
                fused.camera_recording_started = True
                fused.extra_flags.append("camera_recording_started")

        # ── GSM alert ───────────────────────────────────────────────────
        if fused.effective_severity == "critical" and motion:
            msg = (
                f"[IDS CRITICAL] {result.predicted_label.upper()} "
                f"from {result.source_ip} on {result.hostname}. "
                f"Physical motion also detected. Conf={result.confidence:.2f}"
            )
            if _send_gsm_alert(msg):
                fused.gsm_alert_sent = True
                fused.extra_flags.append("gsm_alert_sent")

        return fused
