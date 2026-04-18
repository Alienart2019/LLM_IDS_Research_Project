"""
honeypot.py — route malicious events to a honeypot endpoint.

When HONEYPOT_ENABLED is true, any DetectionResult whose predicted_label is
in HONEYPOT_TRIGGER_LABELS is forwarded (via HTTP POST) to HONEYPOT_URL.
The forward is fire-and-forget so it never blocks the main detection path.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict

import requests

from app.config import (
    HONEYPOT_ENABLED,
    HONEYPOT_URL,
    HONEYPOT_TRIGGER_LABELS,
    HONEYPOT_FORWARD_TIMEOUT,
)
from app.schemas import DetectionResult

logger = logging.getLogger(__name__)


def _forward(payload: dict) -> None:
    """Background thread: POST the event to the honeypot endpoint."""
    try:
        resp = requests.post(
            HONEYPOT_URL,
            json=payload,
            timeout=HONEYPOT_FORWARD_TIMEOUT,
        )
        logger.info(
            "honeypot_forward",
            extra={
                "honeypot_url": HONEYPOT_URL,
                "status_code": resp.status_code,
                "source_ip": payload.get("source_ip"),
            },
        )
    except requests.exceptions.Timeout:
        logger.warning("honeypot_forward_timeout", extra={"honeypot_url": HONEYPOT_URL})
    except Exception as exc:  # noqa: BLE001
        logger.error("honeypot_forward_error", extra={"error": str(exc)})


def maybe_forward_to_honeypot(result: DetectionResult) -> bool:
    """
    Forward result to honeypot if conditions are met.

    Returns True when the event was forwarded.
    """
    if not HONEYPOT_ENABLED:
        return False

    if result.predicted_label not in HONEYPOT_TRIGGER_LABELS:
        return False

    payload = asdict(result)
    # Fire and forget — use a daemon thread so it doesn't block shutdown
    t = threading.Thread(target=_forward, args=(payload,), daemon=True)
    t.start()

    logger.warning(
        "honeypot_triggered",
        extra={
            "source_ip": result.source_ip,
            "hostname": result.hostname,
            "label": result.predicted_label,
            "severity": result.severity,
            "confidence": round(result.confidence, 3),
        },
    )
    return True
