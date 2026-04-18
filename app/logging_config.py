"""
logging_config.py — structured logging for the IDS.

Call setup_logging() once at startup (main.py / api.py lifespan).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.config import LOG_LEVEL, LOG_JSON


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        base: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra= fields passed at the call site
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }:
                base[key] = val

        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        return json.dumps(base, default=str)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    if LOG_JSON:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "scapy.runtime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
