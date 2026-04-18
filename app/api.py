"""
api.py — production FastAPI application.

Endpoints:
  GET  /             health check
  GET  /health       detailed health (model loaded, db reachable)
  POST /events       classify one or more LogEvents in real time
  GET  /alerts       paginated alert list with label/severity filters
  GET  /stats        aggregate counts
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import API_KEY, API_RATE_LIMIT
from app.detector import IDSDetector
from app.honeypot import maybe_forward_to_honeypot
from app.logging_config import setup_logging
from app.schemas import LogEvent
from app.storage import get_alerts, get_stats, init_db, store_alert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (in-process, per IP — replace with Redis for multi-worker)
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window = 60.0
    bucket = _rate_buckets[client_ip]
    # Prune old timestamps
    _rate_buckets[client_ip] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[client_ip]) >= API_RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again in a moment.",
        )
    _rate_buckets[client_ip].append(now)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _verify_api_key(request: Request) -> None:
    if not API_KEY:
        return  # auth disabled (dev mode)
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Lifespan: initialise DB and load model once
# ---------------------------------------------------------------------------

_detector: IDSDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    setup_logging()
    init_db()
    global _detector
    try:
        _detector = IDSDetector()
        logger.info("ids_model_loaded")
    except FileNotFoundError as exc:
        logger.error("ids_model_missing", extra={"error": str(exc)})
        # Allow API to start so /health returns a meaningful error
    yield
    logger.info("ids_api_shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Trainable IDS API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Common dependencies
# ---------------------------------------------------------------------------

def common_deps(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    _verify_api_key(request)


AuthDep = Annotated[None, Depends(common_deps)]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EventRequest(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    source_ip: str = Field(..., examples=["10.0.0.5"])
    hostname: str = Field(..., examples=["web-01"])
    service: str = Field(..., examples=["ssh"])
    message: str = Field(..., examples=["Failed password for root from 10.0.0.5"])


class EventResponse(BaseModel):
    timestamp: float
    source_ip: str
    hostname: str
    service: str
    message: str
    predicted_label: str
    confidence: float
    severity: str
    rule_flags: list[str]
    explanation: str
    deduplicated: bool
    honeypot_forwarded: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {"service": "Trainable IDS API", "version": "2.0.0", "status": "ok"}


@app.get("/health", tags=["meta"])
def health():
    model_ok = _detector is not None
    try:
        # Lightweight DB probe
        get_alerts(limit=1)
        db_ok = True
    except Exception:
        db_ok = False

    code = status.HTTP_200_OK if (model_ok and db_ok) else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=code,
        content={
            "model_loaded": model_ok,
            "db_reachable": db_ok,
            "honeypot_enabled": bool(__import__("app.config", fromlist=["HONEYPOT_ENABLED"]).HONEYPOT_ENABLED),
        },
    )


@app.post("/events", response_model=list[EventResponse], tags=["detection"])
def classify_events(
    _: AuthDep,
    events: list[EventRequest],
) -> list[EventResponse]:
    """
    Classify one or more log events. Returns a DetectionResult per event.
    Malicious events are forwarded to the honeypot (if enabled) and stored.
    """
    if _detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run train.py first.",
        )

    results: list[EventResponse] = []

    for ev in events:
        log_event = LogEvent(
            timestamp=ev.timestamp,
            source_ip=ev.source_ip,
            hostname=ev.hostname,
            service=ev.service,
            message=ev.message,
        )

        detection = _detector.predict_event(log_event)
        forwarded = maybe_forward_to_honeypot(detection)

        if detection.predicted_label != "benign" and not detection.deduplicated:
            store_alert(detection)

        results.append(
            EventResponse(
                timestamp=detection.timestamp,
                source_ip=detection.source_ip,
                hostname=detection.hostname,
                service=detection.service,
                message=detection.message,
                predicted_label=detection.predicted_label,
                confidence=detection.confidence,
                severity=detection.severity,
                rule_flags=detection.rule_flags,
                explanation=detection.explanation,
                deduplicated=detection.deduplicated,
                honeypot_forwarded=forwarded,
            )
        )

    return results


@app.get("/alerts", tags=["alerts"])
def list_alerts(
    _: AuthDep,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    label: str | None = Query(default=None, description="Filter by predicted_label"),
    severity: str | None = Query(default=None, description="Filter by severity"),
):
    """Return stored alerts with optional filtering and pagination."""
    return get_alerts(limit=limit, offset=offset, label=label, severity=severity)


@app.get("/stats", tags=["alerts"])
def stats(_: AuthDep):
    """Aggregate counts by label and severity."""
    return get_stats()
