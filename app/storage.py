"""
storage.py — SQLite persistence for IDS alerts.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.config import DB_PATH
from app.schemas import DetectionResult


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            source_ip       TEXT    NOT NULL,
            hostname        TEXT    NOT NULL,
            service         TEXT    NOT NULL,
            message         TEXT    NOT NULL,
            predicted_label TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            severity        TEXT    NOT NULL,
            rule_flags      TEXT    NOT NULL,
            explanation     TEXT    NOT NULL,
            honeypot_sent   INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label    ON alerts(predicted_label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON alerts(severity)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts       ON alerts(timestamp)")
    conn.commit()
    conn.close()


def store_alert(result: DetectionResult, honeypot_sent: bool = False) -> int:
    conn = _connect()
    cur = conn.execute(
        """
        INSERT INTO alerts (
            timestamp, source_ip, hostname, service, message,
            predicted_label, confidence, severity, rule_flags,
            explanation, honeypot_sent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.timestamp,
            result.source_ip,
            result.hostname,
            result.service,
            result.message,
            result.predicted_label,
            result.confidence,
            result.severity,
            json.dumps(result.rule_flags),
            result.explanation,
            int(honeypot_sent),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id  # type: ignore[return-value]


def get_alerts(
    limit: int = 50,
    offset: int = 0,
    label: str | None = None,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    conn = _connect()
    conditions: list[str] = []
    params: list[Any] = []

    if label:
        conditions.append("predicted_label = ?")
        params.append(label)
    if severity:
        conditions.append("severity = ?")
        params.append(severity)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])

    rows = conn.execute(
        f"""
        SELECT timestamp, source_ip, hostname, service, message,
               predicted_label, confidence, severity, rule_flags,
               explanation, honeypot_sent
        FROM alerts
        {where}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()
    conn.close()

    return [
        {
            "timestamp": r["timestamp"],
            "source_ip": r["source_ip"],
            "hostname": r["hostname"],
            "service": r["service"],
            "message": r["message"],
            "predicted_label": r["predicted_label"],
            "confidence": r["confidence"],
            "severity": r["severity"],
            "rule_flags": json.loads(r["rule_flags"]),
            "explanation": r["explanation"],
            "honeypot_sent": bool(r["honeypot_sent"]),
        }
        for r in rows
    ]


def get_stats() -> dict[str, Any]:
    conn = _connect()

    label_rows = conn.execute(
        "SELECT predicted_label, COUNT(*) as n FROM alerts GROUP BY predicted_label"
    ).fetchall()

    severity_rows = conn.execute(
        "SELECT severity, COUNT(*) as n FROM alerts GROUP BY severity"
    ).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    honeypot_total = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE honeypot_sent = 1"
    ).fetchone()[0]

    conn.close()

    return {
        "total_alerts": total,
        "honeypot_forwarded": honeypot_total,
        "by_label": {r["predicted_label"]: r["n"] for r in label_rows},
        "by_severity": {r["severity"]: r["n"] for r in severity_rows},
    }
