import sqlite3
import json
from app.config import DB_PATH
from app.schemas import DetectionResult


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            source_ip TEXT,
            hostname TEXT,
            service TEXT,
            message TEXT,
            predicted_label TEXT,
            confidence REAL,
            severity TEXT,
            rule_flags TEXT,
            explanation TEXT
        )
    """)

    conn.commit()
    conn.close()


def store_alert(result: DetectionResult) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO alerts (
            timestamp, source_ip, hostname, service, message,
            predicted_label, confidence, severity, rule_flags, explanation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.timestamp,
        result.source_ip,
        result.hostname,
        result.service,
        result.message,
        result.predicted_label,
        result.confidence,
        result.severity,
        json.dumps(result.rule_flags),
        result.explanation
    ))

    conn.commit()
    conn.close()


def get_alerts(limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, source_ip, hostname, service, message,
               predicted_label, confidence, severity, rule_flags, explanation
        FROM alerts
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "timestamp": row[0],
            "source_ip": row[1],
            "hostname": row[2],
            "service": row[3],
            "message": row[4],
            "predicted_label": row[5],
            "confidence": row[6],
            "severity": row[7],
            "rule_flags": json.loads(row[8]),
            "explanation": row[9]
        })

    return results
