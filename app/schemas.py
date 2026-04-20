"""
schemas.py — shared data containers passed between layers of the IDS.

Both :class:`LogEvent` and :class:`DetectionResult` are plain dataclasses.
They're intentionally dumb — no validation, no business logic — so they can
cross freely between the CLI, the API, the detector, and the storage layer.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LogEvent:
    """
    A single security-relevant event observed somewhere in the environment.

    Attributes
    ----------
    timestamp:
        Unix epoch seconds (float). For PCAP-derived events this is the
        packet's capture time; for live API events it's ``time.time()`` at
        ingestion.
    source_ip:
        Source IP address as a string. ``"unknown"`` is the convention when
        the address can't be determined (e.g. text-only log lines).
    hostname:
        Host or sensor that produced the event.
    service:
        Service or protocol (``ssh``, ``web``, ``tcp``, ``udp``, ``log`` …).
    message:
        Free-form event text. Rule extraction and the vectorizer both read
        from this field.
    """

    timestamp: float
    source_ip: str
    hostname: str
    service: str
    message: str


@dataclass
class DetectionResult:
    """
    The output of :meth:`app.detector.IDSDetector.predict_event`.

    Contains every field from the original :class:`LogEvent` plus the
    model's verdict, rule flags, severity, and an LLM (or fallback)
    explanation. The API layer serializes this directly into its response
    body, and the storage layer persists it to SQLite.

    Attributes
    ----------
    predicted_label:
        One of ``benign``, ``suspicious``, ``malicious``.
    confidence:
        Classifier probability for ``predicted_label``, in ``[0.0, 1.0]``.
    severity:
        Derived from label + confidence + flags; one of ``low``, ``medium``,
        ``high``, ``critical``.
    rule_flags:
        Names of every rule that fired on the event (see
        :func:`app.features.extract_rule_flags`).
    explanation:
        Short human-readable rationale. LLM-generated when
        ``IDS_USE_LLM=true``, otherwise a deterministic template.
    deduplicated:
        True if an identical event was already seen within the dedup
        window. Callers typically skip storing / alerting on duplicates.
    """

    timestamp: float
    source_ip: str
    hostname: str
    service: str
    message: str
    predicted_label: str
    confidence: float
    severity: str
    rule_flags: List[str]
    explanation: str
    deduplicated: bool = False
