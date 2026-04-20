"""
detector.py — runtime classification of log events.

The :class:`IDSDetector` is the single object that turns a raw
:class:`~app.schemas.LogEvent` into a fully populated
:class:`~app.schemas.DetectionResult`. It combines three signals:

1. **Allowlist check** — trusted IPs and known-safe keywords short-circuit
   classification to ``benign``.
2. **ML classifier** — a hashing vectorizer + SGD logistic classifier
   trained by :mod:`app.trainer` produces a label and a confidence.
3. **Rule flags** — :func:`app.features.extract_rule_flags` surfaces
   concrete IOCs (failed login, recon tool, SQL-injection-like payload,
   etc.) that are attached to every result.

Results are deduplicated within a short time window so a brute-force attack
doesn't generate a million identical alerts.
"""

import hashlib
import os
import joblib

from app.schemas import LogEvent, DetectionResult
from app.features import event_to_text, extract_rule_flags
from app.llm_explainer import explain_event_with_llm
from app.config import MODEL_PATH, ALLOWLIST_IPS, ALLOWLIST_KEYWORDS, DEDUP_WINDOW_SECONDS


class IDSDetector:
    """
    In-memory classifier that turns log events into detection results.

    One instance is loaded at API startup (see :mod:`app.api`) and reused
    for every request. The underlying model bundle is produced by
    :func:`app.trainer.train_model` and contains a vectorizer, a classifier,
    and the list of class labels.
    """

    def __init__(self) -> None:
        """
        Load the trained model bundle from :data:`app.config.MODEL_PATH`.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist. Callers typically translate
            this into an HTTP 503 ``/health`` failure.
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run 'python train.py' first."
            )

        bundle = joblib.load(MODEL_PATH)
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]
        self.classes = bundle["classes"]

        # SHA-256 of a normalized event string -> last-seen timestamp.
        # Used by :meth:`is_duplicate` for in-memory deduplication.
        self.recent_hashes: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Allowlist
    # ------------------------------------------------------------------ #

    def is_allowlisted(self, event: LogEvent) -> bool:
        """
        Return True if ``event`` should bypass ML classification entirely.

        Allowlists are intentionally simple — an exact IP match or a
        substring match against any allowlist keyword. Anything allowlisted
        is labeled ``benign`` with confidence 0.99 and skips LLM explanation.
        """
        if event.source_ip in ALLOWLIST_IPS:
            return True

        msg = event.message.lower()
        for keyword in ALLOWLIST_KEYWORDS:
            if keyword.lower() in msg:
                return True

        return False

    # ------------------------------------------------------------------ #
    # Deduplication
    # ------------------------------------------------------------------ #

    def is_duplicate(self, event: LogEvent, flags: list[str]) -> bool:
        """
        Return True if an identical event was seen within the dedup window.

        The dedup key includes source IP, hostname, service, message, and
        sorted rule flags. Repeat events inside
        :data:`app.config.DEDUP_WINDOW_SECONDS` are flagged as duplicates
        so downstream code can skip storing or paging on them.

        The method also evicts expired entries from ``self.recent_hashes``
        so it doesn't leak memory.
        """
        raw = f"{event.source_ip}|{event.hostname}|{event.service}|{event.message}|{sorted(flags)}"
        event_hash = hashlib.sha256(raw.encode()).hexdigest()
        now = event.timestamp

        if event_hash in self.recent_hashes:
            if now - self.recent_hashes[event_hash] <= DEDUP_WINDOW_SECONDS:
                return True

        self.recent_hashes[event_hash] = now

        # Prune entries older than the window so memory stays bounded.
        expired = [
            h for h, ts in self.recent_hashes.items()
            if now - ts > DEDUP_WINDOW_SECONDS
        ]
        for h in expired:
            del self.recent_hashes[h]

        return False

    # ------------------------------------------------------------------ #
    # Severity mapping
    # ------------------------------------------------------------------ #

    def severity_from_label(self, label: str, confidence: float, flags: list[str]) -> str:
        """
        Map ``(label, confidence, flags)`` to a severity string.

        ==============  ===================================================
        Label           Severity rule
        ==============  ===================================================
        malicious       ``critical`` if confidence ≥ 0.85, else ``high``.
        suspicious      ``high`` if a dangerous flag is present
                        (privilege_escalation, remote_download); otherwise
                        ``medium``.
        benign / other  ``low``.
        ==============  ===================================================
        """
        if label == "malicious":
            return "critical" if confidence >= 0.85 else "high"

        if label == "suspicious":
            if "privilege_escalation" in flags or "remote_download" in flags:
                return "high"
            if "scan_like_tcp" in flags or "high_risk_port" in flags:
                return "medium"
            return "medium"

        return "low"

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def predict_event(self, event: LogEvent) -> DetectionResult:
        """
        Classify a single :class:`LogEvent` and return a :class:`DetectionResult`.

        The flow is:

        1. Extract rule flags (fast, deterministic).
        2. If allowlisted, short-circuit to ``benign``.
        3. Otherwise vectorize the event text, run it through the classifier,
           pick the highest-probability class, and compute severity.
        4. Ask the LLM explainer for a one-line rationale (or return a
           deterministic fallback if ``IDS_USE_LLM`` is disabled).
        5. Mark the result as deduplicated if we've seen it recently.
        """
        flags = extract_rule_flags(event)

        if self.is_allowlisted(event):
            label = "benign"
            confidence = 0.99
            severity = "low"
            explanation = "Allowlisted activity."
            deduplicated = False
        else:
            text = event_to_text(event)
            X_vec = self.vectorizer.transform([text])

            probabilities = self.classifier.predict_proba(X_vec)[0]
            class_scores = dict(zip(self.classifier.classes_, probabilities))

            label = max(class_scores, key=class_scores.get)
            confidence = float(class_scores[label])

            severity = self.severity_from_label(label, confidence, flags)
            deduplicated = self.is_duplicate(event, flags)
            explanation = explain_event_with_llm(event, label, confidence, flags)

        return DetectionResult(
            timestamp=event.timestamp,
            source_ip=event.source_ip,
            hostname=event.hostname,
            service=event.service,
            message=event.message,
            predicted_label=label,
            confidence=confidence,
            severity=severity,
            rule_flags=flags,
            explanation=explanation,
            deduplicated=deduplicated,
        )
