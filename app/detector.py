import hashlib
import os
import joblib

from app.schemas import LogEvent, DetectionResult
from app.features import event_to_text, extract_rule_flags
from app.llm_explainer import explain_event_with_llm
from app.config import MODEL_PATH, ALLOWLIST_IPS, ALLOWLIST_KEYWORDS, DEDUP_WINDOW_SECONDS


class IDSDetector:
    def __init__(self) -> None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run 'python train.py' first."
            )

        bundle = joblib.load(MODEL_PATH)
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]
        self.classes = bundle["classes"]
        self.recent_hashes: dict[str, float] = {}

    def is_allowlisted(self, event: LogEvent) -> bool:
        if event.source_ip in ALLOWLIST_IPS:
            return True

        msg = event.message.lower()
        for keyword in ALLOWLIST_KEYWORDS:
            if keyword.lower() in msg:
                return True

        return False

    def is_duplicate(self, event: LogEvent, flags: list[str]) -> bool:
        raw = f"{event.source_ip}|{event.hostname}|{event.service}|{event.message}|{sorted(flags)}"
        event_hash = hashlib.sha256(raw.encode()).hexdigest()
        now = event.timestamp

        if event_hash in self.recent_hashes:
            if now - self.recent_hashes[event_hash] <= DEDUP_WINDOW_SECONDS:
                return True

        self.recent_hashes[event_hash] = now

        expired = [
            h for h, ts in self.recent_hashes.items()
            if now - ts > DEDUP_WINDOW_SECONDS
        ]
        for h in expired:
            del self.recent_hashes[h]

        return False

    def severity_from_label(self, label: str, confidence: float, flags: list[str]) -> str:
        if label == "malicious":
            return "critical" if confidence >= 0.85 else "high"

        if label == "suspicious":
            if "privilege_escalation" in flags or "remote_download" in flags:
                return "high"
            if "scan_like_tcp" in flags or "high_risk_port" in flags:
                return "medium"
            return "medium"

        return "low"

    def predict_event(self, event: LogEvent) -> DetectionResult:
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
            deduplicated=deduplicated
        )