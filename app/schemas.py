from dataclasses import dataclass
from typing import List


@dataclass
class LogEvent:
    timestamp: float
    source_ip: str
    hostname: str
    service: str
    message: str


@dataclass
class DetectionResult:
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