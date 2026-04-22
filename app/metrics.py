"""
metrics.py — Extended IDS performance metrics.

Goes beyond accuracy to report metrics that matter in IoT IDS deployments:

* Standard classification metrics (precision, recall, F1, confusion matrix).
* False Positive Rate (FPR) and False Negative Rate (FNR) per class.
* Interaction Ability Score — how many TCP/IP layers the IDS observes.
* Inference latency (wall-clock ms per event).
* Memory overhead of the loaded model bundle.

Usage
-----
    from app.metrics import evaluate_model, InferenceTimer

    report = evaluate_model(y_true, y_pred, latencies_ms=[1.2, 0.8, 1.1])
    print(report.summary())

    # Time individual predictions:
    timer = InferenceTimer()
    with timer:
        result = detector.predict_event(event)
    print(f"{timer.last_ms:.2f} ms")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# TCP/IP layer constants for the Interaction Ability Score
# ---------------------------------------------------------------------------

# Which layers each service/protocol touches.  Extend as needed.
_LAYER_MAP: dict[str, set[int]] = {
    # Layer 1 = Network Interface (link layer)
    "ethernet": {1},
    "arp": {1, 2},
    # Layer 2 = Internet (IP)
    "icmp": {2},
    "ip": {2},
    # Layer 3 = Transport
    "tcp": {2, 3},
    "udp": {2, 3},
    # Layer 4 = Application
    "http": {2, 3, 4},
    "https": {2, 3, 4},
    "ssh": {2, 3, 4},
    "ftp": {2, 3, 4},
    "dns": {2, 3, 4},
    "smtp": {2, 3, 4},
    "web": {2, 3, 4},
    "log": {4},
    "syslog": {4},
}

_MAX_LAYERS = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int
    fpr: float          # False Positive Rate
    fnr: float          # False Negative Rate


@dataclass
class ModelReport:
    """
    Full evaluation report for an IDS model.

    All latency values are in milliseconds.
    """
    classes: list[str]
    per_class: list[ClassMetrics]
    macro_f1: float
    weighted_f1: float
    accuracy: float
    confusion: list[list[int]]

    # Latency
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Interaction ability
    interaction_ability_score: float = 0.0
    layers_observed: list[int] = field(default_factory=list)

    # Memory
    model_memory_mb: float = 0.0

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            "=" * 60,
            "IDS Model Evaluation Report",
            "=" * 60,
            f"Accuracy:      {self.accuracy:.4f}",
            f"Macro F1:      {self.macro_f1:.4f}",
            f"Weighted F1:   {self.weighted_f1:.4f}",
            "",
            "Per-Class Metrics:",
        ]
        for cm in self.per_class:
            lines.append(
                f"  {cm.label:<14} P={cm.precision:.3f}  R={cm.recall:.3f}  "
                f"F1={cm.f1:.3f}  FPR={cm.fpr:.3f}  FNR={cm.fnr:.3f}  "
                f"n={cm.support}"
            )
        lines += [
            "",
            "Latency (ms):",
            f"  Mean={self.mean_latency_ms:.2f}  "
            f"P95={self.p95_latency_ms:.2f}  "
            f"P99={self.p99_latency_ms:.2f}  "
            f"Max={self.max_latency_ms:.2f}",
            "",
            f"Interaction Ability Score: {self.interaction_ability_score:.2f} / 1.00",
            f"  Layers observed: {self.layers_observed}",
            f"Model memory: {self.model_memory_mb:.1f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": [
                {
                    "label": c.label,
                    "precision": c.precision,
                    "recall": c.recall,
                    "f1": c.f1,
                    "fpr": c.fpr,
                    "fnr": c.fnr,
                    "support": c.support,
                }
                for c in self.per_class
            ],
            "confusion_matrix": self.confusion,
            "latency": {
                "mean_ms": self.mean_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "max_ms": self.max_latency_ms,
            },
            "interaction_ability_score": self.interaction_ability_score,
            "layers_observed": self.layers_observed,
            "model_memory_mb": self.model_memory_mb,
        }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_model(
    y_true: list[str],
    y_pred: list[str],
    *,
    latencies_ms: Optional[list[float]] = None,
    services_seen: Optional[list[str]] = None,
    model_path: Optional[str] = None,
) -> ModelReport:
    """
    Compute a full :class:`ModelReport` from ground-truth and predicted labels.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    latencies_ms:
        Optional list of per-prediction wall-clock durations in milliseconds.
    services_seen:
        Optional list of service/protocol strings observed (e.g. ``["ssh",
        "tcp", "http"]``). Used to compute the Interaction Ability Score.
    model_path:
        If provided, the on-disk size of the model file is reported as
        model memory.
    """
    classes = sorted(set(y_true) | set(y_pred))
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Per-class precision / recall / F1.
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=classes, zero_division=0
    )

    # Confusion matrix for FPR / FNR calculation.
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)

    per_class: list[ClassMetrics] = []
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        per_class.append(
            ClassMetrics(
                label=cls,
                precision=float(prec[i]),
                recall=float(rec[i]),
                f1=float(f1[i]),
                support=int(support[i]),
                fpr=fpr,
                fnr=fnr,
            )
        )

    # Macro and weighted F1.
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    accuracy = float((y_true_arr == y_pred_arr).mean())

    # Latency stats.
    mean_lat = p95_lat = p99_lat = max_lat = 0.0
    if latencies_ms:
        lat = np.asarray(latencies_ms)
        mean_lat = float(lat.mean())
        p95_lat = float(np.percentile(lat, 95))
        p99_lat = float(np.percentile(lat, 99))
        max_lat = float(lat.max())

    # Interaction Ability Score.
    layers_seen: set[int] = set()
    if services_seen:
        for svc in services_seen:
            layers_seen.update(_LAYER_MAP.get(svc.lower(), set()))
    ias = len(layers_seen) / _MAX_LAYERS if layers_seen else 0.0

    # Model memory.
    model_memory_mb = 0.0
    if model_path:
        try:
            import os
            model_memory_mb = os.path.getsize(model_path) / (1024 * 1024)
        except Exception:
            pass

    return ModelReport(
        classes=classes,
        per_class=per_class,
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        accuracy=accuracy,
        confusion=cm.tolist(),
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
        p99_latency_ms=p99_lat,
        max_latency_ms=max_lat,
        interaction_ability_score=ias,
        layers_observed=sorted(layers_seen),
        model_memory_mb=model_memory_mb,
    )


# ---------------------------------------------------------------------------
# Inference timer context manager
# ---------------------------------------------------------------------------


class InferenceTimer:
    """
    Context manager that times a single inference call.

    Usage::

        timer = InferenceTimer()
        with timer:
            result = detector.predict_event(event)
        latency_ms = timer.last_ms

    :attr:`history_ms` accumulates all measurements so you can compute
    aggregate stats at the end of a batch.
    """

    def __init__(self) -> None:
        self.last_ms: float = 0.0
        self.history_ms: list[float] = []
        self._start: float = 0.0

    def __enter__(self) -> "InferenceTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.last_ms = (time.perf_counter() - self._start) * 1000.0
        self.history_ms.append(self.last_ms)

    def summary(self) -> dict:
        if not self.history_ms:
            return {}
        arr = np.asarray(self.history_ms)
        return {
            "count": len(arr),
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }
