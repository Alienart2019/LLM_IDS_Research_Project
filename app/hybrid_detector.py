"""
hybrid_detector.py — 2-stage hybrid detection pipeline.

Stage 1: K-means clustering identifies "safe zones" (benign clusters).
         Events falling inside a safe zone are fast-pathed as benign.
Stage 2: A Decision Tree classifier filters remaining events, reducing
         false positives compared to a single-model approach.

Research basis: K-means + Decision Tree hybrids consistently outperform
single-model approaches in IoT IDS literature, achieving lower FPR while
maintaining high recall on attack traffic.

Usage
-----
    from app.hybrid_detector import HybridDetector
    hd = HybridDetector()
    hd.fit(X_text_list, y_label_list)
    label, confidence, stage = hd.predict_one("service=ssh message=...")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.tree import DecisionTreeClassifier

from app.config import MODEL_PATH

logger = logging.getLogger(__name__)

# Path for the hybrid bundle alongside the main model.
HYBRID_MODEL_PATH = os.path.join(os.path.dirname(MODEL_PATH), "ids_hybrid_model.pkl")

# Fraction of a cluster that must be benign for it to be a "safe zone".
SAFE_ZONE_PURITY_THRESHOLD = 0.90

# If Stage 1 confidence is below this, escalate to Stage 2.
STAGE1_CONFIDENCE_THRESHOLD = 0.80


@dataclass
class HybridPrediction:
    label: str
    confidence: float
    stage: int          # 1 = safe-zone hit, 2 = DT classifier
    cluster_id: int     # K-means cluster the event landed in


class HybridDetector:
    """
    2-stage K-means + Decision Tree hybrid IDS model.

    Training
    --------
    Call :meth:`fit` with a list of pre-vectorized text strings and
    corresponding label strings.

    Inference
    ---------
    Call :meth:`predict_one` with a flat event text string.  Returns a
    :class:`HybridPrediction` that includes which stage made the call.

    Persistence
    -----------
    Call :meth:`save` / :meth:`load` to persist the bundle to disk.
    """

    def __init__(self, n_clusters: int = 20) -> None:
        self.n_clusters = n_clusters
        self.vectorizer = HashingVectorizer(
            n_features=2**18,
            alternate_sign=False,
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=4096,
            n_init=3,
        )
        self.dt = DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
        )
        self.safe_zone_clusters: set[int] = set()
        self.classes = ["benign", "suspicious", "malicious"]
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, texts: list[str], labels: list[str]) -> "HybridDetector":
        """
        Train both stages on the provided text/label pairs.

        Parameters
        ----------
        texts:   Flat event strings (output of ``event_to_text``).
        labels:  Corresponding label strings (benign/suspicious/malicious).
        """
        logger.info("hybrid_fit_start", extra={"n_samples": len(texts)})

        X = self.vectorizer.transform(texts)

        # --- Stage 1: K-means clustering --------------------------------
        self.kmeans.fit(X)
        cluster_ids = self.kmeans.predict(X)

        # Determine which clusters are "safe zones".
        from collections import Counter
        cluster_label_counts: dict[int, Counter] = {
            i: Counter() for i in range(self.n_clusters)
        }
        for cid, lbl in zip(cluster_ids, labels):
            cluster_label_counts[cid][lbl] += 1

        self.safe_zone_clusters = set()
        for cid, counts in cluster_label_counts.items():
            total = sum(counts.values())
            if total == 0:
                continue
            benign_frac = counts.get("benign", 0) / total
            if benign_frac >= SAFE_ZONE_PURITY_THRESHOLD:
                self.safe_zone_clusters.add(cid)

        logger.info(
            "hybrid_safe_zones_identified",
            extra={
                "safe_zones": len(self.safe_zone_clusters),
                "total_clusters": self.n_clusters,
            },
        )

        # --- Stage 2: Decision Tree on non-safe-zone events --------------
        # Only train the DT on events that are NOT in safe zones, since
        # safe-zone events are already handled by Stage 1.
        mask = np.array([cid not in self.safe_zone_clusters for cid in cluster_ids])
        if mask.sum() > 0:
            X_stage2 = X[mask]
            y_stage2 = [labels[i] for i in range(len(labels)) if mask[i]]
            self.dt.fit(X_stage2, y_stage2)
            logger.info(
                "hybrid_dt_fit",
                extra={"stage2_samples": int(mask.sum())},
            )
        else:
            logger.warning("hybrid_dt_no_samples_all_safe_zone")

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_one(self, text: str) -> HybridPrediction:
        """Classify a single event text string."""
        if not self._fitted:
            raise RuntimeError("HybridDetector must be fitted before predicting.")

        X = self.vectorizer.transform([text])
        cluster_id = int(self.kmeans.predict(X)[0])

        # Stage 1: safe zone fast-path
        if cluster_id in self.safe_zone_clusters:
            # Estimate confidence as centroid distance — closer = more confident.
            centroid = self.kmeans.cluster_centers_[cluster_id]
            vec_dense = X.toarray()[0]
            dist = float(np.linalg.norm(vec_dense - centroid))
            # Map distance to [0.80, 0.99] range (closer → higher confidence).
            confidence = max(0.80, min(0.99, 0.99 - dist * 0.02))
            return HybridPrediction(
                label="benign",
                confidence=confidence,
                stage=1,
                cluster_id=cluster_id,
            )

        # Stage 2: Decision Tree
        try:
            probas = self.dt.predict_proba(X)[0]
            classes = list(self.dt.classes_)
            label = classes[int(np.argmax(probas))]
            confidence = float(np.max(probas))
        except Exception as exc:
            logger.warning("hybrid_dt_predict_failed", extra={"error": str(exc)})
            label = "suspicious"
            confidence = 0.50

        return HybridPrediction(
            label=label,
            confidence=confidence,
            stage=2,
            cluster_id=cluster_id,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str = HYBRID_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bundle = {
            "vectorizer": self.vectorizer,
            "kmeans": self.kmeans,
            "dt": self.dt,
            "safe_zone_clusters": self.safe_zone_clusters,
            "n_clusters": self.n_clusters,
            "classes": self.classes,
        }
        joblib.dump(bundle, path, compress=3)
        logger.info("hybrid_model_saved", extra={"path": path})

    @classmethod
    def load(cls, path: str = HYBRID_MODEL_PATH) -> "HybridDetector":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Hybrid model not found at {path}. "
                "Run train_hybrid.py first."
            )
        bundle = joblib.load(path)
        detector = cls(n_clusters=bundle["n_clusters"])
        detector.vectorizer = bundle["vectorizer"]
        detector.kmeans = bundle["kmeans"]
        detector.dt = bundle["dt"]
        detector.safe_zone_clusters = bundle["safe_zone_clusters"]
        detector.classes = bundle["classes"]
        detector._fitted = True
        logger.info(
            "hybrid_model_loaded",
            extra={
                "path": path,
                "safe_zones": len(detector.safe_zone_clusters),
            },
        )
        return detector
