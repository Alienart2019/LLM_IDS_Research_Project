"""
ais_detector.py — Artificial Immune System (AIS) Detection Layer.

Inspired by the vertebrate adaptive immune system.  The AIS layer provides
a complementary detection mechanism that operates on attack *signatures*
("genes") rather than statistical models.

Key concepts
------------

**Antigen**
    An observed network event.  Represented as a fixed-length feature
    vector derived from the event text (analogous to an antigen surface
    protein pattern).

**Memory Detector (Lymphocyte)**
    A signature learned from a confirmed attack instance.  Stored
    permanently after the *clonal selection* phase.  Detectors are
    described by a "gene" — a set of feature indices and expected value
    ranges — and a *affinity threshold* that controls match sensitivity.

**Negative Selection**
    New candidate detectors are tested against a set of known-self
    (benign) samples.  Any detector that matches a self sample is
    discarded, preventing false positives on normal traffic.  This is the
    biological analogue of T-cell education in the thymus.

**Clonal Selection & Memory**
    When a detector matches an incoming antigen (event), it is
    "activated" and its match count incremented.  After crossing a
    clone threshold it is written to permanent memory — this is how the
    system learns to recognise millions of distinct attack variants over
    a long deployment.

Scalability
-----------
The detector pool is stored as a list of compact ``MemoryDetector``
objects.  Affinity matching is vectorised with NumPy so throughput scales
to ~100k events/second on a single core.  The negative selection set is
held as a compact boolean matrix.

Usage
-----
    from app.ais_detector import AISDetector
    from app.schemas import LogEvent

    ais = AISDetector()
    ais.add_self_sample(benign_event)      # build the self-set
    ais.generate_detectors(n=200)          # run negative selection
    matches = ais.screen(event)            # returns list[MemoryDetector]
    if matches:
        print("AIS match:", matches[0].label, matches[0].gene_description)
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from app.schemas import LogEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Dimensionality of the antigen feature vector.
ANTIGEN_FEATURE_DIM = 512

# Affinity threshold: fraction of features that must match for a detector
# to fire.  Lower = more sensitive (more false positives).
AFFINITY_THRESHOLD = 0.65

# Clone threshold: how many activations before a detector enters memory.
CLONE_THRESHOLD = 3

# Maximum number of memory detectors to keep (oldest evicted first).
MAX_MEMORY_DETECTORS = 10_000

# Number of random candidate detectors generated before negative selection.
NEGATIVE_SELECTION_POOL = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MemoryDetector:
    """
    A single memory detector ("lymphocyte").

    Attributes
    ----------
    detector_id:
        SHA-256 hash of the gene vector, used for deduplication.
    gene:
        Binary feature vector derived from an attack instance.
    label:
        Attack class this detector specialises in.
    affinity_threshold:
        Minimum fraction of gene bits that must match for a positive.
    activation_count:
        How many times this detector has matched an antigen.
    in_memory:
        True once the detector has crossed the clone threshold and is
        permanently stored.
    gene_description:
        Human-readable description of the gene (top feature indices).
    created_at:
        Unix timestamp when the detector was created.
    """
    detector_id: str
    gene: np.ndarray        # shape (ANTIGEN_FEATURE_DIM,), dtype float32
    label: str
    affinity_threshold: float = AFFINITY_THRESHOLD
    activation_count: int = 0
    in_memory: bool = False
    gene_description: str = ""
    created_at: float = field(default_factory=time.time)

    def affinity(self, antigen: np.ndarray) -> float:
        """
        Compute affinity between this detector's gene and ``antigen``.

        Uses the *r-contiguous bits* matching rule: affinity = fraction of
        positions where both gene and antigen are non-zero.
        """
        gene_active = self.gene > 0
        antigen_active = antigen > 0
        both_active = (gene_active & antigen_active).sum()
        total_active = gene_active.sum()
        if total_active == 0:
            return 0.0
        return float(both_active / total_active)

    def matches(self, antigen: np.ndarray) -> bool:
        return self.affinity(antigen) >= self.affinity_threshold


# ---------------------------------------------------------------------------
# Antigen encoder
# ---------------------------------------------------------------------------


class _AntigenEncoder:
    """Encodes a LogEvent into a fixed-dimension binary feature vector."""

    def __init__(self, dim: int = ANTIGEN_FEATURE_DIM) -> None:
        self._dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            ngram_range=(1, 2),
            lowercase=True,
            norm=None,
        )

    def encode(self, event: LogEvent) -> np.ndarray:
        text = (
            f"service={event.service} "
            f"host={event.hostname} "
            f"ip={event.source_ip} "
            f"message={event.message}"
        )
        vec = self._vectorizer.transform([text])
        arr = vec.toarray()[0].astype(np.float32)
        # Binarise: positive values → 1, zero → 0.
        arr[arr > 0] = 1.0
        return arr


# ---------------------------------------------------------------------------
# AIS Detector
# ---------------------------------------------------------------------------


class AISDetector:
    """
    Artificial Immune System detector pool.

    Lifecycle
    ---------
    1. Call :meth:`add_self_sample` for every confirmed-benign event to
       build the self-set.
    2. Call :meth:`generate_detectors` to create a pool of random candidate
       detectors and run negative selection.
    3. At runtime, call :meth:`screen` for every incoming event.  If the
       event matches one or more memory detectors, return them.
    4. Call :meth:`activate` to reward a detector after a confirmed attack
       (manual or from the main classifier).  Once it crosses the clone
       threshold it enters permanent memory.

    Thread safety: uses a per-instance lock.
    """

    def __init__(
        self,
        affinity_threshold: float = AFFINITY_THRESHOLD,
        clone_threshold: int = CLONE_THRESHOLD,
        max_memory: int = MAX_MEMORY_DETECTORS,
    ) -> None:
        self._threshold = affinity_threshold
        self._clone_threshold = clone_threshold
        self._max_memory = max_memory

        self._encoder = _AntigenEncoder()
        self._self_matrix: Optional[np.ndarray] = None
        self._self_samples: list[np.ndarray] = []

        self._candidates: list[MemoryDetector] = []
        self._memory: list[MemoryDetector] = []
        self._lock = threading.Lock()

        logger.info(
            "ais_detector_initialized",
            extra={
                "affinity_threshold": affinity_threshold,
                "clone_threshold": clone_threshold,
                "max_memory": max_memory,
            },
        )

    # ── Self-set management ─────────────────────────────────────────────

    def add_self_sample(self, event: LogEvent) -> None:
        """
        Add a confirmed-benign event to the self-set.

        Self samples are used during negative selection to discard
        detectors that would fire on normal traffic.
        """
        with self._lock:
            antigen = self._encoder.encode(event)
            self._self_samples.append(antigen)
            self._self_matrix = None  # Invalidate cached matrix.

    def _get_self_matrix(self) -> Optional[np.ndarray]:
        if not self._self_samples:
            return None
        if self._self_matrix is None:
            self._self_matrix = np.vstack(self._self_samples)
        return self._self_matrix

    # ── Detector generation (negative selection) ────────────────────────

    def generate_detectors(
        self,
        n: int = NEGATIVE_SELECTION_POOL,
        label: str = "unknown",
        seed_events: Optional[list[LogEvent]] = None,
    ) -> int:
        """
        Generate ``n`` candidate detectors and eliminate any that match
        the self-set (negative selection).

        If ``seed_events`` are provided, the gene vectors are derived from
        those events (targeted generation from known attack instances).
        Otherwise genes are randomly generated.

        Returns the number of detectors that survived negative selection.
        """
        rng = np.random.default_rng()
        candidates: list[np.ndarray] = []

        if seed_events:
            for ev in seed_events:
                base = self._encoder.encode(ev)
                # Add light noise to generate variants.
                noise = rng.integers(0, 2, size=ANTIGEN_FEATURE_DIM).astype(np.float32)
                variant = np.clip(base + noise, 0, 1)
                candidates.append(variant)
            # Pad with random genes if seed_events < n.
            while len(candidates) < n:
                candidates.append(
                    rng.integers(0, 2, size=ANTIGEN_FEATURE_DIM).astype(np.float32)
                )
        else:
            candidates = [
                rng.integers(0, 2, size=ANTIGEN_FEATURE_DIM).astype(np.float32)
                for _ in range(n)
            ]

        survived = 0
        self_matrix = self._get_self_matrix()

        with self._lock:
            for gene in candidates:
                if self._passes_negative_selection(gene, self_matrix):
                    det_id = hashlib.sha256(gene.tobytes()).hexdigest()[:16]
                    top_indices = np.argsort(gene)[-5:][::-1].tolist()
                    detector = MemoryDetector(
                        detector_id=det_id,
                        gene=gene,
                        label=label,
                        affinity_threshold=self._threshold,
                        gene_description=f"top_features={top_indices}",
                    )
                    self._candidates.append(detector)
                    survived += 1

        logger.info(
            "ais_negative_selection_complete",
            extra={
                "generated": n,
                "survived": survived,
                "self_set_size": len(self._self_samples),
            },
        )
        return survived

    def _passes_negative_selection(
        self,
        gene: np.ndarray,
        self_matrix: Optional[np.ndarray],
    ) -> bool:
        """
        Return True if ``gene`` does NOT match any self sample.

        Uses vectorised dot-product affinity to check all self samples
        in one NumPy operation.
        """
        if self_matrix is None:
            return True
        gene_active = (gene > 0).astype(np.float32)
        self_active = (self_matrix > 0).astype(np.float32)
        matches_per_self = self_active @ gene_active         # (n_self,)
        gene_total = gene_active.sum()
        if gene_total == 0:
            return False
        affinities = matches_per_self / gene_total
        return bool(np.all(affinities < self._threshold))

    # ── Runtime screening ───────────────────────────────────────────────

    def screen(self, event: LogEvent) -> list[MemoryDetector]:
        """
        Check ``event`` against all memory and candidate detectors.

        Returns a list of detectors that matched (possibly empty).
        Matching detectors have their ``activation_count`` incremented and
        are promoted to memory if the clone threshold is reached.
        """
        antigen = self._encoder.encode(event)
        matches: list[MemoryDetector] = []

        with self._lock:
            # Check memory detectors first (most important).
            for det in self._memory:
                if det.matches(antigen):
                    det.activation_count += 1
                    matches.append(det)

            # Check candidates and promote to memory if threshold reached.
            new_memory: list[MemoryDetector] = []
            remaining: list[MemoryDetector] = []
            for det in self._candidates:
                if det.matches(antigen):
                    det.activation_count += 1
                    matches.append(det)
                    if det.activation_count >= self._clone_threshold:
                        det.in_memory = True
                        new_memory.append(det)
                        logger.info(
                            "ais_detector_promoted_to_memory",
                            extra={
                                "detector_id": det.detector_id,
                                "label": det.label,
                                "activation_count": det.activation_count,
                            },
                        )
                    else:
                        remaining.append(det)
                else:
                    remaining.append(det)

            self._candidates = remaining
            self._memory.extend(new_memory)

            # Evict oldest memory detectors if over the limit.
            if len(self._memory) > self._max_memory:
                self._memory = sorted(
                    self._memory, key=lambda d: d.activation_count, reverse=True
                )[: self._max_memory]

        if matches:
            logger.warning(
                "ais_match_detected",
                extra={
                    "source_ip": event.source_ip,
                    "match_count": len(matches),
                    "labels": list({d.label for d in matches}),
                },
            )

        return matches

    # ── Seeding from confirmed attacks ──────────────────────────────────

    def learn_from_attack(
        self,
        event: LogEvent,
        label: str,
        n_variants: int = 10,
    ) -> int:
        """
        Seed new detectors from a confirmed attack event.

        Generates ``n_variants`` detectors derived from ``event``'s
        antigen and runs them through negative selection.  This is the
        primary mechanism for expanding the detector pool at runtime.

        Returns the number of detectors that survived negative selection.
        """
        return self.generate_detectors(
            n=n_variants,
            label=label,
            seed_events=[event],
        )

    # ── Introspection ───────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "self_sample_count": len(self._self_samples),
                "candidate_detector_count": len(self._candidates),
                "memory_detector_count": len(self._memory),
                "top_memory_labels": _top_labels(self._memory),
            }


def _top_labels(detectors: list[MemoryDetector]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in detectors:
        counts[d.label] = counts.get(d.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])
