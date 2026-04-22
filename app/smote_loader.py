"""
smote_loader.py — SMOTE-based minority class oversampling for IDS training.

The Kitsune and similar IoT datasets are heavily imbalanced: benign traffic
can outnumber minority attack classes (e.g. ARP MitM, Fuzzing) by 10:1 or
more.  Research shows SMOTE (Synthetic Minority Over-sampling Technique)
can improve minority-class recall by ~22 % compared to unbalanced training.

This module wraps the imbalanced-learn SMOTE implementation and exposes a
clean ``apply_smote`` helper that works with the sparse matrices produced by
:class:`sklearn.feature_extraction.text.HashingVectorizer`.

Usage
-----
    from app.smote_loader import apply_smote
    X_resampled, y_resampled = apply_smote(X_sparse, y_array)

Requirements
------------
    pip install imbalanced-learn

If imbalanced-learn is not installed the function logs a warning and returns
the original data unchanged so the training pipeline never hard-crashes.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import issparse, csr_matrix

logger = logging.getLogger(__name__)

# Only apply SMOTE when the minority class has at least this many samples.
# Below this threshold the synthetic sample quality degrades.
MIN_SAMPLES_FOR_SMOTE = 10

# Target ratio: after SMOTE the minority class will have at least this
# fraction of the majority class count.  0.5 = 50 % parity.
SMOTE_TARGET_RATIO = 0.5


def apply_smote(
    X: "np.ndarray | csr_matrix",
    y: "np.ndarray | list[str]",
    *,
    random_state: int = 42,
    target_ratio: float = SMOTE_TARGET_RATIO,
) -> tuple["np.ndarray | csr_matrix", "np.ndarray"]:
    """
    Apply SMOTE oversampling to balance class distribution.

    Parameters
    ----------
    X:
        Feature matrix. May be a dense ndarray or a scipy sparse matrix.
    y:
        1-D label array or list.
    random_state:
        Reproducibility seed.
    target_ratio:
        Desired ratio of minority to majority class after resampling.

    Returns
    -------
    (X_resampled, y_resampled)
        Resampled feature matrix and label array.  If SMOTE cannot be
        applied (insufficient samples, missing library, etc.) the original
        data is returned unchanged.
    """
    y_arr = np.asarray(y)

    # Check that we have at least two classes.
    unique, counts = np.unique(y_arr, return_counts=True)
    if len(unique) < 2:
        logger.debug("smote_skipped_single_class")
        return X, y_arr

    min_count = int(counts.min())
    if min_count < MIN_SAMPLES_FOR_SMOTE:
        logger.warning(
            "smote_skipped_too_few_minority",
            extra={"min_class_count": min_count, "threshold": MIN_SAMPLES_FOR_SMOTE},
        )
        return X, y_arr

    try:
        from imblearn.over_sampling import SMOTE  # type: ignore[import]
    except ImportError:
        logger.warning(
            "smote_skipped_imbalanced_learn_missing",
            extra={"hint": "pip install imbalanced-learn"},
        )
        return X, y_arr

    # SMOTE works with dense or sparse matrices.
    X_in = X.toarray() if issparse(X) else X

    # Build the sampling_strategy dict: bring every minority class up to
    # target_ratio × majority_class_count, without touching the majority.
    majority_count = int(counts.max())
    target_count = max(min_count, int(majority_count * target_ratio))

    sampling_strategy: dict[str, int] = {}
    for cls, cnt in zip(unique, counts):
        if cnt < majority_count:
            sampling_strategy[cls] = max(int(cnt), target_count)

    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=min(5, min_count - 1),
        )
        X_res, y_res = smote.fit_resample(X_in, y_arr)

        logger.info(
            "smote_applied",
            extra={
                "original_shape": X_in.shape,
                "resampled_shape": X_res.shape,
                "class_distribution_before": dict(zip(unique.tolist(), counts.tolist())),
                "class_distribution_after": dict(
                    zip(*[arr.tolist() for arr in np.unique(y_res, return_counts=True)])
                ),
            },
        )

        # Return as sparse to keep memory usage consistent with the rest of the pipeline.
        return csr_matrix(X_res), y_res

    except Exception as exc:
        logger.error("smote_failed", extra={"error": str(exc)})
        return X, y_arr


def log_class_distribution(y: "np.ndarray | list[str]", tag: str = "") -> None:
    """Log the class distribution of ``y`` for diagnostic purposes."""
    y_arr = np.asarray(y)
    unique, counts = np.unique(y_arr, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(
        "class_distribution",
        extra={"tag": tag, "distribution": dist, "total": int(len(y_arr))},
    )
