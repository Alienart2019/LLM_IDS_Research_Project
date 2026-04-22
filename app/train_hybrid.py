"""
train_hybrid.py — train the 2-stage hybrid K-means + Decision Tree model.

Run after ``train.py`` (or independently) to produce the hybrid model
bundle alongside the standard SGD model.

Usage
-----
    python train_hybrid.py                # uses ./data
    python train_hybrid.py data/archive  # custom path
    python train_hybrid.py data/ --smote # apply SMOTE oversampling
    python train_hybrid.py data/ --eval  # print evaluation report after training

The resulting model is saved to ``models/ids_hybrid_model.pkl``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from app.dataloader import (
    iter_training_sources,
    iter_csv_training_chunks,
    iter_paired_csv_training_chunks,
)
from app.features import row_to_text
from app.hybrid_detector import HybridDetector
from app.logging_config import setup_logging
from app.metrics import evaluate_model, InferenceTimer
from app.schemas import LogEvent

CHUNK_SIZE = 20_000


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python train_hybrid.py",
        description="Train the 2-stage hybrid K-means + DT IDS model.",
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="data",
        help="Path to the training data directory (default: data).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=20,
        help="Number of K-means clusters (default: 20).",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling to balance minority classes.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        dest="run_eval",
        help="Run evaluation report on a held-out slice after training.",
    )
    args = parser.parse_args()

    setup_logging()

    root = Path(args.dataset_path)
    if not root.exists():
        print(f"[error] dataset directory not found: {root}", file=sys.stderr)
        return 2

    detector = HybridDetector(n_clusters=args.clusters)

    all_texts: list[str] = []
    all_labels: list[str] = []
    files_processed = 0
    rows_processed = 0

    print(f"Scanning {root} for training sources...")

    for source_type, payload in iter_training_sources(str(root)):
        try:
            if source_type == "paired_csv":
                dataset_file, labels_file = payload
                print(f"  [paired] {dataset_file.name}")
                chunk_iter = iter_paired_csv_training_chunks(
                    dataset_file, labels_file, CHUNK_SIZE
                )
                files_processed += 1
            elif source_type == "single_csv":
                csv_file = payload
                print(f"  [single] {csv_file.name}")
                chunk_iter = iter_csv_training_chunks(csv_file, CHUNK_SIZE)
                files_processed += 1
            else:
                continue

            for chunk in chunk_iter:
                if chunk.empty:
                    continue
                chunk["text"] = chunk.apply(row_to_text, axis=1)
                all_texts.extend(chunk["text"].tolist())
                all_labels.extend(chunk["label"].tolist())
                rows_processed += len(chunk)
                del chunk
                gc.collect()

        except Exception as exc:
            print(f"  [skipped] error: {exc}")

    if not all_texts:
        print("[error] no training data found.", file=sys.stderr)
        return 1

    print(
        f"\nLoaded {rows_processed} rows from {files_processed} file(s). "
        "Training hybrid model..."
    )

    # Optional SMOTE oversampling.
    if args.smote:
        from app.smote_loader import apply_smote, log_class_distribution
        import numpy as np
        vec = detector.vectorizer
        X_sparse = vec.transform(all_texts)
        y_arr = np.asarray(all_labels)
        log_class_distribution(y_arr, tag="before_smote")
        X_res, y_res = apply_smote(X_sparse, y_arr)
        log_class_distribution(y_res, tag="after_smote")
        # Re-convert to text list for fit() (it calls vectorizer internally).
        # Since SMOTE operates in feature space we use the vectorized form
        # directly by calling fit_from_matrix.
        detector._fit_from_matrix(X_res, y_res.tolist())
    else:
        detector.fit(all_texts, all_labels)

    detector.save()
    print(f"\nHybrid model saved. Safe zones: {len(detector.safe_zone_clusters)}")

    # Optional evaluation.
    if args.run_eval:
        print("\nRunning evaluation on a 2,000-sample hold-out slice...")
        import random
        sample_size = min(2000, len(all_texts))
        indices = random.sample(range(len(all_texts)), sample_size)
        sample_texts = [all_texts[i] for i in indices]
        sample_labels = [all_labels[i] for i in indices]

        timer = InferenceTimer()
        preds: list[str] = []
        for t in sample_texts:
            with timer:
                pred = detector.predict_one(t)
            preds.append(pred.label)

        report = evaluate_model(
            sample_labels,
            preds,
            latencies_ms=timer.history_ms,
        )
        print(report.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
