"""
trainer.py — incremental training pipeline.

Uses scikit-learn's :class:`HashingVectorizer` + :class:`SGDClassifier` so
the model can be trained in a streaming, memory-bounded way. That matters
here because Kitsune-style datasets routinely contain hundreds of thousands
to millions of rows and won't fit in RAM as a single matrix.

Flow
----
::

    iter_training_sources(data/)  ->  paired_csv | single_csv
         │
         ▼
    iter_*_training_chunks(...)   ->  normalized chunk DataFrames
         │
         ▼
    HashingVectorizer.transform   ->  sparse matrix
         │
         ▼
    SGDClassifier.partial_fit     ->  incremental model update
         │
         ▼
    joblib.dump                   ->  models/ids_model.pkl
"""

import os
import gc

import joblib
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.config import MODEL_PATH
from app.dataloader import (
    iter_training_sources,
    iter_csv_training_chunks,
    iter_paired_csv_training_chunks,
)
from app.features import row_to_text


# Rows per streaming chunk. 20k keeps memory well under 1 GB even on
# Kitsune's wide feature tables.
CHUNK_SIZE = 20000

# Fixed ordered class list. ``SGDClassifier.partial_fit`` requires the
# classes to be declared up front on the first call so it can size its
# coefficient matrix, even if a given chunk doesn't contain all three.
CLASSES = ["benign", "suspicious", "malicious"]


def train_model(dataset_path: str) -> None:
    """
    Train an incremental IDS model over every discoverable dataset under
    ``dataset_path`` and persist the result to :data:`app.config.MODEL_PATH`.

    The function also prints a short classification report computed on a
    held-out slice sampled from the first few training chunks. The sample
    is deliberately small — it's a smoke test, not a formal evaluation.

    Raises
    ------
    ValueError
        If no usable training data was discovered.
    """
    vectorizer = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
    )

    classifier = SGDClassifier(loss="log_loss", random_state=42)

    first_batch = True
    eval_frames: list[pd.DataFrame] = []
    files_processed = 0
    rows_processed = 0

    for source_type, payload in iter_training_sources(dataset_path):
        try:
            if source_type == "paired_csv":
                dataset_file, labels_file = payload
                print(f"Training from paired dataset: {dataset_file}")
                print(f"Using labels file:            {labels_file}")
                chunk_iter = iter_paired_csv_training_chunks(
                    dataset_file, labels_file, CHUNK_SIZE
                )
                files_processed += 1

            elif source_type == "single_csv":
                csv_file = payload
                print(f"Training from single labeled CSV: {csv_file}")
                chunk_iter = iter_csv_training_chunks(csv_file, CHUNK_SIZE)
                files_processed += 1

            else:
                continue

            chunk_index = 0
            for chunk in chunk_iter:
                if chunk.empty:
                    continue

                # Convert each row into its flat text representation.
                chunk["text"] = chunk.apply(row_to_text, axis=1)
                X_text = chunk["text"]
                y = chunk["label"]

                X_vec = vectorizer.transform(X_text)

                if first_batch:
                    # ``classes=`` is only allowed on the first partial_fit.
                    classifier.partial_fit(X_vec, y, classes=CLASSES)
                    first_batch = False
                else:
                    classifier.partial_fit(X_vec, y)

                rows_processed += len(chunk)

                # Hold onto a slice of the first couple of chunks for the
                # end-of-run smoke evaluation.
                if len(eval_frames) < 3:
                    eval_frames.append(chunk.head(2000).copy())

                print(
                    f"  processed chunk {chunk_index} "
                    f"({len(chunk)} rows, total rows={rows_processed})"
                )

                chunk_index += 1
                # Explicit cleanup — these DataFrames can be large.
                del chunk
                del X_text
                del y
                del X_vec
                gc.collect()

        except Exception as exc:
            # Don't let one bad dataset kill the whole run.
            print(f"Skipped source due to error: {exc}")

    if first_batch:
        raise ValueError(
            "No valid training data found. "
            "Provide either paired *_dataset.csv + *_labels.csv files, "
            "or single CSV files containing both features/text and a label column."
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": CLASSES,
    }
    joblib.dump(model_bundle, MODEL_PATH, compress=3)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Files processed: {files_processed}")
    print(f"Rows processed: {rows_processed}")

    # ------------------------------------------------------------------ #
    # Smoke-test evaluation
    # ------------------------------------------------------------------ #
    # NOTE: This is intentionally trained from scratch on the sample slice
    # just to sanity-check that the feature shape works. The real model is
    # the streamed one saved above.
    if eval_frames:
        eval_df = pd.concat(eval_frames, ignore_index=True)

        if len(eval_df) > 10:
            X_eval = eval_df["text"]
            y_eval = eval_df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_eval, y_eval, test_size=0.3, random_state=42
            )

            temp_clf = SGDClassifier(loss="log_loss", random_state=42)
            X_train_vec = vectorizer.transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            temp_clf.partial_fit(X_train_vec, y_train, classes=CLASSES)
            y_pred = temp_clf.predict(X_test_vec)

            print("\nQuick evaluation:\n")
            print(classification_report(y_test, y_pred, zero_division=0))
