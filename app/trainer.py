import os

import joblib
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.config import MODEL_PATH
from app.dataloader import (
    iter_training_sources,
    load_single_csv_for_training,
    load_dataset_and_labels
)
from app.features import row_to_text


CHUNK_SIZE = 5000
CLASSES = ["benign", "suspicious", "malicious"]


def train_model(dataset_path: str) -> None:
    vectorizer = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True
    )

    classifier = SGDClassifier(
        loss="log_loss",
        random_state=42
    )

    first_batch = True
    eval_frames = []
    files_processed = 0
    rows_processed = 0

    for source_type, payload in iter_training_sources(dataset_path):
        try:
            if source_type == "paired_csv":
                dataset_file, labels_file = payload
                print(f"Training from paired dataset: {dataset_file}")
                print(f"Using labels file:            {labels_file}")
                combined_df = load_dataset_and_labels(dataset_file, labels_file)
                files_processed += 1

            elif source_type == "single_csv":
                csv_file = payload
                print(f"Training from single labeled CSV: {csv_file}")
                combined_df = load_single_csv_for_training(csv_file)
                files_processed += 1

            else:
                continue

            for start in range(0, len(combined_df), CHUNK_SIZE):
                chunk = combined_df.iloc[start:start + CHUNK_SIZE].copy()

                if chunk.empty:
                    continue

                chunk["text"] = chunk.apply(row_to_text, axis=1)
                X_text = chunk["text"]
                y = chunk["label"]

                X_vec = vectorizer.transform(X_text)

                if first_batch:
                    classifier.partial_fit(X_vec, y, classes=CLASSES)
                    first_batch = False
                else:
                    classifier.partial_fit(X_vec, y)

                rows_processed += len(chunk)

                if len(eval_frames) < 3:
                    eval_frames.append(chunk)

                print(f"  processed rows {start} to {start + len(chunk) - 1}")

        except Exception as e:
            print(f"Skipped source due to error: {e}")

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

    if eval_frames:
        eval_df = pd.concat(eval_frames, ignore_index=True)

        if len(eval_df) > 10:
            X_eval = eval_df["text"]
            y_eval = eval_df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_eval,
                y_eval,
                test_size=0.3,
                random_state=42
            )

            temp_clf = SGDClassifier(
                loss="log_loss",
                random_state=42
            )

            X_train_vec = vectorizer.transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            temp_clf.partial_fit(X_train_vec, y_train, classes=CLASSES)
            y_pred = temp_clf.predict(X_test_vec)

            print("\nQuick evaluation:\n")
            print(classification_report(y_test, y_pred, zero_division=0))