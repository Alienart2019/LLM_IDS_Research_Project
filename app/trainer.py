import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.config import MODEL_PATH


CHUNK_SIZE = 5000
CLASSES = ["benign", "suspicious", "malicious"]


def _map_label_value(value) -> str | None:
    if pd.isna(value):
        return None

    val = str(value).strip().lower()

    benign_values = {"0", "normal", "benign", "false", "no", "clean"}
    suspicious_values = {"suspicious", "anomaly", "anomalous", "unknown", "suspect"}
    malicious_values = {
        "1", "attack", "malicious", "true", "yes", "wiretap", "dos",
        "ddos", "mitm", "mirai", "fuzzing", "scan", "flood",
        "renegotiation", "injection", "exploit", "malware"
    }

    if val in benign_values:
        return "benign"
    if val in suspicious_values:
        return "suspicious"
    if val in malicious_values:
        return "malicious"

    if "benign" in val or "normal" in val:
        return "benign"
    if "suspicious" in val or "anomaly" in val:
        return "suspicious"
    if any(word in val for word in [
        "attack", "malicious", "wiretap", "dos", "ddos",
        "mitm", "mirai", "scan", "flood", "injection", "exploit"
    ]):
        return "malicious"

    return None


def _row_to_text(row: pd.Series) -> str:
    parts = []
    for col, value in row.items():
        if col == "label":
            continue
        parts.append(f"{col}={value}")
    return " ".join(parts)


def _find_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "label", "labels", "class", "target", "attack", "attack_type",
        "category", "malicious", "outcome", "y", "x"
    ]

    normalized = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        if cand in normalized:
            return normalized[cand]

    non_unnamed = [c for c in df.columns if not str(c).strip().lower().startswith("unnamed:")]
    if len(non_unnamed) == 1:
        return non_unnamed[0]

    if len(df.columns) == 1:
        return df.columns[0]

    raise ValueError(f"Could not determine label column from: {list(df.columns)}")


def _load_dataset_and_labels(dataset_file: Path, labels_file: Path) -> pd.DataFrame:
    dataset_df = pd.read_csv(dataset_file)
    labels_df = pd.read_csv(labels_file)

    label_col = _find_label_column(labels_df)

    if len(dataset_df) != len(labels_df):
        raise ValueError(
            f"Row count mismatch: {dataset_file.name} has {len(dataset_df)} rows, "
            f"but {labels_file.name} has {len(labels_df)} rows."
        )

    combined = dataset_df.copy()
    combined["label"] = labels_df[label_col].apply(_map_label_value)
    combined = combined.dropna(subset=["label"])

    return combined


def _iter_dataset_pairs(dataset_root: str):
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    for dataset_file in root.rglob("*_dataset.csv"):
        base_name = dataset_file.name.replace("_dataset.csv", "")
        folder = dataset_file.parent

        possible_label_files = [
            folder / f"{base_name}_labels.csv",
            folder / f"{base_name.lower()}_labels.csv",
            folder / f"{base_name.upper()}_labels.csv",
        ]

        labels_file = None
        for candidate in possible_label_files:
            if candidate.exists():
                labels_file = candidate
                break

        if labels_file is None:
            label_matches = list(folder.glob("*_labels.csv"))
            if len(label_matches) == 1:
                labels_file = label_matches[0]

        if labels_file is not None:
            yield dataset_file, labels_file
        else:
            print(f"Skipping {dataset_file} because no matching labels file was found.")


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

    for dataset_file, labels_file in _iter_dataset_pairs(dataset_path):
        print(f"Training from dataset: {dataset_file}")
        print(f"Using labels file:   {labels_file}")

        try:
            combined_df = _load_dataset_and_labels(dataset_file, labels_file)
            files_processed += 1

            for start in range(0, len(combined_df), CHUNK_SIZE):
                chunk = combined_df.iloc[start:start + CHUNK_SIZE].copy()

                if chunk.empty:
                    continue

                chunk["text"] = chunk.apply(_row_to_text, axis=1)
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
            print(f"Skipped pair due to error: {e}")

    if first_batch:
        raise ValueError("No valid training data found from dataset/label pairs.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": CLASSES,
    }
    joblib.dump(model_bundle, MODEL_PATH, compress=3)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Dataset pairs processed: {files_processed}")
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