import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.features import row_to_text
from app.config import MODEL_PATH


def train_model(dataset_path: str) -> None:
    df = pd.read_csv(dataset_path)

    required_columns = {"message", "service", "hostname", "source_ip", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["text"] = df.apply(row_to_text, axis=1)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
