"""
SmartShield — Fast ML Classifier Trainer
Uses TF-IDF Char/Word N-grams + Calibrated Logistic Regression.
Ideal for ARM edge devices due to ultra-low memory (<5MB) and sub-millisecond inference!

Usage:
    python scripts/train_baseline.py
"""

import os
import sys
import csv
import json
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(split_name):
    filepath = os.path.join(DATA_DIR, f"{split_name}.csv")
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(1 if row["label"] == "scam" else 0)
    return texts, labels


def train():
    print("=" * 60)
    print("SmartShield — Fast Edge ML Classifier Training")
    print("=" * 60)

    # 1. Load data
    train_texts, train_labels = load_data("train")
    val_texts, val_labels = load_data("val")
    test_texts, test_labels = load_data("test")

    print(f"Loaded: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # 2. Build pipeline: Word N-grams + Character N-grams (crucial for Hinglish & obfuscation)
    start_t = time.perf_counter()
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=3000, lowercase=True)),
        ("char_tfidf", TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=5000, lowercase=True))
    ])

    pipeline = Pipeline([
        ("features", feature_union),
        ("clf", LogisticRegression(class_weight="balanced", C=1.5, max_iter=500, random_state=42))
    ])

    # 3. Fit
    pipeline.fit(train_texts, train_labels)
    train_time_ms = (time.perf_counter() - start_t) * 1000
    print(f"Model trained in {train_time_ms:.1f} ms")

    # 4. Evaluate on Test set
    test_preds = pipeline.predict(test_texts)
    test_probs = pipeline.predict_proba(test_texts)[:, 1]

    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary")
    cm = confusion_matrix(test_labels, test_preds)
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0

    print("\n--- Test Evaluation Results ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {p * 100:.2f}%")
    print(f"Recall:    {r * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"False Positive Rate (FPR): {fpr * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"             Pred Legit  Pred Scam")
    print(f"  Act Legit     {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  Act Scam      {cm[1][0]:>5}      {cm[1][1]:>5}")

    # 5. Save model
    model_path = os.path.join(MODEL_DIR, "smartshield_ml_model.joblib")
    joblib.dump(pipeline, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"\nModel saved to: {model_path} ({model_size_kb:.1f} KB)")

    # 6. Save benchmark report
    benchmark_file = os.path.join(PROJECT_ROOT, "benchmarks", "ml_results.json")
    results = {
        "model_type": "TF-IDF (Word+Char) + Balanced LogisticRegression",
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "test_accuracy": acc,
        "test_precision": p,
        "test_recall": r,
        "test_f1": f1,
        "false_positive_rate": fpr,
        "confusion_matrix": cm.tolist(),
        "model_size_kb": model_size_kb,
        "train_time_ms": train_time_ms
    }
    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {benchmark_file}")
    print("=" * 60)


if __name__ == "__main__":
    train()
