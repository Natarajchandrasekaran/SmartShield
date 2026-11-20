"""
SmartShield — Standalone Edge Model Exporter
Exports scikit-learn model weights into pure-Python JSON format.
Enables 100% dependency-free ML inference on Android (Zero pip/ninja/clang required!).
"""

import os
import sys
import json
import math
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "smartshield_ml_model.joblib")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "edge_model_weights.json")


def export():
    print("Loading joblib model...")
    pipeline = joblib.load(MODEL_PATH)
    union = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]

    word_vec = union.transformer_list[0][1]
    char_vec = union.transformer_list[1][1]

    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    # Word weights: combine idf and logistic coef
    # tf-idf value for a term is (tf) * idf
    word_weights = {}
    for term, idx in word_vec.vocabulary_.items():
        weight = coef[idx] * word_vec.idf_[idx]
        if abs(weight) > 0.001:  # prune tiny weights
            word_weights[term] = round(weight, 5)

    # Char weights:
    char_offset = len(word_vec.vocabulary_)
    char_weights = {}
    for term, idx in char_vec.vocabulary_.items():
        weight = coef[char_offset + idx] * char_vec.idf_[idx]
        if abs(weight) > 0.001:  # prune tiny weights
            char_weights[term] = round(weight, 5)

    data = {
        "intercept": round(intercept, 5),
        "word_weights": word_weights,
        "char_weights": char_weights
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Standalone weights exported to: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"Non-zero Word terms: {len(word_weights)}, Char terms: {len(char_weights)}")


if __name__ == "__main__":
    export()
