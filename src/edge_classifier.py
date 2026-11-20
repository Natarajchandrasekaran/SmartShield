"""
SmartShield — Pure-Python Standalone Edge Classifier
Zero-dependency ML inference engine (requires NO scikit-learn, NO numpy, NO C/C++ compiler!).
Perfect for Android Termux, embedded Linux, and ultra-low-resource ARM environments.
"""

import os
import json
import math
import re


class StandaloneEdgeClassifier:
    """
    Pure-Python inference engine for the TF-IDF + Logistic Regression model.
    Runs entirely using Python standard library (re, json, math).
    """

    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "edge_model_weights.json"
            )
        
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.intercept = data["intercept"]
        self.word_weights = data["word_weights"]
        self.char_weights = data["char_weights"]

    def predict_proba(self, text: str) -> float:
        """
        Predict probability of SMS being a scam.
        Returns: float between 0.0 (legitimate) and 1.0 (scam).
        """
        tl = text.lower()
        score = self.intercept

        # Extract words (1-3 ngrams)
        words = re.findall(r'\b\w+\b', tl)
        n_words = len(words)
        for n in (1, 2, 3):
            for i in range(n_words - n + 1):
                gram = ' '.join(words[i:i+n])
                if gram in self.word_weights:
                    score += self.word_weights[gram]

        # Extract characters (3-5 ngrams)
        n_chars = len(tl)
        for n in (3, 4, 5):
            for i in range(n_chars - n + 1):
                gram = tl[i:i+n]
                if gram in self.char_weights:
                    score += self.char_weights[gram]

        # Sigmoid with overflow clamping
        clamped_score = max(-20.0, min(20.0, score))
        return 1.0 / (1.0 + math.exp(-clamped_score))
