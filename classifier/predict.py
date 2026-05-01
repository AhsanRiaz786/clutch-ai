"""
classifier/predict.py
Clutch.ai — Classifier Inference Wrapper

Loads the saved QuestionClassifier from models/question_clf.pkl and exposes
a single predict(text) function that returns one of:
    'technical_question' | 'small_talk' | 'other'

The model is loaded lazily on the first call (module-level singleton).

Usage:
    from classifier.predict import predict
    label = predict("What is a binary search tree?")
    # → 'technical_question'
"""

import os
import sys
import pickle
import threading
from pathlib import Path
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# We import the class so pickle can reconstruct it
from classifier.train import QuestionClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_PATH = Path(os.getenv("MODELS_PATH", "./models"))
MODEL_FILE  = MODELS_PATH / "question_clf.pkl"

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_model:    Optional[QuestionClassifier] = None
_embedder = None
_class_names: list = ["technical_question", "small_talk", "other"]
_lock = threading.Lock()


def load_classifier() -> QuestionClassifier:
    """
    Loads models/question_clf.pkl and returns the model in eval mode.
    Raises FileNotFoundError if model has not been trained yet.
    """
    global _model, _embedder, _class_names

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}. "
            "Run 'python classifier/train.py' first."
        )

    with open(MODEL_FILE, "rb") as f:
        save_data = pickle.load(f)

    model = QuestionClassifier()
    model.load_state_dict(save_data["model_state_dict"])
    model.eval()

    _class_names = save_data.get("class_names", _class_names)
    embedder_name = save_data.get("embedder_name", "all-MiniLM-L6-v2")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(embedder_name)

    return model, embedder


def _ensure_loaded() -> None:
    """Lazy-loads model and embedder on first predict() call."""
    global _model, _embedder

    if _model is None:
        with _lock:
            if _model is None:
                print("[CLASSIFY] Loading question classifier ...")
                _model, _embedder = load_classifier()
                print("[CLASSIFY] Classifier loaded ✓")


def predict(text: str) -> tuple:
    """
    Classifies text as one of: 'technical_question', 'small_talk', 'other'.

    Args:
        text: The transcribed sentence to classify.

    Returns:
        Tuple of (label: str, confidence: float 0–100).
    """
    _ensure_loaded()

    embedding = _embedder.encode([text], show_progress_bar=False)
    x = torch.FloatTensor(embedding)

    with torch.no_grad():
        logits = _model(x)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()

    confidence = probs[0][pred].item() * 100
    label = _class_names[pred]
    print(f"[CLASSIFY] '{text[:60]}...' → {label} ({confidence:.1f}%)")
    return label, confidence


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        "What is the difference between TCP and UDP?",
        "How was your weekend?",
        "I worked on a React project last year.",
        "Explain how a hash table handles collisions.",
        "Tell me about yourself.",
        "Let me think about that for a second.",
    ]

    print("\n--- Clutch.ai Classifier Test ---")
    for sentence in test_sentences:
        label, conf = predict(sentence)
        print(f"  [{label} {conf:.1f}%] {sentence}")
