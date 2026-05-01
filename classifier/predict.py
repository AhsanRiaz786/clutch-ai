"""
classifier/predict.py
Clutch.ai — Classifier Inference

Priority:
    1. BiLSTM model (models/lstm_clf.pkl) — if trained, uses token-level BiLSTM
    2. MLP fallback  (models/question_clf.pkl) — sentence-embedding MLP baseline

Labels: 'technical_question' | 'personal_behavioral' | 'noise'

Usage:
    from classifier.predict import predict
    label, confidence = predict("What is a binary search tree?")
"""

import os
import sys
import pickle
import threading
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_PATH   = Path(os.getenv("MODELS_PATH", "./models"))
LSTM_FILE     = MODELS_PATH / "lstm_clf.pkl"
MLP_FILE      = MODELS_PATH / "question_clf.pkl"
_DEFAULT_CLASS_NAMES = ["technical_question", "personal_behavioral", "noise"]

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_lstm_model    = None
_lstm_embedder = None
_lstm_classes: List[str] = _DEFAULT_CLASS_NAMES

_mlp_model    = None
_mlp_embedder = None
_mlp_classes:  List[str] = _DEFAULT_CLASS_NAMES

_using_lstm = False   # which backend is active
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lazy loader
# ---------------------------------------------------------------------------

def _ensure_loaded() -> None:
    global _lstm_model, _lstm_embedder, _lstm_classes
    global _mlp_model,  _mlp_embedder,  _mlp_classes
    global _using_lstm

    if _lstm_model is not None or _mlp_model is not None:
        return

    with _lock:
        if _lstm_model is not None or _mlp_model is not None:
            return

        # Try BiLSTM first
        if LSTM_FILE.exists():
            try:
                print("[CLASSIFY] Loading BiLSTM classifier ...")
                from classifier.lstm_classifier import (
                    BiLSTMClassifier, load_lstm_classifier,
                )
                _lstm_model, _lstm_embedder, _lstm_classes = load_lstm_classifier()
                _using_lstm = True
                print("[CLASSIFY] BiLSTM classifier loaded ✓")
                return
            except Exception as e:
                print(f"[CLASSIFY] BiLSTM load failed ({e}), falling back to MLP ...")

        # Fall back to MLP
        if not MLP_FILE.exists():
            raise FileNotFoundError(
                f"No classifier model found. Run either:\n"
                f"  python classifier/lstm_classifier.py   (BiLSTM — recommended)\n"
                f"  python classifier/train.py             (MLP baseline)"
            )

        print("[CLASSIFY] Loading MLP classifier ...")
        from classifier.train import QuestionClassifier

        with open(MLP_FILE, "rb") as f:
            save_data = pickle.load(f)

        _mlp_model = QuestionClassifier()
        _mlp_model.load_state_dict(save_data["model_state_dict"])
        _mlp_model.eval()
        _mlp_classes = save_data.get("class_names", _DEFAULT_CLASS_NAMES)

        from sentence_transformers import SentenceTransformer
        _mlp_embedder = SentenceTransformer(
            save_data.get("embedder_name", "all-MiniLM-L6-v2")
        )
        _using_lstm = False
        print("[CLASSIFY] MLP classifier loaded ✓")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(text: str) -> Tuple[str, float]:
    """
    Classify text as one of: 'technical_question', 'personal_behavioral', 'noise'.

    Uses BiLSTM if available (models/lstm_clf.pkl), otherwise falls back to
    the MLP baseline (models/question_clf.pkl).

    Returns:
        (label, confidence_percent)
    """
    _ensure_loaded()

    if _using_lstm:
        from classifier.lstm_classifier import predict_lstm
        return predict_lstm(text, _lstm_model, _lstm_embedder, _lstm_classes)

    # MLP path
    embedding = _mlp_embedder.encode([text], show_progress_bar=False)
    x = torch.FloatTensor(embedding)
    with torch.no_grad():
        logits = _mlp_model(x)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()

    label      = _mlp_classes[pred]
    confidence = probs[0][pred].item() * 100
    return label, confidence


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        "What is the difference between TCP and UDP?",
        "Tell me about yourself.",
        "So I would start by initializing a pointer to null.",
        "Explain how a hash table handles collisions.",
        "Walk me through how merge sort works.",
        "What is your biggest weakness?",
        "Um okay right.",
        "How would you design a rate limiter?",
        "Tell me about a time you faced a difficult challenge.",
    ]

    print("\n--- Clutch.ai Classifier Test ---")
    for sentence in test_sentences:
        label, conf = predict(sentence)
        print(f"  [{label} {conf:.1f}%] {sentence}")
