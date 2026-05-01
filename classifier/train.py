"""
classifier/train.py
Clutch.ai — MLP Question Classifier Training

Defines QuestionClassifier (2-layer MLP on 384-dim MiniLM embeddings),
trains it on the dataset from dataset.py, evaluates on the test set,
saves training curves to eval/training_curves.png, and saves the trained
model to models/question_clf.pkl.

Usage:
    python classifier/train.py
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for headless environments)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from classifier.dataset import build_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_PATH   = Path(os.getenv("MODELS_PATH", "./models"))
EVAL_PATH     = Path("./eval")
MODEL_FILE    = MODELS_PATH / "question_clf.pkl"
CURVES_FILE   = EVAL_PATH / "training_curves.png"
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS     = 50
BATCH_SIZE = 16
LR         = 1e-3
WD         = 1e-4
INPUT_DIM  = 384
NUM_CLASSES = 3
CLASS_NAMES = ["technical_question", "personal_behavioral", "noise"]

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class QuestionClassifier(nn.Module):
    """
    2-layer MLP on 384-dim MiniLM embeddings.
    Architecture: 384 → 128 (ReLU, Dropout 0.3) → 64 (ReLU, Dropout 0.2) → 3
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    X_train, X_test, y_train, y_test, embedder_used = build_dataset()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t  = torch.FloatTensor(X_test)
    y_test_t  = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model ---
    model     = QuestionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    print("\n" + "=" * 65)
    print("Clutch.ai — Question Classifier Training")
    print(f"Model:     {model}")
    print(f"Epochs:    {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print("=" * 65 + "\n")

    # --- Training loop ---
    train_losses: list = []
    train_accs:   list = []
    test_accs:    list = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

        avg_loss  = epoch_loss / total
        train_acc = correct / total * 100

        # --- Test evaluation (no gradient) ---
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_preds  = test_logits.argmax(dim=1)
            test_acc    = (test_preds == y_test_t).float().mean().item() * 100

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Test Acc: {test_acc:.1f}%")

    # --- Final evaluation ---
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds  = test_logits.argmax(dim=1).numpy()

    print("\n" + "=" * 65)
    print("Final Classification Report:")
    print("=" * 65)
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_preds)
    print(cm)

    final_test_acc = (test_preds == y_test).mean() * 100
    print(f"\nFinal Test Accuracy: {final_test_acc:.1f}%")
    if final_test_acc < 80:
        print("⚠️  WARNING: Test accuracy is below 80%. "
              "Consider adding more training examples in classifier/dataset.py "
              "or checking class balance.")

    # --- Save training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Clutch.ai — Classifier Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(range(1, EPOCHS + 1), train_losses, color="#3B82F6", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, EPOCHS + 1), train_accs, label="Train Acc", color="#10B981", linewidth=2)
    ax2.plot(range(1, EPOCHS + 1), test_accs,  label="Test Acc",  color="#F59E0B", linewidth=2, linestyle="--")
    ax2.set_title("Train vs Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(CURVES_FILE, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[TRAIN] Training curves saved to: {CURVES_FILE}")

    # --- Save model ---
    save_data = {
        "model_state_dict": model.state_dict(),
        "embedder_name":    EMBEDDER_NAME,
        "class_names":      CLASS_NAMES,
        "input_dim":        INPUT_DIM,
        "num_classes":      NUM_CLASSES,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_data, f)
    print(f"[TRAIN] Model saved to: {MODEL_FILE}")
    print("[TRAIN] Training complete ✓")


if __name__ == "__main__":
    train()
