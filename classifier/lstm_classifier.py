"""
classifier/lstm_classifier.py
Clutch.ai — BiLSTM Question Classifier with Attention Pooling

Deep Learning Enhancement #1  (CS-419 Week 11 — RNNs and LSTMs)

Architecture:
    MiniLM token embeddings (384-dim per token, from transformer last hidden state)
    → BiLSTM  (2 layers, 128 hidden units, bidirectional  →  256-dim output)
    → Learned attention pooling over all hidden states
    → MLP head (256 → 64 → 3 classes)

Why BiLSTM over MLP baseline:
    The MLP compresses the entire question to a single 384-dim vector before classification.
    The BiLSTM sees the sequence of token embeddings and learns which tokens are
    discriminative (e.g. "what IS", "TELL me about", "HOW does"). The attention
    mechanism then weights those tokens to produce a richer context vector.

    MLP baseline:  384-dim sentence embedding → 384→128→64→3
    BiLSTM:        (seq_len × 384) token embeddings → BiLSTM → attention → 256→64→3

Usage:
    python classifier/lstm_classifier.py          # train + evaluate
    python classifier/lstm_classifier.py --eval   # only run eval on saved model
"""

import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from classifier.dataset import TECHNICAL_QUESTIONS, PERSONAL_BEHAVIORAL, NOISE, LABEL_MAP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_PATH   = Path(os.getenv("MODELS_PATH", "./models"))
EVAL_PATH     = Path("./eval")
MODEL_FILE    = MODELS_PATH / "lstm_clf.pkl"
CURVES_FILE   = EVAL_PATH / "lstm_training_curves.png"
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
CLASS_NAMES   = ["technical_question", "personal_behavioral", "noise"]

EPOCHS      = 50
BATCH_SIZE  = 16
LR          = 5e-4
WD          = 1e-4
INPUT_DIM   = 384    # MiniLM-L6-v2 hidden size
HIDDEN_DIM  = 128    # per direction; bidirectional → 256 total
NUM_LAYERS  = 2
NUM_CLASSES = 3
DROPOUT     = 0.3


# ---------------------------------------------------------------------------
# Model: BiLSTM with multi-head attention pooling
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier with learned attention pooling.

    Input:  padded token embeddings  (batch, max_seq_len, 384)
    Output: class logits             (batch, num_classes)

    Attention pooling:
        score_t = tanh(W * h_t) · v           (learned query vector v)
        weight_t = softmax(score_t)
        context  = Σ weight_t * h_t
    This is additive (Bahdanau-style) self-attention over the LSTM outputs.
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        hidden_dim:  int   = HIDDEN_DIM,
        num_layers:  int   = NUM_LAYERS,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * 2   # bidirectional concatenation → 256

        # Additive attention: projects each hidden state to a scalar score
        self.attn_proj = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        self.norm = nn.LayerNorm(out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (batch, max_seq_len, input_dim)  padded token embeddings
            lengths: (batch,)  actual un-padded sequence lengths
        Returns:
            logits:  (batch, num_classes)
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out: (batch, max_len, hidden*2)

        # Attention: mask padding positions before softmax
        attn_scores = self.attn_proj(lstm_out).squeeze(-1)   # (batch, max_len)
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)   # (batch, max_len, 1)

        context = (lstm_out * attn_weights).sum(dim=1)   # (batch, hidden*2)
        context = self.norm(context)
        return self.classifier(context)


# ---------------------------------------------------------------------------
# Token embedding extraction
# ---------------------------------------------------------------------------

def _extract_token_embeddings(
    texts: List[str], embedder
) -> List[torch.Tensor]:
    """
    Passes each text through MiniLM's Transformer layer only (before mean pooling)
    to obtain per-token hidden states. Returns a list of (seq_len, 384) tensors —
    one per text, variable length, no padding.
    """
    token_seqs = []
    transformer_module = embedder[0]   # sentence_transformers: [Transformer, Pooling, ...]

    for text in texts:
        features = embedder.tokenize([text])
        features = {k: v.to(embedder.device) for k, v in features.items()}

        with torch.no_grad():
            out = transformer_module.forward(features)
            all_embs = out["token_embeddings"]     # (1, padded_len, 384)
            mask     = out["attention_mask"]       # (1, padded_len)  1=real, 0=pad

        real_embs = all_embs[0][mask[0].bool()].cpu()   # (real_seq_len, 384)
        token_seqs.append(real_embs)

    return token_seqs


# ---------------------------------------------------------------------------
# PyTorch Dataset + DataLoader utilities
# ---------------------------------------------------------------------------

class TokenSequenceDataset(Dataset):
    def __init__(self, seqs: List[torch.Tensor], labels: List[int]) -> None:
        self.seqs   = seqs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.seqs[idx], self.labels[idx]


def _collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


# ---------------------------------------------------------------------------
# Dataset builder (token-level)
# ---------------------------------------------------------------------------

def build_token_dataset(embedder_name: str = EMBEDDER_NAME):
    """
    Builds the token-embedding dataset for BiLSTM training.

    Returns:
        train_seqs, test_seqs  — lists of (seq_len, 384) tensors
        train_labels, test_labels — lists of ints
        embedder_name — the model used
    """
    from sentence_transformers import SentenceTransformer
    print(f"[LSTM] Loading embedder: {embedder_name} ...")
    embedder = SentenceTransformer(embedder_name)
    print("[LSTM] Embedder loaded ✓")

    all_texts:  List[str] = []
    all_labels: List[int] = []

    for t in TECHNICAL_QUESTIONS:
        all_texts.append(t)
        all_labels.append(LABEL_MAP["technical_question"])
    for t in PERSONAL_BEHAVIORAL:
        all_texts.append(t)
        all_labels.append(LABEL_MAP["personal_behavioral"])
    for t in NOISE:
        all_texts.append(t)
        all_labels.append(LABEL_MAP["noise"])

    counts = {c: all_labels.count(i) for c, i in LABEL_MAP.items()}
    print(f"[LSTM] Dataset: {' | '.join(f'{k}={v}' for k,v in counts.items())} | Total={len(all_texts)}")

    print("[LSTM] Extracting token embeddings (this may take ~30s) ...")
    token_seqs = _extract_token_embeddings(all_texts, embedder)
    lengths = [t.shape[0] for t in token_seqs]
    print(f"[LSTM] Sequence lengths — min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")

    indices     = list(range(len(all_texts)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_seqs   = [token_seqs[i]  for i in train_idx]
    train_labels = [all_labels[i]  for i in train_idx]
    test_seqs    = [token_seqs[i]  for i in test_idx]
    test_labels  = [all_labels[i]  for i in test_idx]

    print(f"[LSTM] Train: {len(train_seqs)} | Test: {len(test_seqs)}")
    return train_seqs, test_seqs, train_labels, test_labels, embedder_name


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    train_seqs, test_seqs, train_labels, test_labels, emb_name = build_token_dataset()

    train_loader = DataLoader(
        TokenSequenceDataset(train_seqs, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        TokenSequenceDataset(test_seqs, test_labels),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_fn,
    )

    model     = BiLSTMClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 65)
    print("Clutch.ai — BiLSTM Classifier Training")
    print(f"Parameters: {total_params:,}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | WD: {WD}")
    print("=" * 65 + "\n")

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = correct = total = 0

        for padded, lengths, labels in train_loader:
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        scheduler.step()

        train_acc = correct / total * 100
        avg_loss  = epoch_loss / total

        # Test accuracy
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for padded, lengths, labels in test_loader:
                preds         = model(padded, lengths).argmax(1)
                test_correct += (preds == labels).sum().item()
                test_total   += len(labels)
        test_acc = test_correct / test_total * 100

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

    # --- Final evaluation ---
    model.eval()
    all_preds, all_labels_np = [], []
    with torch.no_grad():
        for padded, lengths, labels in test_loader:
            preds = model(padded, lengths).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels_np.extend(labels.numpy())

    print("\n" + "=" * 65)
    print("Final Classification Report:")
    print(classification_report(all_labels_np, all_preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, all_preds))
    final_acc = np.mean(np.array(all_preds) == np.array(all_labels_np)) * 100
    print(f"\nFinal Test Accuracy: {final_acc:.1f}%")

    # --- Training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Clutch.ai — BiLSTM Classifier Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(range(1, EPOCHS + 1), train_losses, color="#3B82F6", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, EPOCHS + 1), train_accs, label="Train", color="#10B981", linewidth=2)
    ax2.plot(range(1, EPOCHS + 1), test_accs,  label="Test",  color="#F59E0B", linewidth=2, linestyle="--")
    ax2.set_title("Train vs Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(CURVES_FILE, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[LSTM] Training curves saved to {CURVES_FILE}")

    # --- Save model ---
    save_data = {
        "model_state_dict": model.state_dict(),
        "embedder_name":    emb_name,
        "class_names":      CLASS_NAMES,
        "input_dim":        INPUT_DIM,
        "hidden_dim":       HIDDEN_DIM,
        "num_layers":       NUM_LAYERS,
        "num_classes":      NUM_CLASSES,
        "dropout":          DROPOUT,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_data, f)
    print(f"[LSTM] Model saved to {MODEL_FILE}")
    print("[LSTM] Training complete ✓\n")


# ---------------------------------------------------------------------------
# Standalone inference helper (used by predict.py)
# ---------------------------------------------------------------------------

def load_lstm_classifier():
    """
    Loads the saved BiLSTM model from disk.
    Returns (model, embedder, class_names) or raises FileNotFoundError.
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"BiLSTM model not found at {MODEL_FILE}. "
            "Run 'python classifier/lstm_classifier.py' first."
        )
    with open(MODEL_FILE, "rb") as f:
        save_data = pickle.load(f)

    model = BiLSTMClassifier(
        input_dim   = save_data.get("input_dim",   INPUT_DIM),
        hidden_dim  = save_data.get("hidden_dim",  HIDDEN_DIM),
        num_layers  = save_data.get("num_layers",  NUM_LAYERS),
        num_classes = save_data.get("num_classes", NUM_CLASSES),
        dropout     = save_data.get("dropout",     DROPOUT),
    )
    model.load_state_dict(save_data["model_state_dict"])
    model.eval()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(save_data.get("embedder_name", EMBEDDER_NAME))
    class_names = save_data.get("class_names", CLASS_NAMES)

    return model, embedder, class_names


def predict_lstm(
    text: str,
    model: BiLSTMClassifier,
    embedder,
    class_names: List[str],
) -> Tuple[str, float]:
    """
    Single-sample inference using the BiLSTM model.

    Args:
        text:        Transcribed question text.
        model:       Loaded BiLSTMClassifier in eval mode.
        embedder:    SentenceTransformer used for token embeddings.
        class_names: List of class name strings.

    Returns:
        (label, confidence_percent)
    """
    seqs = _extract_token_embeddings([text], embedder)
    seq  = seqs[0].unsqueeze(0)           # (1, seq_len, 384)
    lengths = torch.tensor([seq.shape[1]], dtype=torch.long)

    with torch.no_grad():
        logits = model(seq, lengths)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()

    label      = class_names[pred]
    confidence = probs[0][pred].item() * 100
    return label, confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Skip training, only evaluate saved model")
    args = parser.parse_args()

    if args.eval:
        print("[LSTM] Loading saved model for evaluation ...")
        model, embedder, class_names = load_lstm_classifier()
        test_sentences = [
            "What is the difference between TCP and UDP?",
            "Tell me about yourself.",
            "So I would start by initializing a pointer.",
            "Explain how a hash table handles collisions.",
            "Walk me through how merge sort works.",
            "What is your biggest weakness?",
            "Uh huh, okay.",
        ]
        print("\n--- BiLSTM Classifier Evaluation ---")
        for sentence in test_sentences:
            label, conf = predict_lstm(sentence, model, embedder, class_names)
            print(f"  [{label} {conf:.1f}%] {sentence}")
    else:
        train()
