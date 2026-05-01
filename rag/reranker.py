"""
rag/reranker.py
Clutch.ai — Cross-Attention Neural Reranker

Deep Learning Enhancement #3  (CS-419 Week 12/13 — Attention & Transformers)

Architecture:
    Given a query q and a candidate chunk c:
        - Project both to Q, K, V matrices via separate linear layers
        - Cross-attention: Q attends over K/V (query asks "which chunk parts matter?")
        - Self-attention on query side to capture internal structure
        - Interaction features: [q_ctx, c_ctx, |q_ctx - c_ctx|, q_ctx * c_ctx]
        - MLP scorer: 4*proj_dim → 256 → 64 → 1
    Training: MarginRankingLoss on (positive, negative) chunk pairs per query.

    This improves retrieval quality: ChromaDB returns top-5 by cosine similarity,
    the reranker re-orders them by learned semantic relevance to the question.

Usage:
    python rag/reranker.py          # train on synthetic pairs from ChromaDB
    python rag/reranker.py --demo   # demo reranking on sample queries
"""

import os
import sys
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_PATH   = Path(os.getenv("MODELS_PATH", "./models"))
EVAL_PATH     = Path("./eval")
MODEL_FILE    = MODELS_PATH / "reranker.pkl"
CURVES_FILE   = EVAL_PATH / "reranker_training_curves.png"
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH   = os.getenv("CHROMA_PATH", "./db")

EMBED_DIM  = 384
PROJ_DIM   = 128   # projection dimension for cross-attention
NUM_HEADS  = 4
DROPOUT    = 0.1
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-3
WD         = 1e-4
MARGIN     = 0.3   # MarginRankingLoss margin


# ---------------------------------------------------------------------------
# Model: Cross-Attention Reranker
# ---------------------------------------------------------------------------

class CrossAttentionReranker(nn.Module):
    """
    Cross-attention reranker for (query, chunk) relevance scoring.

    Intuition:
        The query embedding encodes "what is being asked."
        The chunk embedding encodes "what information is available."
        Cross-attention allows the query representation to selectively focus
        on the most relevant aspects of the chunk, and vice versa.
        The interaction features then capture fine-grained similarity.

    Architecture:
        query_emb (384) --→ proj_q (proj_dim)
        chunk_emb (384) --→ proj_k, proj_v (proj_dim)

        Cross-attention: proj_q attends over proj_k/proj_v
            → q_ctx (proj_dim)   [chunk-aware query representation]

        Self-interaction on chunk side:
            proj_k --→ c_ctx (proj_dim)  [query-aware chunk representation via key]

        Interaction vector: [q_ctx, c_ctx, |q_ctx - c_ctx|, q_ctx * c_ctx]
            size = 4 * proj_dim

        MLP: 4*proj_dim → 256 → 64 → 1 → score
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        proj_dim:  int = PROJ_DIM,
        num_heads: int = NUM_HEADS,
        dropout:   float = DROPOUT,
    ) -> None:
        super().__init__()

        # Projection layers
        self.proj_q = nn.Linear(embed_dim, proj_dim)
        self.proj_k = nn.Linear(embed_dim, proj_dim)
        self.proj_v = nn.Linear(embed_dim, proj_dim)

        # Multi-head cross-attention: query attends to chunk
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = proj_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm_q = nn.LayerNorm(proj_dim)
        self.norm_c = nn.LayerNorm(proj_dim)

        # Interaction scorer
        self.scorer = nn.Sequential(
            nn.Linear(4 * proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _encode_pair(
        self,
        query_embs: torch.Tensor,   # (batch, embed_dim)
        chunk_embs: torch.Tensor,   # (batch, embed_dim)
    ) -> torch.Tensor:
        """Returns relevance scores (batch,)."""
        # Project to lower-dim space
        q = self.proj_q(query_embs).unsqueeze(1)   # (batch, 1, proj_dim)
        k = self.proj_k(chunk_embs).unsqueeze(1)   # (batch, 1, proj_dim)
        v = self.proj_v(chunk_embs).unsqueeze(1)   # (batch, 1, proj_dim)

        # Cross-attention: query attends to chunk
        q_ctx, _ = self.cross_attn(q, k, v)        # (batch, 1, proj_dim)
        q_ctx    = self.norm_q(q_ctx.squeeze(1))   # (batch, proj_dim)
        c_ctx    = self.norm_c(k.squeeze(1))        # (batch, proj_dim)  — chunk key repr

        # Interaction features (captures asymmetric similarity)
        interaction = torch.cat([
            q_ctx,
            c_ctx,
            torch.abs(q_ctx - c_ctx),
            q_ctx * c_ctx,
        ], dim=-1)   # (batch, 4*proj_dim)

        return self.scorer(interaction).squeeze(-1)   # (batch,)

    def forward(
        self,
        query_embs: torch.Tensor,
        chunk_embs: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_pair(query_embs, chunk_embs)

    def score(self, query_emb: np.ndarray, chunk_emb: np.ndarray) -> float:
        """Single-pair inference. Accepts numpy arrays, returns float."""
        q = torch.FloatTensor(query_emb).unsqueeze(0)
        c = torch.FloatTensor(chunk_emb).unsqueeze(0)
        with torch.no_grad():
            return self._encode_pair(q, c).item()


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def _build_training_pairs(embedder) -> Tuple[List, List, List]:
    """
    Generates (query_emb, pos_chunk_emb, neg_chunk_emb) triplets from:
        1. CS interview questions from our classifier dataset
        2. ChromaDB chunks as the retrieval corpus

    Positive: the chunk most similar to the query (top-1 by cosine similarity)
    Hard negative: a chunk from a different topic cluster (top-5 to top-10 range)

    Returns:
        query_embs  (N, 384)
        pos_embs    (N, 384)
        neg_embs    (N, 384)
    """
    import chromadb
    from classifier.dataset import TECHNICAL_QUESTIONS, PERSONAL_BEHAVIORAL

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in client.list_collections()]
    if "clutch_notes" not in existing:
        raise RuntimeError(
            "ChromaDB collection 'clutch_notes' not found. "
            "Run python ingest/ingest.py first."
        )
    collection = client.get_collection("clutch_notes")
    n_chunks   = collection.count()
    print(f"[RERANKER] ChromaDB has {n_chunks} chunks available.")

    # All queries: technical + behavioral (behavioral won't match well — useful negatives)
    queries = TECHNICAL_QUESTIONS[:80] + PERSONAL_BEHAVIORAL[:20]
    random.shuffle(queries)

    print(f"[RERANKER] Embedding {len(queries)} training queries ...")
    query_embs = embedder.encode(queries, show_progress_bar=True, batch_size=32)

    # Fetch ALL chunk embeddings from ChromaDB
    print("[RERANKER] Fetching all chunk embeddings from ChromaDB ...")
    result = collection.get(include=["embeddings", "documents"])
    chunk_embs  = np.array(result["embeddings"], dtype=np.float32)   # (n_chunks, 384)
    chunk_texts = result["documents"]
    print(f"[RERANKER] {len(chunk_texts)} chunks loaded ✓")

    # Build triplets via cosine similarity ranking
    query_list, pos_list, neg_list = [], [], []

    for q_emb in query_embs:
        q_norm     = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        c_norms    = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
        sims       = c_norms @ q_norm                             # (n_chunks,)
        ranked     = np.argsort(-sims)                           # descending

        pos_idx = ranked[0]                                       # most relevant
        # Hard negative: randomly pick from rank 5–min(15, n_chunks)
        hard_neg_pool = ranked[5:min(15, len(ranked))]
        if len(hard_neg_pool) == 0:
            hard_neg_pool = ranked[1:min(5, len(ranked))]
        neg_idx = int(random.choice(hard_neg_pool))

        query_list.append(q_emb)
        pos_list.append(chunk_embs[pos_idx])
        neg_list.append(chunk_embs[neg_idx])

    print(f"[RERANKER] Generated {len(query_list)} training triplets ✓")
    return (
        np.array(query_list, dtype=np.float32),
        np.array(pos_list,   dtype=np.float32),
        np.array(neg_list,   dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _TripletDataset(torch.utils.data.Dataset):
    def __init__(self, Q, P, N):
        self.Q = torch.FloatTensor(Q)
        self.P = torch.FloatTensor(P)
        self.N = torch.FloatTensor(N)

    def __len__(self):
        return len(self.Q)

    def __getitem__(self, idx):
        return self.Q[idx], self.P[idx], self.N[idx]


def train() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    print(f"[RERANKER] Loading embedder: {EMBEDDER_NAME} ...")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    Q, P, N = _build_training_pairs(embedder)

    # Train/val split
    n      = len(Q)
    n_val  = max(1, int(n * 0.15))
    perm   = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = _TripletDataset(Q[train_idx], P[train_idx], N[train_idx])
    val_ds   = _TripletDataset(Q[val_idx],   P[val_idx],   N[val_idx])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = CrossAttentionReranker()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MarginRankingLoss(margin=MARGIN)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 65)
    print("Clutch.ai — Cross-Attention Reranker Training")
    print(f"Parameters: {total_params:,}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} triplets")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Margin: {MARGIN}")
    print("=" * 65 + "\n")

    train_losses, val_losses = [], []
    target = torch.ones(BATCH_SIZE)   # pos should score higher than neg

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = n_batches = 0

        for q, p, neg in train_loader:
            bsz = q.size(0)
            t   = target[:bsz]

            optimizer.zero_grad()
            score_pos = model(q, p)
            score_neg = model(q, neg)
            loss      = criterion(score_pos, score_neg, t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # Validation: measure ranking accuracy (pos > neg rate)
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for q, p, neg in val_loader:
                bsz = q.size(0)
                t   = target[:bsz]
                sp  = model(q, p)
                sn  = model(q, neg)
                val_loss    += criterion(sp, sn, t).item()
                val_correct += (sp > sn).sum().item()
                val_total   += bsz

        avg_val  = val_loss / max(len(val_loader), 1)
        rank_acc = val_correct / max(val_total, 1) * 100

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | Ranking Acc: {rank_acc:.1f}%")

    # --- Training curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Clutch.ai — Cross-Attention Reranker Training", fontsize=14, fontweight="bold")
    ax.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="#3B82F6", linewidth=2)
    ax.plot(range(1, EPOCHS + 1), val_losses,   label="Val Loss",   color="#F59E0B", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MarginRankingLoss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_FILE, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[RERANKER] Training curves saved to {CURVES_FILE}")

    # --- Save model ---
    save_data = {
        "model_state_dict": model.state_dict(),
        "embedder_name":    EMBEDDER_NAME,
        "embed_dim":        EMBED_DIM,
        "proj_dim":         PROJ_DIM,
        "num_heads":        NUM_HEADS,
        "dropout":          DROPOUT,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_data, f)
    print(f"[RERANKER] Model saved to {MODEL_FILE}")
    print("[RERANKER] Training complete ✓")


# ---------------------------------------------------------------------------
# Inference: load + rerank
# ---------------------------------------------------------------------------

_reranker_model: Optional[CrossAttentionReranker] = None
_reranker_lock  = __import__("threading").Lock()


def load_reranker() -> CrossAttentionReranker:
    """Load saved reranker model. Returns None if not trained yet."""
    if not MODEL_FILE.exists():
        return None
    with open(MODEL_FILE, "rb") as f:
        save_data = pickle.load(f)
    model = CrossAttentionReranker(
        embed_dim = save_data.get("embed_dim", EMBED_DIM),
        proj_dim  = save_data.get("proj_dim",  PROJ_DIM),
        num_heads = save_data.get("num_heads", NUM_HEADS),
        dropout   = save_data.get("dropout",   DROPOUT),
    )
    model.load_state_dict(save_data["model_state_dict"])
    model.eval()
    return model


def rerank(
    query_emb: np.ndarray,
    chunk_embs: List[np.ndarray],
    chunk_texts: List[str],
    model: CrossAttentionReranker,
) -> Tuple[List[str], List[float]]:
    """
    Reranks retrieved chunks using the cross-attention model.

    Args:
        query_emb:   (384,) numpy embedding of the query
        chunk_embs:  list of (384,) numpy embeddings for each candidate chunk
        chunk_texts: list of raw chunk text strings
        model:       loaded CrossAttentionReranker

    Returns:
        (reranked_texts, scores) sorted by relevance score descending
    """
    if len(chunk_embs) == 0:
        return chunk_texts, []

    q = torch.FloatTensor(query_emb).unsqueeze(0).expand(len(chunk_embs), -1)
    c = torch.FloatTensor(np.array(chunk_embs))

    with torch.no_grad():
        scores = model(q, c).numpy()   # (n_chunks,)

    order  = np.argsort(-scores)
    return [chunk_texts[i] for i in order], [float(scores[i]) for i in order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Demo reranking on sample queries")
    args = parser.parse_args()

    if args.demo:
        from sentence_transformers import SentenceTransformer
        from rag.retriever import retrieve, _get_embedder, _get_collection

        print("[RERANKER] Loading saved reranker ...")
        model = load_reranker()
        if model is None:
            print("No reranker found. Run without --demo first to train.")
            sys.exit(1)

        embedder   = _get_embedder()
        collection = _get_collection()

        queries = [
            "What is the time complexity of binary search?",
            "How does a hash table handle collisions?",
            "Explain the difference between TCP and UDP.",
        ]

        for q in queries:
            print(f"\nQuery: {q}")
            q_emb = embedder.encode([q], show_progress_bar=False)[0]
            q_vec = q_emb.tolist()

            results = collection.query(
                query_embeddings=[q_vec], n_results=5, include=["documents", "embeddings"]
            )
            texts  = results["documents"][0]
            embs   = results["embeddings"][0]

            print("Before reranking:")
            for i, t in enumerate(texts, 1):
                print(f"  {i}. {t[:80]} ...")

            reranked, scores = rerank(q_emb, embs, texts, model)
            print("After reranking:")
            for i, (t, s) in enumerate(zip(reranked, scores), 1):
                print(f"  {i}. [{s:.3f}] {t[:80]} ...")
    else:
        train()
