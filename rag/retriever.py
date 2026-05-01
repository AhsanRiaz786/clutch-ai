"""
rag/retriever.py
Clutch.ai — ChromaDB RAG Retriever

Takes a query string, embeds it with MiniLM-L6-v2, queries the ChromaDB
collection 'clutch_notes', and returns the top-k most similar text chunks.

Usage:
    from rag.retriever import retrieve
    chunks = retrieve("What is a binary search tree?", k=3)
"""

import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./db")
COLLECTION_NAME = "clutch_notes"
EMBEDDER_MODEL  = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
DEFAULT_K       = int(os.getenv("TOP_K_CHUNKS", "3"))

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_embedder   = None
_client     = None
_collection = None   # cached after first verified fetch


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[RETRIEVE] Loading embedder: {EMBEDDER_MODEL} ...")
        _embedder = SentenceTransformer(EMBEDDER_MODEL)
        print("[RETRIEVE] Embedder loaded ✓")
    return _embedder


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_collection():
    """Returns cached collection; loads on first call."""
    global _collection
    if _collection is not None:
        return _collection
    client = _get_client()
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' not found at '{CHROMA_PATH}'. "
            "Run 'python ingest/ingest.py' first to ingest your notes."
        )
    _collection = client.get_collection(COLLECTION_NAME)
    print(f"[RETRIEVE] Collection '{COLLECTION_NAME}' ({_collection.count()} chunks) loaded ✓")
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = DEFAULT_K) -> List[str]:
    """
    Returns the top-k relevant text chunks from ChromaDB for the given query.

    Args:
        query: The question/transcript to find context for.
        k:     Number of top chunks to return (default: 3).

    Returns:
        List of strings (the retrieved document chunks).

    Raises:
        RuntimeError: If the ChromaDB collection is not found (ingest.py not run).
    """
    embedder   = _get_embedder()
    collection = _get_collection()

    # Embed query
    query_vec = embedder.encode([query], show_progress_bar=False).tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(k, collection.count()),
    )

    # Extract the document texts
    chunks = results["documents"][0] if results["documents"] else []
    return chunks


def verify_collection_exists() -> bool:
    """Returns True if the ChromaDB collection exists and has documents."""
    try:
        client = _get_client()
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in existing:
            return False
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def retrieve_resume(query: str, k: int = 6) -> List[str]:
    """
    Retrieves chunks specifically from Ahsan's resume (filtered by source metadata).
    Used for personal/behavioral questions so the LLM has accurate personal context.

    Falls back to unfiltered retrieve() if no resume chunks exist.
    """
    embedder   = _get_embedder()
    collection = _get_collection()

    query_vec = embedder.encode([query], show_progress_bar=False).tolist()

    # Filter to only resume document chunks by source path
    try:
        results = collection.query(
            query_embeddings=query_vec,
            n_results=min(k, collection.count()),
            where={"source": {"$contains": "Resume"}},
        )
        chunks = results["documents"][0] if results["documents"] else []
        if chunks:
            print(f"[RETRIEVE] Got {len(chunks)} resume chunk(s) for personal query.")
            return chunks
    except Exception as e:
        print(f"[RETRIEVE] Resume filter failed ({e}), using unfiltered fallback.")

    # Fallback: unfiltered search (resume content will still score high for personal Qs)
    return retrieve(query, k=k)





# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is a binary search tree?",
        "How does TCP handle packet loss?",
        "Explain the difference between SQL and NoSQL.",
    ]

    print("\n--- Clutch.ai RAG Retriever Test ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        chunks = retrieve(q, k=3)
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:150].replace("\n", " ")
            print(f"  [{i}] {preview} ...")
