"""
ingest/ingest.py
Clutch.ai — Document Ingestion Pipeline

Reads every file inside data/notes/ and data/code/, splits into chunks,
embeds each chunk using MiniLM-L6-v2, and stores all embeddings + raw text
in ChromaDB under a collection named 'clutch_notes'.

Run once before the first demo. Re-run whenever new notes are added.
Usage:
    python ingest/ingest.py
"""

import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "./db")
DATA_PATH   = os.getenv("DATA_PATH",   "./data")
COLLECTION_NAME = "clutch_notes"
EMBEDDER_MODEL  = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "text",
    ".py":  "text",
    ".js":  "text",
    ".cpp": "text",
    ".java":"text",
    ".md":  "text",
    ".ts":  "text",
    ".c":   "text",
    ".h":   "text",
}

# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: str) -> List[Document]:
    """
    Walks data/notes/ and data/code/, returns list of LangChain Document objects.
    Supports .pdf, .txt, .py, .js, .cpp, .java, .md and more.
    """
    data_path = Path(data_dir)
    all_docs: List[Document] = []

    subdirs = ["notes", "code"]

    for subdir in subdirs:
        folder = data_path / subdir
        if not folder.exists():
            print(f"[INGEST] Warning: {folder} does not exist, skipping.")
            continue

        for file_path in sorted(folder.rglob("*")):
            if file_path.is_dir():
                continue
            if file_path.name.startswith("."):
                continue  # skip .gitkeep etc.

            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"[INGEST] Skipping unsupported file type: {file_path.name}")
                continue

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                else:
                    loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)

                docs = loader.load()

                # Tag each document with source metadata
                for doc in docs:
                    doc.metadata["source"] = file_path.name

                all_docs.extend(docs)
                print(f"[INGEST] Loaded: {file_path.name} ({len(docs)} page(s)/section(s))")

            except Exception as e:
                print(f"[INGEST] Error loading {file_path.name}: {e}")

    return all_docs


# ---------------------------------------------------------------------------
# Embed & Store
# ---------------------------------------------------------------------------

def embed_and_store(docs: List[Document], collection_name: str = COLLECTION_NAME) -> None:
    """
    Chunks all documents, embeds with MiniLM-L6-v2, upserts into ChromaDB.
    Deletes and recreates the collection on each run for a fresh ingest.
    """
    if not docs:
        print("[INGEST] No documents found to ingest. Drop files into data/notes/ or data/code/ first.")
        return

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Total chunks after splitting: {len(chunks)}")

    # --- Embedder ---
    print(f"[INGEST] Loading embedder: {EMBEDDER_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    # --- ChromaDB ---
    print(f"[INGEST] Connecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete and recreate for fresh ingest
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"[INGEST] Deleted existing collection '{collection_name}'.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[INGEST] Created collection '{collection_name}'.")

    # --- Batch embed and upsert ---
    texts      = [c.page_content for c in chunks]
    metadatas  = []
    ids        = []

    # Build per-source chunk index counters
    source_counters: dict = {}
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(source, 0)
        source_counters[source] = idx + 1
        metadatas.append({"source": source, "chunk_index": idx})
        ids.append(f"{source}__chunk{idx}")

    print(f"[INGEST] Embedding {len(texts)} chunks (this may take a minute) ...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    # ChromaDB upsert in batches of 500 to avoid memory issues
    BATCH = 500
    for start in range(0, len(texts), BATCH):
        end = min(start + BATCH, len(texts))
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    total = collection.count()
    print(f"[INGEST] ✓ Total chunks stored: {total}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Clutch.ai — Document Ingestion")
    print("=" * 60)
    docs = load_documents(DATA_PATH)
    if docs:
        print(f"[INGEST] Loaded {len(docs)} document section(s) total.")
    embed_and_store(docs)
    print("[INGEST] Done.")
