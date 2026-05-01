"""
eval/eval_retrieval.py
Clutch.ai — RAG Retrieval Precision@3 Evaluation

Tests whether the RAG system retrieves relevant context.
Uses 20 pre-written test questions paired with expected keywords.
A "hit" is when the expected keyword appears (case-insensitive) in any
of the 3 returned chunks.

Prerequisites: Run python ingest/ingest.py first with data in data/notes/

Usage:
    python eval/eval_retrieval.py
    Results also saved to eval/retrieval_results.txt
"""

import sys
import io
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from rag.retriever import retrieve

# ---------------------------------------------------------------------------
# Test Cases: 20 (question, expected_keyword) pairs
# ---------------------------------------------------------------------------
TEST_CASES: List[Tuple[str, str]] = [
    ("What is a binary search tree?",                   "node"),
    ("How does TCP handle packet loss?",                 "retransmit"),
    ("Explain how a hash table works.",                  "hash"),
    ("What is the time complexity of binary search?",    "log"),
    ("What is a deadlock in operating systems?",         "deadlock"),
    ("How does garbage collection work in Python?",      "memory"),
    ("Explain the difference between SQL and NoSQL.",    "database"),
    ("What is dynamic programming?",                     "subproblem"),
    ("How does merge sort work?",                        "merge"),
    ("What is the CAP theorem?",                         "consistency"),
    ("What is polymorphism in OOP?",                     "method"),
    ("How does HTTPS work?",                             "encrypt"),
    ("What is a semaphore?",                             "thread"),
    ("Explain how virtual memory works.",                "page"),
    ("What is load balancing?",                          "server"),
    ("What is a closure in JavaScript?",                 "scope"),
    ("What is database normalization?",                  "normal"),
    ("How does a heap work?",                            "heap"),
    ("What is the singleton design pattern?",            "instance"),
    ("Explain the producer-consumer problem.",           "buffer"),
]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    output_lines: List[str] = []

    def log(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    log("=" * 75)
    log("Clutch.ai — RAG Retrieval Evaluation (Precision@3)")
    log("=" * 75)
    log(f"{'#':<3} {'Question':<50} {'Expected Keyword':<20} {'Result':<8}")
    log("-" * 75)

    hits = 0

    for i, (question, keyword) in enumerate(TEST_CASES, 1):
        try:
            chunks = retrieve(question, k=3)
            combined_text = " ".join(chunks).lower()
            hit = keyword.lower() in combined_text

            if hit:
                hits += 1
                result = "HIT  ✓"
            else:
                result = "MISS ✗"

            q_display = (question[:48] + "..") if len(question) > 50 else question
            log(f"{i:<3} {q_display:<50} {keyword:<20} {result}")

        except Exception as e:
            log(f"{i:<3} ERROR: {e}")

    log("-" * 75)
    precision = hits / len(TEST_CASES)
    log(f"\nRetrieval Precision@3: {hits}/{len(TEST_CASES)} ({precision * 100:.1f}%)")

    if precision >= 0.6:
        log(f"✓ PASS — Meets the 12/20 (60%) threshold.")
    else:
        log(f"✗ FAIL — Below 12/20 (60%) threshold.")
        log("  → Add more relevant documents to data/notes/ and re-run ingest.py")

    # Save results to file
    results_path = Path("eval/retrieval_results.txt")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write("\n".join(output_lines))
    log(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_evaluation()
