"""
eval/eval_latency.py
Clutch.ai — Stage-by-Stage Latency Evaluation

Times each stage of the pipeline on 5 sample sentences and prints a
breakdown table. No mic input required — uses hardcoded transcripts.

Stages timed:
    (a) Classifier inference
    (b) Embedding + RAG retrieval
    (c) Groq API call (LLM hint generation)
    Total: a + b + c

Target: total average under 4000ms
NOTE: beam_size=5 adds ~200-400ms vs greedy (beam_size=1) but gives significantly
better transcription accuracy. This is the correct tradeoff for quality.
 (warn if over 5000ms)

Usage:
    python eval/eval_latency.py
"""

import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# Import pipeline stages
from classifier.predict import predict
from rag.retriever import retrieve, verify_collection_exists
from llm.hint_gen import generate_hint

# ---------------------------------------------------------------------------
# 5 test sentences
# ---------------------------------------------------------------------------
TEST_SENTENCES: List[str] = [
    "What is the difference between a stack and a queue?",
    "Explain how a hash table works and what happens during a collision.",
    "What is dynamic programming and when would you use it?",
    "How does TCP ensure reliable delivery of packets?",
    "What is the time complexity of quicksort in the average case?",
]

# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def time_stage(fn, *args) -> tuple:
    """Runs a function, returns (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


def run_latency_benchmark() -> None:
    has_chroma = verify_collection_exists()
    if not has_chroma:
        print("[LATENCY] ⚠️  ChromaDB collection not found. Retrieval times will be 0ms.")
        print("[LATENCY]    Run python ingest/ingest.py for accurate retrieval latency.\n")

    print("=" * 70)
    print("Clutch.ai — Pipeline Latency Evaluation")
    print("=" * 70)
    print(f"Testing {len(TEST_SENTENCES)} sentences ...\n")

    # Storage for all stage times
    all_classify: List[float] = []
    all_retrieve: List[float] = []
    all_llm:      List[float] = []
    all_total:    List[float] = []

    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"[{i}/{len(TEST_SENTENCES)}] '{sentence[:60]}...'")

        # (a) Classifier inference
        (label, confidence), t_classify = time_stage(predict, sentence)
        all_classify.append(t_classify)
        print(f"    (a) Classify: {t_classify:.0f}ms → {label} ({confidence:.1f}%)")


        # (b) Embedding + retrieval
        if has_chroma:
            chunks, t_retrieve = time_stage(retrieve, sentence, 3)
        else:
            chunks, t_retrieve = [], 0.0
        all_retrieve.append(t_retrieve)
        print(f"    (b) Retrieve: {t_retrieve:.0f}ms ({len(chunks)} chunks)")

        # (c) LLM hint generation (only for technical questions to save API quota)
        if label == "technical_question":
            hint, t_llm = time_stage(generate_hint, sentence, chunks)
            all_llm.append(t_llm)
            print(f"    (c) LLM:      {t_llm:.0f}ms")
        else:
            t_llm = 0.0
            all_llm.append(t_llm)
            print(f"    (c) LLM:      skipped (not technical question)")

        total = t_classify + t_retrieve + t_llm
        all_total.append(total)
        print(f"    TOTAL: {total:.0f}ms\n")

    # --------------- Summary Table ---------------
    def stats(times: List[float]) -> tuple:
        valid = [t for t in times if t > 0]
        if not valid:
            return 0, 0, 0
        return min(valid), max(valid), sum(valid) / len(valid)

    c_min, c_max, c_avg = stats(all_classify)
    r_min, r_max, r_avg = stats(all_retrieve)
    l_min, l_max, l_avg = stats(all_llm)
    t_min, t_max, t_avg = stats(all_total)

    print("=" * 70)
    print("Stage Latency Summary")
    print("=" * 70)
    print(f"{'Stage':<28} {'Min (ms)':>10} {'Max (ms)':>10} {'Avg (ms)':>10}")
    print("-" * 70)
    print(f"{'(a) Classifier inference':<28} {c_min:>10.0f} {c_max:>10.0f} {c_avg:>10.0f}")
    print(f"{'(b) Embedding + Retrieval':<28} {r_min:>10.0f} {r_max:>10.0f} {r_avg:>10.0f}")
    print(f"{'(c) Groq LLM call':<28} {l_min:>10.0f} {l_max:>10.0f} {l_avg:>10.0f}")
    print("-" * 70)
    print(f"{'TOTAL end-to-end':<28} {t_min:>10.0f} {t_max:>10.0f} {t_avg:>10.0f}")
    print("=" * 70)

    # Pass/Fail
    target_ms = 4000
    warn_ms   = 5000
    if t_avg <= target_ms:
        print(f"\n✓ PASS — Average total latency {t_avg:.0f}ms is under {target_ms}ms target.")
    elif t_avg <= warn_ms:
        print(f"\n⚠️  WARN — Average total latency {t_avg:.0f}ms is over {target_ms}ms but under {warn_ms}ms.")
        print("   Consider reducing AUDIO_CHUNK_SECONDS to 4 in .env if latency is too high.")
    else:
        print(f"\n✗ FAIL — Average total latency {t_avg:.0f}ms exceeds {warn_ms}ms warning threshold.")
        print("   Check your internet connection for Groq latency.")
        print("   Consider using Ollama as primary if on slow network.")


if __name__ == "__main__":
    run_latency_benchmark()
