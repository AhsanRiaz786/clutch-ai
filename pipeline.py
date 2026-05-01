"""
pipeline.py
Clutch.ai — Main Entry Point

Initializes all modules, starts the audio capture thread, and processes
transcripts in a loop. This is what the user runs to start the system.

Usage:
    python pipeline.py

Keyboard Interrupt (Ctrl+C) cleanly stops the audio thread and exits.
"""

import os
import sys
import signal
import threading
from pathlib import Path

# Must import QApplication before anything else Qt-related
from PyQt5.QtWidgets import QApplication

# Load env variables first
from dotenv import load_dotenv
load_dotenv()

# Project imports
from classifier.predict import predict
from rag.retriever import retrieve, retrieve_resume, verify_collection_exists

from llm.hint_gen import generate_hint
from audio.capture import start_capture
from ui.overlay import HintOverlay

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
overlay: HintOverlay = None
stop_event: threading.Event = threading.Event()


# ---------------------------------------------------------------------------
# Transcript callback — runs in audio worker thread
# ---------------------------------------------------------------------------

MIN_CLASSIFIER_CONFIDENCE = 85.0   # confidence floor

def on_transcript(text: str) -> None:
    """
    Pipeline:
        1. Classify → technical_question | personal_behavioral | noise
        2. noise → skip
        3. confidence < floor → skip
        4. technical_question → CS notes retrieval → structured 4-line hint (streamed)
        5. personal_behavioral → resume retrieval → OPEN/DETAIL/CLOSE scaffold (streamed)
        6. Stream tokens to overlay in real time; emit final formatted hint when done
    """
    try:
        label, confidence = predict(text)
        print(f"[CLASSIFY] {label} ({confidence:.1f}%) | '{text[:80]}'")

        # Noise = background speech, filler, candidate's own answers
        if label == "noise":
            print("[CLASSIFY] Skipping — noise.")
            return

        # Confidence gate
        if confidence < MIN_CLASSIFIER_CONFIDENCE:
            print(f"[CLASSIFY] Skipping — confidence {confidence:.1f}% < {MIN_CLASSIFIER_CONFIDENCE}%.")
            return

        # Route retrieval by question type
        if label == "personal_behavioral":
            try:
                chunks = retrieve_resume(text, k=5)
            except Exception as e:
                print(f"[RETRIEVE] Warning: {e}")
                chunks = []
        else:
            try:
                chunks = retrieve(text, k=3)
                print(f"[RETRIEVE] {len(chunks)} chunk(s).")
            except RuntimeError as e:
                print(f"[RETRIEVE] Warning: {e}. Proceeding without context.")
                chunks = []

        # Stream tokens to overlay as they arrive, then emit final formatted hint
        def _stream_chunk(partial: str):
            overlay.stream_signal.emit(partial)

        hint = generate_hint(text, chunks, question_type=label, on_chunk=_stream_chunk)
        print(f"[HINT]\n{hint}")

        # Final emit: switches overlay from plain streaming text to formatted HTML
        overlay.hint_signal.emit(hint)

    except Exception as e:
        print(f"[PIPELINE] Error: {e}")






# ---------------------------------------------------------------------------
# Signal handler for clean Ctrl+C shutdown
# ---------------------------------------------------------------------------

def _handle_sigint(sig, frame):
    print("\n[PIPELINE] Ctrl+C received. Shutting down ...")
    stop_event.set()
    QApplication.quit()


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def _check_prerequisites() -> bool:
    """
    Verifies that:
    - GROQ_API_KEY is set
    - models/question_clf.pkl exists
    - ChromaDB collection exists
    Returns True if all checks pass, False otherwise.
    """
    all_ok = True

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("[PIPELINE] ⚠️  WARNING: GROQ_API_KEY not set. Will use Ollama fallback.")

    model_file = Path(os.getenv("MODELS_PATH", "./models")) / "question_clf.pkl"
    if not model_file.exists():
        print(f"[PIPELINE] ❌ ERROR: Classifier model not found at {model_file}.")
        print("[PIPELINE]    Run: python classifier/train.py")
        all_ok = False

    if not verify_collection_exists():
        print("[PIPELINE] ⚠️  WARNING: ChromaDB collection 'clutch_notes' not found.")
        print("[PIPELINE]    The system will work but without note context.")
        print("[PIPELINE]    Run: python ingest/ingest.py  (after dropping files in data/)")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global overlay

    print("=" * 65)
    print("  ⚡ Clutch.ai — Real-Time Interview Assistance")
    print("  CS-419 Deep Learning | NUST SEECS | Spring 2026")
    print("=" * 65)
    print("  [PRO TIP] To make the overlay 100% invisible to the interviewer,")
    print("            share a specific Window (e.g. Chrome, VSCode) in Zoom/")
    print("            Google Meet instead of sharing your 'Entire Screen'.")
    print("            The overlay will float on top of your screen but will")
    print("            not be captured by the meeting software.")
    print("=" * 65)

    # Init Qt application
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Create overlay
    overlay = HintOverlay()

    # Show ready message briefly
    overlay.show_hint("⚡ Clutch.ai is ready.\nSpeak a technical question ...")

    # Run prerequisite checks
    print("\n[PIPELINE] Checking prerequisites ...")
    if not _check_prerequisites():
        print("[PIPELINE] ❌ Cannot start — fix errors above first.")
        sys.exit(1)
    print("[PIPELINE] Prerequisites OK ✓")

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Parallel model pre-warming ──────────────────────────────────────────
    # Audio thread loads Whisper (~3-5s). Meanwhile we load the classifier,
    # MiniLM embedder, and ChromaDB in a second thread so everything is hot
    # by the time the first question is transcribed.

    def _prewarm():
        import time
        t0 = time.perf_counter()
        print("[PREWARM] Loading classifier + embedder ...")
        try:
            predict("binary search tree")          # warms classifier + its MiniLM copy
        except Exception as e:
            print(f"[PREWARM] Classifier warm-up failed: {e}")

        print("[PREWARM] Loading RAG embedder + ChromaDB ...")
        try:
            retrieve("binary search tree", k=1)    # warms RAG embedder + ChromaDB
        except Exception as e:
            print(f"[PREWARM] RAG warm-up failed (collection may not exist yet): {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[PREWARM] All models hot in {elapsed:.0f}ms ✓")

    prewarm_thread = threading.Thread(target=_prewarm, daemon=True, name="PrewarmThread")

    # Start audio capture in a daemon thread
    print("\n[PIPELINE] Starting audio capture + model pre-warming ...")
    audio_thread = threading.Thread(
        target=start_capture,
        args=(on_transcript, stop_event),
        daemon=True,
        name="AudioCaptureThread",
    )
    audio_thread.start()
    prewarm_thread.start()
    print("[PIPELINE] Audio + prewarm threads started ✓")
    print("[PIPELINE] Listening for technical questions ... (Ctrl+C to stop)\n")

    # Enter Qt event loop (blocks until quit)
    exit_code = app.exec_()

    # Clean shutdown
    stop_event.set()
    audio_thread.join(timeout=5)
    print("[PIPELINE] Clutch.ai stopped.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
