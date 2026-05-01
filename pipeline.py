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

MIN_CLASSIFIER_CONFIDENCE = 88.0   # confidence floor for all question types

def on_transcript(text: str) -> None:
    """
    Pipeline:
        1. Classify → technical | personal_behavioral | noise
        2. noise → skip
        3. technical    → retrieve CS notes   → bullet hint
        4. personal_behavioral → retrieve resume → paragraph hint
        5. Emit to overlay
    """
    try:
        label, confidence = predict(text)
        print(f"[CLASSIFY] Label: {label} ({confidence:.1f}%) | '{text[:80]}'")

        # Skip noise entirely
        if label == "noise":
            print(f"[CLASSIFY] Skipping (noise).")
            return

        # Confidence gate
        if confidence < MIN_CLASSIFIER_CONFIDENCE:
            print(f"[CLASSIFY] Skipping (confidence {confidence:.1f}% < {MIN_CLASSIFIER_CONFIDENCE}%).")
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
                print(f"[RETRIEVE] Got {len(chunks)} chunk(s).")
            except RuntimeError as e:
                print(f"[RETRIEVE] Warning: {e}. Proceeding without context.")
                chunks = []

        # Generate hint with question_type so prompt format is correct
        hint = generate_hint(text, chunks, question_type=label)
        print(f"[HINT] Generated:\n{hint}")

        overlay.hint_signal.emit(hint)
        print("[UI] Hint signal emitted.")

    except Exception as e:
        print(f"[PIPELINE] Error in on_transcript: {e}")






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

    # Start audio capture in a daemon thread
    print("\n[PIPELINE] Starting audio capture thread ...")
    audio_thread = threading.Thread(
        target=start_capture,
        args=(on_transcript, stop_event),
        daemon=True,
        name="AudioCaptureThread",
    )
    audio_thread.start()
    print("[PIPELINE] Audio thread started ✓")
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
