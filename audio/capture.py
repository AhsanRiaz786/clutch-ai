"""
audio/capture.py
Clutch.ai — Voice Activity Detection (VAD) Capture + Whisper Transcription

Architecture:
    - sounddevice.InputStream callback feeds 32ms audio blocks into a queue
    - Main loop runs a VAD state machine: WAITING → SPEAKING → TRAILING
    - Records until silence is detected → sends the complete utterance to Whisper
    - No more fixed-interval chunking — every question is captured in full

VAD State Machine:
    WAITING  → RMS above SPEECH_THRESHOLD for consecutive frames → SPEAKING
    SPEAKING → accumulate audio; RMS below SILENCE_THRESHOLD for 1.2s → TRAILING
    TRAILING → confirm silence; finalize utterance → transcribe → WAITING
    SPEAKING → if audio exceeds 20s max → force-finalize → WAITING

Model: base.en (upgraded from tiny.en)
    - ~145MB vs 39MB for tiny.en
    - Significantly better word error rate, especially for CS terminology
    - CPU transcription: ~1.5-3s per 5-15s utterance (within latency budget)
"""

import os
import sys
import queue
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Deque

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000
BLOCK_SIZE:  int = 512          # ~32ms per callback block

# Timing (in blocks of BLOCK_SIZE samples)
_BLOCKS_PER_SEC = SAMPLE_RATE / BLOCK_SIZE

# VAD thresholds — overridden at runtime by _calibrate()
# These are conservative defaults; calibration will tune them to your mic.
SPEECH_THRESHOLD:  float = 0.020
SILENCE_THRESHOLD: float = 0.010

# Calibration params
_CALIB_SECONDS      = 1.5    # how long to measure ambient noise at startup
_SILENCE_MULTIPLIER = 1.8    # silence_threshold = noise_rms * this
_SPEECH_MULTIPLIER  = 5.0    # speech_threshold  = noise_rms * this
_MIN_SPEECH_T       = 0.015  # never go below this for speech threshold
_MIN_SILENCE_T      = 0.008  # never go below this for silence threshold

SILENCE_BLOCKS:    int = int(0.9 * _BLOCKS_PER_SEC)   # 0.9s silence → end utterance
MIN_SPEECH_BLOCKS: int = int(0.3 * _BLOCKS_PER_SEC)   # min 0.3s of speech
MAX_SPEECH_BLOCKS: int = int(12.0 * _BLOCKS_PER_SEC)  # force-finalize at 12s (was 20s)
PRE_BUFFER_BLOCKS: int = int(0.35 * _BLOCKS_PER_SEC)  # 0.35s pre-roll before speech

# Model
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base.en")

# Deduplication
DEDUP_SIMILARITY: float = 0.80
DEDUP_HISTORY:    int   = 2

# Min words for a valid transcript (after VAD, questions are usually complete)
MIN_TRANSCRIPT_WORDS: int = 4

# ---------------------------------------------------------------------------
# CS interview initial prompt — primes Whisper with domain vocabulary
# ---------------------------------------------------------------------------
INITIAL_PROMPT = (
    "CS technical interview. Topics: data structures — stack and queue, "
    "binary search tree BST, linked list, min-heap and max-heap, hash table, "
    "graph, trie, deque, priority queue. "
    "Algorithms — binary search, merge sort, quicksort, dynamic programming, "
    "memoization, backtracking, invert a binary tree, inorder traversal, "
    "BFS breadth-first search, DFS depth-first search, Dijkstra's algorithm, "
    "time complexity Big O notation O of n log n. "
    "Networking — TCP versus UDP, three-way handshake SYN SYN-ACK ACK, "
    "HTTPS TLS encryption, REST API, DNS, load balancing, CDN. "
    "Operating systems — mutex lock, semaphore, deadlock, race condition, "
    "virtual memory, paging, context switching, process versus thread. "
    "Databases — SQL versus NoSQL, ACID, normalization, indexing, sharding, "
    "foreign key, primary key, JOIN inner left outer. "
    "OOP — polymorphism, encapsulation, inheritance, abstract class interface, "
    "design patterns singleton factory observer decorator. "
    "Web — React hooks, closures, async await, garbage collection, GIL. "
    "System design — microservices, message queue, cache eviction LRU, "
    "rate limiter, horizontal vertical scaling, distributed system, CAP theorem."
)

# ---------------------------------------------------------------------------
# Post-processing: known Whisper CS mishearings → correct terms
# ---------------------------------------------------------------------------
CS_CORRECTIONS = {
    "stack and a cube":       "stack and a queue",
    "stack and cube":         "stack and queue",
    " a cube":                " a queue",
    "mean heap":              "min-heap",
    "mean heap and max here": "min-heap and max-heap",
    "min heap and max here":  "min-heap and max-heap",
    "react books":            "React hooks",
    "invite a tree":          "invert a binary tree",
    "invite a binary tree":   "invert a binary tree",
    "invite the tree":        "invert the tree",
    "mute x":                 "mutex",
    "mute ex":                "mutex",
    "some more":              "semaphore",
    "import clocks":          "important locks",
    "big o":                  "Big O",
    "d fs":                   "DFS",
    "b fs":                   "BFS",
    "t c p":                  "TCP",
    "u d p":                  "UDP",
}

HALLUCINATIONS = [
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "[blank_audio]",
    "(upbeat music)",
    "[music]",
]

FILLER_TOKENS = {
    "", "um", "uh", "hmm", "hm", "ah", "er", "erm", "mm", "mmm",
    "um hm", "uh huh", "mhm", "huh", "oh",
    "okay", "ok", "right", "sure", "alright", "yeah", "yep",
    "you", "i", "me", "my",
}

# ---------------------------------------------------------------------------
# VAD state machine states
# ---------------------------------------------------------------------------
WAITING  = "waiting"
SPEAKING = "speaking"


# ---------------------------------------------------------------------------
# Whisper singleton
# ---------------------------------------------------------------------------
_whisper_model = None
_model_lock    = threading.Lock()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                print(f"[AUDIO] Loading Whisper model: {WHISPER_MODEL} (CPU, int8) ...")
                _whisper_model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                )
                print(f"[AUDIO] Whisper {WHISPER_MODEL} loaded ✓")
    return _whisper_model


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _normalize(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Normalise to target RMS. Always returns float32."""
    rms = _rms(audio)
    if rms < 1e-6:
        return audio.astype(np.float32)
    gain = float(np.clip(target_rms / rms, 0.1, 10.0))
    return np.clip(audio.astype(np.float64) * gain, -1.0, 1.0).astype(np.float32)


def _apply_cs_corrections(text: str) -> str:
    lower = text.lower()
    for wrong, correct in CS_CORRECTIONS.items():
        if wrong in lower:
            idx = lower.find(wrong)
            text = text[:idx] + correct + text[idx + len(wrong):]
            lower = text.lower()
    return text


def _clean_transcript(text: str) -> str:
    text = text.strip()
    lower = text.lower()
    for artifact in HALLUCINATIONS:
        if lower.startswith(artifact):
            return ""
    return " ".join(text.split())


def _is_valid(text: str) -> bool:
    """Returns True if the transcript is long enough and not a filler."""
    cleaned = text.strip().lower().rstrip(".,!?…")
    if cleaned in FILLER_TOKENS:
        return False
    return len(cleaned.split()) >= MIN_TRANSCRIPT_WORDS


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _transcribe(audio: np.ndarray) -> str:
    """Transcribe a complete utterance captured by the VAD loop."""
    model = _get_whisper_model()

    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=3,                    # 3 = 30% faster than 5, negligible WER δ with initial_prompt
        temperature=0.0,
        patience=1.0,
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=False,
        no_speech_threshold=0.50,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        suppress_blank=True,
        vad_filter=False,               # we run our own VAD
        word_timestamps=False,
    )

    raw = " ".join(seg.text.strip() for seg in segments).strip()
    cleaned = _clean_transcript(raw)
    return _apply_cs_corrections(cleaned)


# ---------------------------------------------------------------------------
# Main VAD capture loop
# ---------------------------------------------------------------------------

def start_capture(callback: Callable[[str], None], stop_event: threading.Event) -> None:
    """
    Runs continuously. Uses sounddevice.InputStream (non-blocking callback) to
    feed 32ms audio blocks into a queue. The main loop runs a VAD state machine
    to detect utterance start and end, then transcribes the complete utterance.

    Args:
        callback:   Called with the transcribed text of each complete utterance.
        stop_event: Set by pipeline to stop cleanly.
    """
    _get_whisper_model()   # pre-load before starting capture

    # ---- Adaptive noise calibration ----
    # Measure actual ambient noise floor and set thresholds relative to it.
    # This fixes environments where background noise > fixed SILENCE_THRESHOLD.
    global SPEECH_THRESHOLD, SILENCE_THRESHOLD
    print(f"[AUDIO] Calibrating noise floor ({_CALIB_SECONDS}s) — stay quiet ...")
    calib_frames = int(_CALIB_SECONDS * SAMPLE_RATE)
    calib_audio = sd.rec(
        calib_frames, samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocking=True,
    ).flatten()
    noise_rms = _rms(calib_audio)
    SILENCE_THRESHOLD = max(noise_rms * _SILENCE_MULTIPLIER, _MIN_SILENCE_T)
    SPEECH_THRESHOLD  = max(noise_rms * _SPEECH_MULTIPLIER,  _MIN_SPEECH_T)
    print(f"[AUDIO] Noise floor: {noise_rms:.4f} RMS")
    print(f"[AUDIO] Thresholds  — speech > {SPEECH_THRESHOLD:.4f} | silence < {SILENCE_THRESHOLD:.4f}")
    print(f"[AUDIO] Silence gate — {SILENCE_BLOCKS} blocks = {SILENCE_BLOCKS/_BLOCKS_PER_SEC:.1f}s | max utterance = {MAX_SPEECH_BLOCKS/_BLOCKS_PER_SEC:.0f}s")

    audio_q: queue.Queue         = queue.Queue()
    recent_transcripts: Deque[str] = deque(maxlen=DEDUP_HISTORY)

    def _audio_callback(indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread, must be fast."""
        if status:
            print(f"[AUDIO] PortAudio: {status}")
        audio_q.put(indata[:, 0].copy())   # mono flatten

    # VAD state
    state:          str        = WAITING
    speech_blocks:  list       = []
    pre_buffer:     deque      = deque(maxlen=PRE_BUFFER_BLOCKS)
    silence_count:  int        = 0
    speech_count:   int        = 0

    def _finalize_utterance():
        """Process accumulated speech_blocks as a complete utterance."""
        nonlocal state, speech_blocks, silence_count, speech_count
        if speech_count >= MIN_SPEECH_BLOCKS:
            audio = np.concatenate(speech_blocks).astype(np.float32)
            audio = _normalize(audio)

            t0 = time.perf_counter()
            transcript = _transcribe(audio)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if not transcript:
                print(f"[AUDIO] Empty transcript ({elapsed_ms:.0f}ms) — skipping.")
            elif not _is_valid(transcript):
                print(f"[AUDIO] ({elapsed_ms:.0f}ms) '{transcript}' — too short/filler, skipping.")
            else:
                # Deduplication
                is_dup = any(_similarity(transcript, prev) > DEDUP_SIMILARITY
                             for prev in recent_transcripts)
                if is_dup:
                    print(f"[AUDIO] ({elapsed_ms:.0f}ms) Duplicate — skipping.")
                else:
                    print(f"[AUDIO] ({elapsed_ms:.0f}ms) '{transcript}'")
                    recent_transcripts.append(transcript)
                    callback(transcript)

        # Reset state
        state         = WAITING
        speech_blocks = []
        silence_count = 0
        speech_count  = 0

    print(f"[AUDIO] VAD capture ready — listening for speech ...")


    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=_audio_callback,
    ):
        while not stop_event.is_set():
            try:
                block = audio_q.get(timeout=0.05)
            except queue.Empty:
                continue

            rms = _rms(block)

            if state == WAITING:
                pre_buffer.append(block)
                if rms >= SPEECH_THRESHOLD:
                    # Speech onset detected
                    state         = SPEAKING
                    speech_blocks = [b.copy() for b in pre_buffer]   # include pre-roll
                    speech_count  = len(speech_blocks)
                    silence_count = 0
                    print(f"[AUDIO] 🎙️  Speech detected (RMS={rms:.3f})")

            elif state == SPEAKING:
                speech_blocks.append(block)
                speech_count += 1

                if rms < SILENCE_THRESHOLD:
                    silence_count += 1
                    if silence_count >= SILENCE_BLOCKS:
                        # Sustained silence → end of utterance
                        dur_s = speech_count / _BLOCKS_PER_SEC
                        print(f"[AUDIO] 🔇  Silence detected — utterance {dur_s:.1f}s, transcribing ...")
                        _finalize_utterance()
                else:
                    silence_count = 0   # reset on any speech

                # Force-finalize if utterance is too long
                if speech_count >= MAX_SPEECH_BLOCKS:
                    print(f"[AUDIO] ⏱️  Max duration reached — force-finalizing ...")
                    _finalize_utterance()

    print("[AUDIO] Capture stopped.")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import signal

    stop = threading.Event()

    def _print(text: str):
        print(f"\n>>> TRANSCRIPT: [{text}]\n")

    def _sig(sig, frame):
        print("\n[AUDIO] Ctrl+C — stopping ...")
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    t = threading.Thread(target=start_capture, args=(_print, stop), daemon=True)
    t.start()
    stop.wait()
