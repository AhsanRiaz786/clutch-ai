"""
llm/hint_gen.py
Clutch.ai — LLM Hint Generation

Supports two question types with distinct prompts and response formats:

    technical        → 4-line adaptive format (definition / comparison /
                        algorithm / system-design — detected automatically)
    personal_behavioral → 3-part spoken response scaffold tailored to Ahsan

Streaming support: pass on_chunk(partial_text) callback for live updates.

Model: llama-3.3-70b-versatile on Groq (fast, high-quality)
Fallback: Ollama llama3.2:3b for offline use
"""

import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MAX_TOKENS   = 180
TEMPERATURE  = 0.2

# ---------------------------------------------------------------------------
# Ahsan's static bio — fallback when ChromaDB resume chunks unavailable
# ---------------------------------------------------------------------------
AHSAN_STATIC_BIO = """
Name: Ahsan Riaz | NUST SEECS, BS Computer Science, Spring 2026, Islamabad Pakistan
Skills: Python, PyTorch, deep learning, computer vision, NLP, C++, JavaScript, SQL, REST APIs, Git
Projects: Clutch.ai (real-time interview AI, PyQt5 overlay, RAG pipeline, Whisper ASR),
          ML/AI coursework projects, software engineering work
Interests: AI systems, real-time applications, systems programming, LLM tooling
"""

_RESUME_CACHE: str = ""

def _load_resume_context() -> str:
    global _RESUME_CACHE
    if _RESUME_CACHE:
        return _RESUME_CACHE
    try:
        from rag.retriever import retrieve_resume
        chunks = retrieve_resume("experience projects skills background education", k=6)
        if chunks:
            _RESUME_CACHE = "\n\n".join(chunks)
            return _RESUME_CACHE
    except Exception:
        pass
    _RESUME_CACHE = AHSAN_STATIC_BIO.strip()
    return _RESUME_CACHE


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _technical_prompt(question: str, context_chunks: List[str]) -> tuple:
    """
    Adaptive 4-line format that adjusts to the question type.
    The model detects whether the question is a definition, comparison,
    algorithm walkthrough, system design, or tradeoff question —
    and uses the appropriate template.

    Complexity (Big-O) is only included when the question is explicitly
    about an algorithm, data structure, or complexity itself.
    """
    context = "\n\n".join(context_chunks) if context_chunks else "No additional context."

    system = """\
You are a real-time CS interview assistant. Output exactly 4 lines.

First, silently identify the question type. Then apply the matching template:

DEFINITION ("What is X?", "Explain X"):
  Line 1: [Term]: [precise one-clause definition distinguishing it from related concepts]
  Line 2: → [key property, invariant, or core mechanism]
  Line 3: → [time/space complexity if algorithmic, OR main advantage/trade-off if not]
  Line 4: → [when to use it, common gotcha, or real-world application]

COMPARISON ("Difference between X and Y?", "X vs Y?"):
  Line 1: [X] vs [Y]: [the single most important distinction]
  Line 2: → [X]: [defining characteristic]
  Line 3: → [Y]: [defining characteristic]
  Line 4: → [rule of thumb for when to choose which]

ALGORITHM / HOW-IT-WORKS ("How does X work?", "Walk me through X"):
  Line 1: [Algorithm/Concept]: [one-sentence summary of the mechanism]
  Line 2: → [core step or key insight that makes it work]
  Line 3: → time: O(?) | space: O(?) — be precise
  Line 4: → [best vs worst case, or when it fails / better alternative]

SYSTEM DESIGN ("How would you design X?", "What considerations for X?"):
  Line 1: [System]: [high-level approach in one clause]
  Line 2: → [key component or first design decision]
  Line 3: → [main bottleneck or trade-off to address]
  Line 4: → [how to scale it or make it fault-tolerant]

TRADEOFF / WHEN-TO-USE ("When would you use X?", "Why choose X over Y?"):
  Line 1: [X]: [what it optimises for]
  Line 2: → use when [specific condition]
  Line 3: → avoid when [specific condition]
  Line 4: → [concrete example or real-world scenario]

HARD RULES:
- Output ONLY the 4 lines — no preamble, no labels like "DEFINITION:", no trailing text
- Lines 2-4 start with → and nothing else before it
- Max 15 words per line — scannable at a glance during a live conversation
- Big-O ONLY when the question is about an algorithm, data structure, or explicitly asks about complexity
- Never write "O(n) time" if the correct answer is O(log n) — be accurate
- Use the reference notes for factual grounding; do not hallucinate complexities\
"""

    user = (
        f"Interview question: {question}\n\n"
        f"Reference notes:\n{context}\n\n"
        "4-line answer:"
    )

    return system, user


def _personal_prompt(question: str, resume_context: str) -> tuple:
    """
    Produces a tight spoken-answer scaffold shown on the overlay.
    Each line is a short cue Ahsan reads and expands on verbally.
    """
    system = f"""\
You are a real-time interview coach for Ahsan Riaz (CS student, NUST SEECS Pakistan).
Ahsan's background:

{resume_context}

Output EXACTLY 3 lines — no more, no less:

OPEN:   [≤12 words — the very first sentence to say; confident, first-person]
DETAIL: [≤15 words — ONE specific project/example/action to expand on verbally; for STAR: sketch Situation→Action→Result]
CLOSE:  [≤10 words — how it connects to growth or the role]

Critical rules:
- Each line is a SHORT CUE to read and expand on verbally — not a paragraph
- OPEN must DIRECTLY answer the question — never re-introduce "I'm a student at NUST" unless it IS an intro question
- "Tell me about yourself" → OPEN: "CS student at NUST, building AI systems" | DETAIL: Clutch.ai + ML/RAG pipeline | CLOSE: seeking AI/ML role
- STAR behavioral ("a time you...") → OPEN: 1-sentence situation setup | DETAIL: action taken → result | CLOSE: lesson learned
- Strengths → name the actual strength with one concrete proof point
- Weaknesses → name real weakness + what you're doing about it
- Motivation/goals → genuine specific answer, not "I'm passionate about AI" alone
- ZERO extra words. ZERO preamble. Output ONLY the 3 labeled lines.\
"""

    user = f'Interviewer: "{question}"\n\nCue card:'

    return system, user


# ---------------------------------------------------------------------------
# LLM calls (streaming + non-streaming)
# ---------------------------------------------------------------------------

def _call_groq(
    system: str,
    user: str,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    t0 = time.perf_counter()

    if on_chunk is not None:
        # ── Streaming mode ─────────────────────────────────────────────────
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        accumulated = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated += delta
                on_chunk(accumulated)       # fire with full text so far
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[HINT] Groq stream complete in {elapsed:.0f}ms")
        return accumulated.strip()
    else:
        # ── Non-streaming fallback ─────────────────────────────────────────
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[HINT] Groq ({GROQ_MODEL}) {elapsed:.0f}ms")
        return response.choices[0].message.content.strip()


def _call_ollama(system: str, user: str) -> str:
    import requests
    t0 = time.perf_counter()
    resp = requests.post(
        os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
        json={
            "model":    OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[HINT] Ollama ({OLLAMA_MODEL}) {elapsed:.0f}ms")
    return resp.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hint(
    question: str,
    context_chunks: List[str],
    question_type: str = "technical_question",
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate a hint for the given interview question.

    Args:
        question:       Transcribed interview question text.
        context_chunks: RAG-retrieved note chunks.
        question_type:  'technical_question' | 'personal_behavioral'.
        on_chunk:       Optional callback(partial_text) for streaming.
                        Called repeatedly as tokens arrive.

    Returns:
        Final complete hint string.
    """
    if question_type == "personal_behavioral":
        resume_ctx = _load_resume_context()
        system, user = _personal_prompt(question, resume_ctx)
    else:
        system, user = _technical_prompt(question, context_chunks)

    # Groq first (with streaming if callback provided), then Ollama fallback
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            return _call_groq(system, user, on_chunk=on_chunk)
        except Exception as e:
            print(f"[HINT] Groq failed ({e}), falling back to Ollama ...")

    try:
        # Ollama has no streaming wrapper here — call without on_chunk
        result = _call_ollama(system, user)
        if on_chunk:
            on_chunk(result)   # emit once so overlay still updates
        return result
    except Exception as e:
        print(f"[HINT] Ollama also failed: {e}")
        fallback = (
            "OPEN:   Draw on your relevant experience with this topic.\n"
            "DETAIL: Walk through a specific example you know well.\n"
            "CLOSE:  Tie it back to the core concept."
        ) if question_type == "personal_behavioral" else (
            "Topic: review core definition and key properties\n"
            "→ identify the central trade-off or invariant\n"
            "→ recall time/space complexity if algorithmic\n"
            "→ think of a concrete real-world example"
        )
        if on_chunk:
            on_chunk(fallback)
        return fallback


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("What is a binary search tree?",           "technical_question"),
        ("What is the difference between TCP and UDP?", "technical_question"),
        ("How does merge sort work?",                "technical_question"),
        ("How would you design a URL shortener?",    "technical_question"),
        ("When would you use a hash table over a BST?", "technical_question"),
        ("Tell me about yourself.",                  "personal_behavioral"),
        ("Tell me about a time you faced a difficult challenge.", "personal_behavioral"),
        ("What is your biggest weakness?",           "personal_behavioral"),
    ]

    for q, qtype in tests:
        print(f"\n{'─'*60}")
        print(f"Q [{qtype}]: {q}")
        hint = generate_hint(q, [], question_type=qtype)
        print(hint)
