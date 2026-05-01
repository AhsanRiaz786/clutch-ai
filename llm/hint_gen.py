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
    Generates a conversational, natural paragraph response for a technical question.
    """
    context = "\n\n".join(context_chunks) if context_chunks else "No additional context."

    system = """\
You are a real-time CS interview assistant. Your goal is to provide a natural, conversational answer that the candidate can use as a reference or speak aloud.

Analyze the question and provide a highly accurate, concise, and professional answer in a single, flowing paragraph (about 3-4 sentences).

First, silently identify the question type. Then structure your paragraph using the following logic as a guide, but blend it naturally into flowing prose:

- DEFINITION ("What is X?"): Start with a precise definition, state a key property, mention trade-offs/complexity, and give a real-world application.
- COMPARISON ("Difference between X and Y?"): State the most important distinction, define both sides, and give a rule of thumb for when to choose which.
- ALGORITHM ("How does X work?"): Summarize the mechanism, mention the core step/insight, state exact time/space complexity, and note best/worst cases.
- SYSTEM DESIGN ("How would you design X?"): State the high-level approach, point out the key component, address the main bottleneck, and mention scaling.
- TRADEOFF ("When would you use X?"): State what it optimizes for, when to use it, when to avoid it, and give a concrete example.

RULES:
- Do NOT use bullet points, arrows (→), or labeled sections.
- Write as a natural, conversational paragraph that directly answers the question.
- If it's an algorithm, mention time/space complexity naturally in the text.
- Be highly accurate, using the reference notes for factual grounding.
- Keep the language professional yet accessible.
- Keep it concise (under 60 words) so it can be read quickly at a glance.
"""

    user = (
        f"Interview question: {question}\n\n"
        f"Reference notes:\n{context}\n\n"
        "Conversational answer:"
    )

    return system, user


def _personal_prompt(question: str, resume_context: str) -> tuple:
    """
    Produces a natural, first-person conversational response.
    """
    system = f"""\
You are a real-time interview coach for Ahsan Riaz (CS student, NUST SEECS Pakistan).
Ahsan's background:

{resume_context}

Your goal is to write a highly natural, confident, first-person paragraph (3-4 sentences) that Ahsan can use to answer behavioral or personal questions.

Use the following logical flow to structure your paragraph, but blend it naturally into flowing prose:
- OPEN: Directly answer the question in the first sentence.
- DETAIL: Provide a specific example, project, or action from the resume context to back it up (use the STAR method: Situation -> Action -> Result, if applicable).
- CLOSE: Conclude by connecting it to your growth, a lesson learned, or what you're looking for in the role.

RULES:
- Do NOT use bullet points, labels (like OPEN, DETAIL, CLOSE), or scaffolding.
- Write a single, cohesive paragraph that sounds like a real person talking.
- Directly answer the question using specific examples from the resume context where relevant.
- Keep it concise (under 60 words).
- ZERO extra words or preamble. Output ONLY the paragraph.\
"""

    user = f'Interviewer: "{question}"\n\nConversational answer:'

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
            "I'd draw on my experience with similar projects. For instance, I can walk through the specific challenges we faced, the actions I took to resolve them, and how that ties back to the core concept here."
        ) if question_type == "personal_behavioral" else (
            "This concept is primarily about understanding the core mechanism and its main trade-offs. I'd evaluate the specific time and space constraints, and consider the practical real-world scenarios where this approach shines."
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
