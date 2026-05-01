"""
llm/hint_gen.py
Clutch.ai — LLM Hint Generation

Supports two question types with distinct prompts and response formats:

    technical        → 3-4 concise bullet points covering core concept + detail + application
    personal_behavioral → Natural 2-3 sentence first-person response as Ahsan Riaz

Model: llama-3.3-70b-versatile on Groq (much better quality than 8b-instant)
Fallback: Ollama llama3.2:3b for offline/local use
"""

import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MAX_TOKENS      = 220
TEMPERATURE     = 0.3

# ---------------------------------------------------------------------------
# Ahsan's background — loaded as static context for personal questions.
# This is updated once at import time from the resume chunks in ChromaDB.
# Falls back to inline text if ChromaDB is unavailable.
# ---------------------------------------------------------------------------
_AHSAN_CONTEXT_CACHE: str = ""

AHSAN_STATIC_BIO = """
Name: Ahsan Riaz
University: NUST SEECS (National University of Sciences and Technology)
Degree: BS Computer Science (CS-419 Deep Learning), Spring 2026
Location: Islamabad, Pakistan

Key technical skills: Python, PyTorch, machine learning, deep learning,
computer vision, NLP, C++, JavaScript, SQL, REST APIs, Git.
Experience: ML/AI projects, software engineering, data science.
Interests: AI systems, real-time applications, systems programming.
"""

def _load_resume_context() -> str:
    """Load Ahsan's resume chunks from ChromaDB for richer personal context."""
    global _AHSAN_CONTEXT_CACHE
    if _AHSAN_CONTEXT_CACHE:
        return _AHSAN_CONTEXT_CACHE
    try:
        from rag.retriever import retrieve_resume
        chunks = retrieve_resume("experience projects skills background", k=6)
        if chunks:
            _AHSAN_CONTEXT_CACHE = "\n\n".join(chunks)
            return _AHSAN_CONTEXT_CACHE
    except Exception:
        pass
    _AHSAN_CONTEXT_CACHE = AHSAN_STATIC_BIO.strip()
    return _AHSAN_CONTEXT_CACHE


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _technical_prompt(question: str, context_chunks: List[str]) -> tuple:
    """System + user messages for a technical CS question."""
    context = "\n\n".join(context_chunks) if context_chunks else "No additional context."

    system = (
        "You are a real-time interview assistant helping a software engineering candidate. "
        "Answer technical CS interview questions with concise, accurate bullet points. "
        "Rules:\n"
        "- Exactly 3 bullet points\n"
        "- Each bullet starts with '• '\n"
        "- Each bullet is a single line, max 15 words\n"
        "- Cover: (1) core definition, (2) key detail/distinction, (3) real-world use or complexity\n"
        "- No intro text, no preamble, no trailing summary\n"
        "- Output only the 3 bullet lines"
    )

    user = (
        f"Question: {question}\n\n"
        f"Study context:\n{context}\n\n"
        "Provide 3 bullet point hints:"
    )

    return system, user


def _personal_prompt(question: str, resume_context: str) -> tuple:
    """System + user messages for a personal/behavioral question."""

    system = (
        "You are an interview coach for Ahsan Riaz, a CS student at NUST SEECS Pakistan.\n"
        "Below is Ahsan's background:\n\n"
        f"{resume_context}\n\n"
        "Rules:\n"
        "- Write a natural, first-person response that Ahsan can say verbatim or paraphrase\n"
        "- 2-3 sentences maximum — keep it tight and confident\n"
        "- Sound genuine and conversational, not rehearsed or listy\n"
        "- Reference specific resume details (projects, skills, NUST) when relevant\n"
        "- For behavioral STAR questions: briefly hint at Situation→Action→Result\n"
        "- No bullet points. No preamble. Output ONLY what Ahsan should say.\n"
        "- Use 'I' (first person)"
    )

    user = f"Interviewer says: \"{question}\"\n\nAhsan's response:"

    return system, user


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _call_groq(system: str, user: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    t0 = time.perf_counter()
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
    print(f"[HINT] Groq ({GROQ_MODEL}) response in {elapsed:.0f}ms")
    return response.choices[0].message.content.strip()


def _call_ollama(system: str, user: str) -> str:
    import requests
    t0 = time.perf_counter()
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model":    OLLAMA_MODEL,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[HINT] Ollama ({OLLAMA_MODEL}) response in {elapsed:.0f}ms")
    return resp.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hint(question: str, context_chunks: List[str], question_type: str = "technical") -> str:
    """
    Generates a hint for the given question.

    Args:
        question:       The interview question text.
        context_chunks: Retrieved RAG chunks (CS notes or resume).
        question_type:  'technical' or 'personal_behavioral'.

    Returns:
        Formatted hint string ready for display on the overlay.
    """
    if question_type == "personal_behavioral":
        resume_ctx = _load_resume_context()
        system, user = _personal_prompt(question, resume_ctx)
    else:
        system, user = _technical_prompt(question, context_chunks)

    # Try Groq first, fall back to Ollama
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            return _call_groq(system, user)
        except Exception as e:
            print(f"[HINT] Groq failed ({e}), falling back to Ollama ...")

    try:
        return _call_ollama(system, user)
    except Exception as e:
        print(f"[HINT] Ollama also failed: {e}")
        if question_type == "personal_behavioral":
            return "Speak naturally about your background and relevant experience."
        return "• Review the core concept\n• Consider key trade-offs\n• Think of a real-world example"


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Technical question ---")
    hint = generate_hint(
        "What is the difference between a stack and a queue?",
        ["Stack: LIFO. Queue: FIFO. Both are linear data structures."],
        question_type="technical",
    )
    print(hint)

    print("\n--- Personal question ---")
    hint = generate_hint(
        "Tell me about yourself.",
        [],
        question_type="personal_behavioral",
    )
    print(hint)

    print("\n--- Behavioral question ---")
    hint = generate_hint(
        "Tell me about a challenging project you worked on.",
        [],
        question_type="personal_behavioral",
    )
    print(hint)
