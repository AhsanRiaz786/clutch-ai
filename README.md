# Clutch.ai

**Real-Time Interview Assistance Using Local Transformer Pipelines**  
CS-419 Deep Learning | NUST SEECS | Spring 2026  
Ahsan Riaz (479561) | Farhad Khan (453770) | Furqan Ahmad Basra (462974)

---

## What it does

Clutch.ai listens to your mic during a technical interview, detects when a CS question is asked, retrieves relevant context from your own notes, and flashes a private 3-bullet hint overlay on your screen — all in under 6 seconds, entirely locally.

**6-Stage Pipeline:**
```
Mic Audio (5s chunks)
  → Faster-Whisper ASR (CPU, tiny.en, ~1.5s)
  → MLP Classifier (technical? / small talk / other)
  → MiniLM Embedding → ChromaDB RAG retrieval
  → Groq API (Llama-3.1-8b-instant) → 3-bullet hint
  → PyQt5 frameless transparent overlay (auto-hides 12s)
```

---

## Setup

### 1. Clone & install dependencies
```bash
git clone <repo-url>
cd clutch-ai
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and paste your Groq API key
# Sign up free at https://console.groq.com (no credit card required)
```

### 3. Add your study notes
```
data/notes/    ← Drop your PDF lecture notes and .txt files here
data/code/     ← Drop your .py, .js, .cpp project files here
```

### 4. Ingest notes into ChromaDB
```bash
python ingest/ingest.py
# Wait for: Total chunks stored: N
```

### 5. Train the question classifier
```bash
python classifier/train.py
# Wait for training completion (target accuracy > 85%)
# This creates: models/question_clf.pkl and eval/training_curves.png
```

### 6. Run evaluation scripts (optional but recommended)
```bash
python eval/eval_retrieval.py   # Verify precision@3 ≥ 12/20
python eval/eval_latency.py     # Verify total avg < 5000ms
```

### 7. Start Clutch.ai
```bash
python pipeline.py
# Speak a technical question into your mic
# The overlay will appear in the bottom-right corner within 6 seconds
```

---

## Offline Fallback (Ollama)

If Groq is unavailable (rate limit, no internet), the system falls back to Ollama automatically.

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:3b
ollama serve   # Run in a separate terminal
```

---

## Project Structure

```
clutch-ai/
├── .env.example          # Config template
├── requirements.txt      # All dependencies
├── data/
│   ├── notes/            # Your PDF/txt study materials
│   └── code/             # Your code files
├── ingest/ingest.py      # Ingests data/ into ChromaDB
├── audio/capture.py      # Mic → Whisper transcription
├── classifier/
│   ├── dataset.py        # Training data (200 labeled examples)
│   ├── train.py          # Trains MLP classifier
│   └── predict.py        # Inference wrapper
├── rag/retriever.py      # ChromaDB cosine search
├── llm/hint_gen.py       # Groq + Ollama fallback
├── ui/overlay.py         # PyQt5 transparent overlay
├── eval/
│   ├── eval_retrieval.py # Precision@3 evaluation
│   └── eval_latency.py   # Stage-by-stage latency
└── pipeline.py           # Main entry point
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Audio capture | sounddevice + numpy |
| Transcription | faster-whisper (tiny.en, int8, CPU) |
| Classifier | PyTorch MLP (384→128→64→3) |
| Embeddings | sentence-transformers (MiniLM-L6-v2) |
| Vector DB | ChromaDB (persistent, file-based) |
| Document ingestion | LangChain + pypdf |
| LLM hint generation | Groq API (llama-3.1-8b-instant) |
| Offline LLM fallback | Ollama (llama3.2:3b) |
| UI Overlay | PyQt5 (frameless, transparent, always-on-top) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| ChromaDB collection not found | Run `python ingest/ingest.py` first |
| Whisper too slow (>3s/chunk) | Using tiny.en + int8 is already optimal; reduce chunk to 4s if needed |
| Overlay not on top of Zoom | Qt.WindowDoesNotAcceptFocus is set; test on demo machine |
| Groq rate limit hit | Classifier filters non-technical audio; check that it's working |
| Classifier accuracy < 80% | Add more training examples in `classifier/dataset.py` |
| PyQt5 crashes on import | `pip install PyQt5==5.15.9` |
| Ollama connection refused | Run `ollama serve` in a separate terminal |
| Audio garbled to Whisper | Ensure sounddevice records at exactly 16000 Hz |
