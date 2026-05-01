# ⚡ Clutch.ai

**Real-Time, Undetectable Interview Assistance Powered by Local Deep Learning**  
**Repository:** [https://github.com/AhsanRiaz786/clutch-ai](https://github.com/AhsanRiaz786/clutch-ai)  
**CS-419 Deep Learning | NUST SEECS | Spring 2026**  
**Team:** Ahsan Riaz (479561) | Farhad Khan (453770) | Furqan Ahmad Basra (462974)

---

## 🌟 Overview

Clutch.ai is a production-grade, real-time interview co-pilot. It seamlessly listens to interview audio, intelligently classifies question types (Technical vs. Behavioral vs. Noise), dynamically retrieves the perfect context from your personal codebase and resume, and streams tailored hints to an invisible overlay—all in under a few seconds.

Built to exceed standard course requirements, Clutch.ai features a highly optimized, custom deep learning pipeline incorporating sequence models, contrastive representation learning, cross-attention mechanisms, and parameter-efficient fine-tuning (LoRA).

---

## 🧠 Deep Learning Architecture (CS-419 Enhancements)

This project implements **5 custom deep learning modules** built from scratch using PyTorch:

1. **BiLSTM + Attention Classifier:** Replaced the baseline MLP with a Bidirectional LSTM incorporating self-attention pooling. Trained on 900+ labeled examples to distinguish technical questions, behavioral questions, and background noise with **>96% accuracy**.
2. **Contrastive Embedding Fine-Tuning:** The base `all-MiniLM-L6-v2` model is fine-tuned using `TripletMarginWithDistanceLoss` (Margin-based Contrastive Learning) to pull semantically related CS topics closer together in vector space, drastically improving RAG retrieval precision.
3. **GRU-based Voice Activity Detection (VAD):** A sequence model that processes 60-feature MFCC audio inputs to precisely determine when the interviewer is speaking vs. background silence.
4. **Cross-Attention Reranker:** A custom interaction module that reranks retrieved RAG candidate chunks by computing cross-attention scores against the interviewer's query.
5. **Whisper LoRA Fine-Tuning:** Parameter-efficient fine-tuning (Rank 8) applied to the Distil-Whisper encoder-decoder architecture, utilizing synthetic data generation for domain-specific vocabulary adaptation.

---

## ⚙️ The Real-Time Pipeline

```text
Mic Audio stream
  ↳ GRU VAD (Silence/Noise filtering)
    ↳ Faster-Whisper ASR (Int8, CPU)
      ↳ BiLSTM Question Classifier (Technical / Behavioral / Noise)
        ↳ Triplet-Tuned Embedder → ChromaDB RAG
          ↳ Cross-Attention Reranker (Top-K Selection)
            ↳ Groq API (Llama-3.1-8b) → Context-Aware Prompting
              ↳ PyQt5 Stealth Overlay (Streams tokens in real-time)
```

> **Pro Tip:** The UI overlay uses `Qt.FramelessWindowHint` and `Qt.WindowStaysOnTopHint`. It is 100% invisible to the interviewer if you share a *Specific Window* (e.g., Chrome, VSCode) instead of your full desktop in Zoom/Google Meet.

---

## 🚀 Setup & Installation

### 1. Clone & Install
```bash
git clone https://github.com/AhsanRiaz786/clutch-ai.git
cd clutch-ai
pip install -r requirements.txt
```

### 2. Environment Variables
```bash
cp .env.example .env
# Edit .env and paste your Groq API key (https://console.groq.com)
```

### 3. Add Your Data
Drop your personal context files into the `data/` directory so the RAG pipeline knows about you:
- `data/notes/` → PDF lecture notes, technical guides.
- `data/code/` → Your `.py`, `.js`, `.cpp` project files.
- `data/resume/` → Your resume/CV for behavioral questions.

### 4. Build the Vector Database & Train Models
Clutch.ai uses a dynamic dataset (`data.csv` containing ~525 interview questions). Run the following commands to initialize the system:

```bash
# Ingest notes and resume into ChromaDB
python ingest/ingest.py

# Train the baseline MLP classifier (optional)
python classifier/train.py

# Train the primary BiLSTM + Attention Classifier
python classifier/lstm_classifier.py

# Fine-tune the MiniLM embeddings using Triplet Loss
python classifier/finetune_embeddings.py
```

### 5. Launch the Assistant
```bash
python pipeline.py
```
*Note: A background daemon thread will automatically pre-warm the BiLSTM, Embedder, and ChromaDB upon startup, guaranteeing zero-latency on the first question.*

---

## 📊 Evaluation & Metrics
Run the evaluation scripts to verify pipeline health and generate figures for the final report:
```bash
python eval/eval_retrieval.py   # RAG Precision@K metrics
python eval/eval_latency.py     # End-to-end latency breakdown
```

## 🛡️ Fallback Systems
If Groq rate limits are hit or the internet connection drops, Clutch.ai automatically falls back to a local Ollama instance (`llama3.2:3b`). To enable:
```bash
ollama pull llama3.2:3b
ollama serve
```
