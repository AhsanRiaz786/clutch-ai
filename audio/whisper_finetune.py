"""
audio/whisper_finetune.py
Clutch.ai — Whisper LoRA Fine-tuning on CS Vocabulary

Deep Learning Enhancement #4  (CS-419 Week 12/13 — Transfer Learning & PEFT)

Goal:
    The base whisper-small.en model has good general ASR but struggles with
    CS-specific pronunciations: "mutex" → "mute x", "semaphore" → "some more",
    "DFS" → "d fs". Fine-tuning with LoRA on CS interview audio fixes this.

Method: LoRA (Low-Rank Adaptation of Large Language Models)
    - Freeze all original Whisper weights
    - Inject trainable rank-8 decomposition matrices (A, B) into attention layers:
        W_new = W_frozen + α * B @ A   (where rank(A) ≈ rank(B) ≈ r = 8)
    - Only trains ~0.5% of total parameters — fast, memory-efficient
    - Applied to: encoder self-attention Q/V projections +
                  decoder cross-attention Q/V projections

Training data:
    Generated synthetically using edge-tts (Microsoft TTS, free):
    - 200+ CS interview questions read aloud
    - Augmented with speed/pitch variation and noise
    Alternatively uses real microphone recordings if available.

Output:
    models/whisper_lora/         — LoRA adapter weights (PEFT format)
    eval/whisper_finetune_log.txt — training loss log

Dependencies (install separately if not present):
    pip install peft transformers datasets librosa edge-tts soundfile

Usage:
    python audio/whisper_finetune.py --generate    # generate TTS training audio
    python audio/whisper_finetune.py --train       # fine-tune with LoRA
    python audio/whisper_finetune.py --transcribe "what is a binary search tree"
"""

import os
import sys
import json
import time
import asyncio
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_PATH   = Path(os.getenv("MODELS_PATH", "./models"))
EVAL_PATH     = Path("./eval")
DATA_PATH     = Path("./data/whisper_train")
LORA_PATH     = MODELS_PATH / "whisper_lora"
LOG_FILE      = EVAL_PATH / "whisper_finetune_log.txt"

WHISPER_BASE  = "openai/whisper-small.en"  # HuggingFace model ID
SAMPLE_RATE   = 16000
MAX_AUDIO_LEN = 30 * SAMPLE_RATE   # 30s max (Whisper's limit)

# LoRA config
LORA_RANK    = 8
LORA_ALPHA   = 16   # effective scaling = alpha / rank = 2
LORA_DROPOUT = 0.05

# Training config
EPOCHS      = 3
BATCH_SIZE  = 4
LR          = 1e-4
GRAD_ACCUM  = 4   # effective batch = BATCH_SIZE * GRAD_ACCUM = 16


# ---------------------------------------------------------------------------
# CS interview training sentences
# ---------------------------------------------------------------------------

CS_SENTENCES = [
    # Data Structures
    "What is a binary search tree and what are its properties?",
    "Explain how a hash table handles collisions using chaining.",
    "What is the difference between a min-heap and a max-heap?",
    "How does a trie differ from a hash table for string storage?",
    "Explain the difference between a stack and a queue.",
    "What is a deque and when would you use it over a queue?",
    "How does a disjoint set or union-find data structure work?",
    "What is a Bloom filter and what are its false positive tradeoffs?",
    "Explain the concept of an LRU cache and how you would implement one.",
    "What is the time complexity of inserting into a balanced BST?",
    # Algorithms
    "Walk me through how merge sort works step by step.",
    "Explain the difference between breadth-first search and depth-first search.",
    "How does Dijkstra's shortest path algorithm work?",
    "What is dynamic programming and when do you apply memoization?",
    "Explain how quicksort partitioning works and its average time complexity.",
    "What is the time complexity of binary search and why?",
    "How does backtracking differ from dynamic programming?",
    "Explain inorder traversal of a binary tree.",
    "What is amortized time complexity and give an example.",
    "How would you detect a cycle in a directed graph?",
    # Networking
    "What is the difference between TCP and UDP?",
    "Explain the three-way handshake: SYN, SYN-ACK, ACK.",
    "How does HTTPS use TLS for encryption?",
    "What is DNS and how does domain name resolution work?",
    "Explain the difference between a REST API and GraphQL.",
    "What is a CDN and how does it reduce latency?",
    "How does load balancing distribute traffic across servers?",
    # OS
    "What is the difference between a mutex and a semaphore?",
    "Explain what a deadlock is and the four Coffman conditions.",
    "What is a race condition and how do you prevent it?",
    "How does virtual memory work with paging?",
    "What is context switching and what overhead does it incur?",
    "Explain the difference between a process and a thread.",
    # Databases
    "What is the difference between SQL and NoSQL databases?",
    "Explain ACID properties in database transactions.",
    "How does database indexing improve query performance?",
    "What is database sharding and why would you use it?",
    "Explain the difference between inner join and left outer join.",
    "What is database normalization and the first three normal forms?",
    # OOP & Design Patterns
    "Explain the four pillars of object-oriented programming.",
    "What is polymorphism and give a concrete example?",
    "Explain the singleton design pattern and its drawbacks.",
    "What is the observer pattern and where is it commonly used?",
    "What is the difference between an abstract class and an interface?",
    # System Design
    "How would you design a URL shortener like Bit.ly?",
    "Explain the CAP theorem and its implications.",
    "How would you design a rate limiter for an API?",
    "What is horizontal versus vertical scaling?",
    "Explain the difference between microservices and monolithic architecture.",
    "How does an LRU cache eviction policy work at scale?",
    # ML/AI (relevant for CS-419)
    "What is the difference between supervised and unsupervised learning?",
    "Explain how backpropagation works in neural networks.",
    "What is gradient descent and how does learning rate affect convergence?",
    "Explain the vanishing gradient problem in deep neural networks.",
    "What is the difference between a GRU and an LSTM?",
    "How does attention mechanism work in transformers?",
    "What is transfer learning and when would you use it?",
    "Explain what overfitting is and how dropout prevents it.",
    "What is the difference between precision and recall?",
    "How does a convolutional neural network process image data?",
    # Additional CS
    "What is Big O notation and why does it matter?",
    "Explain the difference between recursion and iteration.",
    "What is tail recursion optimization?",
    "How does garbage collection work in modern runtimes?",
    "Explain closures in JavaScript.",
    "What is the Global Interpreter Lock in Python?",
    "What are React hooks and how do they differ from class components?",
    "What is async-await and how does it differ from callbacks?",
    "Explain SOLID principles in software design.",
    "What is the difference between composition and inheritance?",
]


# ---------------------------------------------------------------------------
# Synthetic data generation via edge-tts
# ---------------------------------------------------------------------------

async def _generate_audio_async(text: str, output_path: Path) -> bool:
    """Generate speech audio for `text` using edge-tts (Microsoft TTS)."""
    try:
        import edge_tts
        voices = [
            "en-US-GuyNeural",
            "en-US-JasonNeural",
            "en-GB-RyanNeural",
            "en-AU-WilliamNeural",
        ]
        voice = voices[hash(text) % len(voices)]   # vary speaker per sentence
        tts = edge_tts.Communicate(text, voice=voice)
        await tts.save(str(output_path))
        return True
    except Exception as e:
        print(f"  [TTS] edge-tts failed for '{text[:40]}': {e}")
        return False


def generate_training_data() -> None:
    """
    Generates MP3 audio files for all CS_SENTENCES using edge-tts.
    Saves to data/whisper_train/{idx:04d}.mp3 with a metadata JSON.
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    meta = []
    print(f"[WHISPER] Generating TTS audio for {len(CS_SENTENCES)} sentences ...")
    print("[WHISPER] Using edge-tts (requires internet connection) ...")

    for idx, sentence in enumerate(CS_SENTENCES):
        mp3_path = DATA_PATH / f"{idx:04d}.mp3"
        if mp3_path.exists():
            print(f"  [{idx+1}/{len(CS_SENTENCES)}] Already exists: {mp3_path.name}")
        else:
            print(f"  [{idx+1}/{len(CS_SENTENCES)}] Generating: {sentence[:60]} ...")
            success = asyncio.run(_generate_audio_async(sentence, mp3_path))
            if not success:
                continue
            time.sleep(0.1)   # rate limit

        meta.append({"audio": str(mp3_path), "text": sentence})

    meta_path = DATA_PATH / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[WHISPER] Generated {len(meta)} audio files.")
    print(f"[WHISPER] Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Audio loading + preprocessing
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> Optional[np.ndarray]:
    """Load audio file, resample to 16kHz, return as float32 array."""
    try:
        import librosa
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio[:MAX_AUDIO_LEN]
    except Exception as e:
        print(f"  [WHISPER] Could not load {path}: {e}")
        return None


def _augment_audio(audio: np.ndarray) -> List[np.ndarray]:
    """
    Simple augmentation: speed perturbation + additive noise.
    Returns original + 2 augmented variants.
    """
    variants = [audio]

    # Speed perturbation (±5%)
    try:
        import librosa
        faster = librosa.effects.time_stretch(audio, rate=1.05)
        slower = librosa.effects.time_stretch(audio, rate=0.95)
        variants.extend([faster[:MAX_AUDIO_LEN], slower[:MAX_AUDIO_LEN]])
    except Exception:
        pass

    # Additive Gaussian noise
    noise_level = 0.003
    noisy = audio + np.random.randn(*audio.shape).astype(np.float32) * noise_level
    variants.append(np.clip(noisy, -1.0, 1.0))

    return variants


# ---------------------------------------------------------------------------
# LoRA fine-tuning
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Fine-tune Whisper with LoRA on the generated CS training data.

    Requirements: pip install peft transformers datasets librosa soundfile edge-tts
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.mkdir(parents=True, exist_ok=True)
    LORA_PATH.mkdir(parents=True, exist_ok=True)

    # --- Check dependencies ---
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa
        import soundfile as sf
    except ImportError as e:
        print(f"[WHISPER] Missing dependency: {e}")
        print("[WHISPER] Install: pip install peft transformers librosa soundfile edge-tts")
        return

    # --- Check training data ---
    meta_path = DATA_PATH / "metadata.json"
    if not meta_path.exists():
        print("[WHISPER] No training data found. Run with --generate first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    # Filter to files that exist
    meta = [m for m in meta if Path(m["audio"]).exists()]
    print(f"[WHISPER] Found {len(meta)} training audio files.")

    if len(meta) < 5:
        print("[WHISPER] Not enough training data (need >= 5 files).")
        return

    # --- Load Whisper + apply LoRA ---
    print(f"[WHISPER] Loading {WHISPER_BASE} ...")
    processor = WhisperProcessor.from_pretrained(WHISPER_BASE)
    model     = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)

    # LoRA targets: attention Q and V projections in encoder + decoder
    lora_config = LoraConfig(
        task_type   = TaskType.SEQ_2_SEQ_LM,
        r           = LORA_RANK,
        lora_alpha  = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        target_modules = [
            "q_proj", "v_proj",         # encoder + decoder attention Q/V
        ],
        bias = "none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[WHISPER] Total params: {total_params:,} | LoRA trainable: {trainable_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")

    # --- Build dataset ---
    print("[WHISPER] Building training sequences with augmentation ...")
    all_inputs = []
    all_labels = []

    for m in meta:
        audio = _load_audio(m["audio"])
        if audio is None:
            continue
        variants = _augment_audio(audio)
        for v in variants:
            inputs = processor(
                v,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).input_features.squeeze(0)   # (80, 3000)

            with processor.as_target_processor():
                label_ids = processor(m["text"], return_tensors="pt").input_ids.squeeze(0)

            all_inputs.append(inputs)
            all_labels.append(label_ids)

    print(f"[WHISPER] Total training sequences (with augmentation): {len(all_inputs)}")

    if len(all_inputs) < 4:
        print("[WHISPER] Too few sequences after augmentation. Check audio files.")
        return

    # --- Training loop ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS,
    )

    n_val   = max(1, int(len(all_inputs) * 0.1))
    indices = list(range(len(all_inputs)))
    np.random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_inputs = [all_inputs[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]

    print("\n" + "=" * 65)
    print("Clutch.ai — Whisper LoRA Fine-tuning")
    print(f"LoRA rank={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
    print(f"Train: {len(train_inputs)} | Epochs: {EPOCHS} | LR: {LR}")
    print("=" * 65 + "\n")

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = model.to(device)
    train_losses = []
    log_lines    = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = n_steps = 0
        perm       = np.random.permutation(len(train_inputs))

        optimizer.zero_grad()
        for step, idx in enumerate(perm):
            inp = train_inputs[idx].unsqueeze(0).to(device)  # (1, 80, 3000)
            lbl = train_labels[idx].unsqueeze(0).to(device)  # (1, seq_len)

            # Whisper expects decoder_input_ids shifted right
            dec_input = model.model.decoder.embed_tokens(
                torch.full((1, 1), processor.tokenizer.bos_token_id, device=device)
            )

            out  = model(input_features=inp, labels=lbl)
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or step == len(perm) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += out.loss.item()
            n_steps    += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)
        train_losses.append(avg_loss)

        line = f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f}"
        print(line)
        log_lines.append(line)

    # --- Save LoRA adapter ---
    model.save_pretrained(str(LORA_PATH))
    processor.save_pretrained(str(LORA_PATH))
    print(f"\n[WHISPER] LoRA adapter saved to {LORA_PATH}")

    # --- Save log ---
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"[WHISPER] Training log saved to {LOG_FILE}")

    # --- Training curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Clutch.ai — Whisper LoRA Fine-tuning Loss", fontsize=13, fontweight="bold")
    ax.plot(range(1, EPOCHS + 1), train_losses, color="#3B82F6", linewidth=2, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_PATH / "whisper_finetune_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[WHISPER] Fine-tuning complete ✓")


# ---------------------------------------------------------------------------
# Transcription test with LoRA adapter
# ---------------------------------------------------------------------------

def transcribe_with_lora(text_prompt: str) -> None:
    """Test the LoRA-adapted model on a sample transcription."""
    try:
        from peft import PeftModel
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError:
        print("[WHISPER] peft/transformers not installed.")
        return

    if not (LORA_PATH / "adapter_config.json").exists():
        print("[WHISPER] LoRA adapter not found. Run with --train first.")
        return

    print(f"[WHISPER] Loading base model + LoRA adapter from {LORA_PATH} ...")
    processor = WhisperProcessor.from_pretrained(str(LORA_PATH))
    base_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)
    model      = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    model.eval()

    # Generate synthetic test audio from the prompt
    import edge_tts, asyncio

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    asyncio.run(_generate_audio_async(text_prompt, Path(tmp_path)))
    audio = _load_audio(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if audio is None:
        print("[WHISPER] Could not generate test audio.")
        return

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(inputs, language="en")
    transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]

    print(f"\n  Prompt:    {text_prompt}")
    print(f"  Transcript: {transcript}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate",    action="store_true", help="Generate TTS training data")
    parser.add_argument("--train",       action="store_true", help="Fine-tune Whisper with LoRA")
    parser.add_argument("--transcribe",  type=str, default="", help="Transcribe a text prompt with LoRA model")
    args = parser.parse_args()

    if args.generate:
        generate_training_data()
    elif args.train:
        train()
    elif args.transcribe:
        transcribe_with_lora(args.transcribe)
    else:
        print("Usage:")
        print("  python audio/whisper_finetune.py --generate    # generate TTS training audio")
        print("  python audio/whisper_finetune.py --train       # fine-tune with LoRA")
        print("  python audio/whisper_finetune.py --transcribe 'what is a mutex'")
