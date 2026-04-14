"""
config.py — Central Configuration for RAG Benchmarking Project
==============================================================

All hyperparameters, model identifiers, and benchmark reference values
are defined here to ensure a single source of truth across the project.

The parameters strictly replicate the experimental setup from:
  Oro, Granata & Ruffolo (2025) — "A Comprehensive Evaluation of
  Embedding Models and LLMs for IR and QA Across English and Italian"
  (Big Data and Cognitive Computing, 9(5), 141)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ──────────────────────────────────────────────────────────────
# Model Configuration  (tuned for RTX 3080 — 10 GB VRAM)
# ──────────────────────────────────────────────────────────────
# Embedding model — intfloat/multilingual-e5-base (local via SentenceTransformers)
# Switched from e5-large (~1.2 GB) to e5-base (~440 MB) to save VRAM.
# This model requires "query: " and "passage: " prefixes for optimal performance.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# LLM — Llama 3.1 8B Instruct (local via HuggingFace Transformers)
# Loaded in 4-bit quantization (~5 GB) so it fits alongside the embedding model.
# Change this to a local filesystem path if the model is stored outside HF cache.
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LLM_LOAD_IN_4BIT = True     # Enable 4-bit quantization via bitsandbytes

# Device for inference
DEVICE = "cuda"

# ──────────────────────────────────────────────────────────────
# Retrieval Parameters (from the paper)
# ──────────────────────────────────────────────────────────────
TOP_K = 10  # Number of retrieved passages per query

# ──────────────────────────────────────────────────────────────
# Generation Parameters (from the paper)
# ──────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 100

# System prompt — exact template from Table 5 of the paper
SYSTEM_PROMPT = (
    "You are a Question Answering system that is rewarded if the response "
    "is short, concise and straight to the point, use the following pieces "
    "of context to answer the question at the end. If the context doesn't "
    "provide the required information simply respond ."
)

# ──────────────────────────────────────────────────────────────
# Sampling & Ingestion Protocol (from the paper)
# ──────────────────────────────────────────────────────────────
SQUAD_SAMPLE_SIZE = 150
SQUAD_SEED = 433

COVIDQA_SAMPLE_SIZE = 124  # All samples — no sub-sampling

NARRATIVEQA_TOTAL = 100  # 50 books + 50 movies
NARRATIVEQA_SEED = 42

# ──────────────────────────────────────────────────────────────
# GMM Chunker Defaults
# ──────────────────────────────────────────────────────────────
GMM_DEFAULTS = {
    "num_clusters": None,           # Auto via Elbow Method
    "probability_threshold": 0.85,
    "soft_assignment_margin": 0.15,
    "max_gap_threshold": 1,
    "semantic_gap_threshold": 0.75,
    "window_expansion_k": 1,
    "max_sentences_per_chunk": 12,
    "max_chunk_words": 300,
    "max_cluster_size": 1500,
}

# ──────────────────────────────────────────────────────────────
# Paper Benchmark Reference Values (Table 9 — Llama 3.1 8B)
# ──────────────────────────────────────────────────────────────
PAPER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "SQuAD-en": {
        "rouge_l": 0.72,
        "f1": 0.69,
    },
    "CovidQA": {
        "rouge_l": 0.22,
        "f1": 0.15,
    },
    "NarrativeQA (Books)": {
        "rouge_l": 0.12,
        "f1": 0.11,
    },
    "NarrativeQA (Movies)": {
        "rouge_l": 0.13,
        "f1": 0.11,
    },
}
