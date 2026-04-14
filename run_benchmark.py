"""
run_benchmark.py — Main Orchestrator for RAG Benchmarking
=========================================================

End-to-end pipeline that:
  1. Loads the embedding model (intfloat/multilingual-e5-large)
  2. Initializes the GMM Chunker
  3. Initializes the Llama 3.1 8B Instruct generator
  4. For each dataset (SQuAD, CovidQA, NarrativeQA):
     a. Load data via the appropriate loader
     b. Ingest documents (chunk → embed → index into ChromaDB)
     c. For each evaluation question: retrieve → generate → score
     d. Aggregate metrics
  5. Generate final comparison report vs. paper benchmarks

Usage:
    # Full benchmark (all datasets)
    python run_benchmark.py

    # Specific datasets only
    python run_benchmark.py --datasets squad covidqa

    # Quick dry-run (5 samples per dataset)
    python run_benchmark.py --dry-run

    # Skip ingestion (reuse existing ChromaDB index)
    python run_benchmark.py --skip-ingestion
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Project imports ─────────────────────────────────────────
from config import (
    EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, DEVICE,
    TOP_K, MAX_NEW_TOKENS, SYSTEM_PROMPT,
    CHROMA_PERSIST_DIR, RESULTS_DIR,
    GMM_DEFAULTS,
)
from loaders import SquadLoader, CovidQALoader, NQLoader
from chunking import GmmChunker
from pipeline.ingestion import ingest_dataset
from pipeline.retrieval import retrieve
from pipeline.generation import LlamaGenerator
from evaluation.metrics import compute_rouge_l, compute_token_f1, compute_dataset_metrics
from evaluation.report import generate_report

# ── Logging setup ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Dataset Registry
# ──────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "squad": {
        "loader_class": SquadLoader,
        "loader_kwargs": {},
        "collection_name": "squad",
        "report_name": "SQuAD-en",
    },
    "covidqa": {
        "loader_class": CovidQALoader,
        "loader_kwargs": {},
        "collection_name": "covidqa",
        "report_name": "CovidQA",
    },
    "narrativeqa": {
        "loader_class": NQLoader,
        "loader_kwargs": {},
        "collection_name": "narrativeqa",
        "report_name": None,  # Split into Books/Movies at reporting time
    },
}


def evaluate_dataset(
        eval_set: List[Dict],
        generator: LlamaGenerator,
        embedding_model: SentenceTransformer,
        chroma_collection,
        dry_run: bool = False,
) -> List[Dict]:
    """
    Run the full retrieve → generate → evaluate loop for one dataset.

    Returns a list of per-question result dicts with keys:
      question, ground_truth, prediction, rouge_l, f1
    """
    results = []
    items = eval_set[:5] if dry_run else eval_set

    for item in tqdm(items, desc="Evaluating"):
        question = item["question"]
        ground_truth = item["ground_truth"]

        # ── Retrieve ──────────────────────────────────────────
        passages = retrieve(
            query=question,
            embedding_model=embedding_model,
            chroma_collection=chroma_collection,
            top_k=TOP_K,
        )

        # ── Generate ─────────────────────────────────────────
        prediction = generator.generate(
            question=question,
            context_passages=passages,
            system_prompt=SYSTEM_PROMPT,
        )

        # ── Score ─────────────────────────────────────────────
        rouge_l = compute_rouge_l(prediction, ground_truth)
        f1 = compute_token_f1(prediction, ground_truth)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "rouge_l": rouge_l,
            "f1": f1,
            # Carry forward any extra fields (like 'kind' for NarrativeQA)
            **{k: v for k, v in item.items()
               if k not in ("question", "ground_truth")},
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmark: GMM Chunking vs. Paper Baselines"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=list(DATASET_REGISTRY.keys()),
        help="Which datasets to benchmark (default: all).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run on only 5 samples per dataset for quick testing.",
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip ingestion and reuse existing ChromaDB collections.",
    )
    args = parser.parse_args()

    start_time = time.time()
    print("\n" + "=" * 70)
    print("  RAG BENCHMARKING — GMM Semantic Chunking Evaluation")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────
    # Step 1: Load Embedding Model
    # ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    print(f"       ✓ Embedding model loaded on {DEVICE}.")

    # ──────────────────────────────────────────────────────────
    # Step 2: Initialize GMM Chunker
    # ──────────────────────────────────────────────────────────
    print(f"\n[2/4] Initializing GMM Chunker...")
    chunker = GmmChunker(
        embedding_model=embedding_model,
        **GMM_DEFAULTS,
    )
    print(f"       ✓ GMM Chunker ready.")

    # ──────────────────────────────────────────────────────────
    # Step 3: Initialize LLM Generator
    # ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Loading LLM: {LLM_MODEL_NAME}")
    generator = LlamaGenerator(
        model_name=LLM_MODEL_NAME,
        device=DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ──────────────────────────────────────────────────────────
    # Step 4: Initialize ChromaDB
    # ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    print(f"       ✓ ChromaDB ready.")

    # ──────────────────────────────────────────────────────────
    # Run Benchmarks
    # ──────────────────────────────────────────────────────────
    all_our_results = {}       # {report_name: {rouge_l, f1}}
    all_detailed_results = {}  # {dataset_name: [per-question results]}

    for ds_key in args.datasets:
        ds_info = DATASET_REGISTRY[ds_key]
        print(f"\n{'─' * 70}")
        print(f"  DATASET: {ds_key.upper()}")
        print(f"{'─' * 70}")

        # ── Load Data ──────────────────────────────────────────
        loader = ds_info["loader_class"](**ds_info["loader_kwargs"])
        documents, eval_set = loader.get_data()

        # ── Ingest ─────────────────────────────────────────────
        collection_name = ds_info["collection_name"]
        if not args.skip_ingestion:
            ingest_dataset(
                dataset_name=collection_name,
                documents=documents,
                chunker=chunker,
                embedding_model=embedding_model,
                chroma_client=chroma_client,
            )
        else:
            print(f"[Ingestion] Skipped (--skip-ingestion). "
                  f"Reusing '{collection_name}' collection.")

        # ── Get collection for retrieval ──────────────────────
        collection = chroma_client.get_collection(name=collection_name)
        print(f"[Retrieval] Collection '{collection_name}' has "
              f"{collection.count()} indexed chunks.")

        # ── Evaluate ──────────────────────────────────────────
        results = evaluate_dataset(
            eval_set=eval_set,
            generator=generator,
            embedding_model=embedding_model,
            chroma_collection=collection,
            dry_run=args.dry_run,
        )

        all_detailed_results[ds_key] = results

        # ── Compute & store metrics ──────────────────────────
        if ds_key == "narrativeqa":
            # Split NarrativeQA into Books and Movies
            book_results = [r for r in results if r.get("kind") == "book"]
            movie_results = [r for r in results if r.get("kind") == "movie"]

            book_metrics = compute_dataset_metrics(book_results)
            movie_metrics = compute_dataset_metrics(movie_results)

            all_our_results["NarrativeQA (Books)"] = book_metrics
            all_our_results["NarrativeQA (Movies)"] = movie_metrics

            print(f"\n  NarrativeQA (Books)  — ROUGE-L: {book_metrics['rouge_l']:.4f}, "
                  f"F1: {book_metrics['f1']:.4f} ({len(book_results)} samples)")
            print(f"  NarrativeQA (Movies) — ROUGE-L: {movie_metrics['rouge_l']:.4f}, "
                  f"F1: {movie_metrics['f1']:.4f} ({len(movie_results)} samples)")
        else:
            report_name = ds_info["report_name"]
            metrics = compute_dataset_metrics(results)
            all_our_results[report_name] = metrics

            print(f"\n  {report_name} — ROUGE-L: {metrics['rouge_l']:.4f}, "
                  f"F1: {metrics['f1']:.4f} ({len(results)} samples)")

    # ──────────────────────────────────────────────────────────
    # Generate Final Report
    # ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  GENERATING COMPARISON REPORT")
    print(f"{'─' * 70}")

    report_text = generate_report(all_our_results, save_to_disk=True)
    print(f"\n{report_text}")

    # ── Save detailed per-question results ────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    detailed_path = os.path.join(RESULTS_DIR, "detailed_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(all_detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Results] ✓ Detailed per-question results saved to: {detailed_path}")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE — Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
