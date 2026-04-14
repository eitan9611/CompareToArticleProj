"""
report.py — Final Comparison Report Generator
==============================================

Generates a formatted comparison table showing:
  - Dataset Name
  - Paper Result (Llama 3.1 8B baseline)
  - Our GMM Result (Llama 3.1 8B)
  - Performance Delta (%)

Also saves raw results to a JSON file for reproducibility.
"""

import json
import os
from typing import Dict
from datetime import datetime

from tabulate import tabulate

from config import PAPER_BENCHMARKS, RESULTS_DIR


def _compute_delta(ours: float, paper: float) -> str:
    """
    Compute percentage delta between our result and the paper's baseline.

    Returns a formatted string like "+12.5%" or "-3.2%".
    """
    if paper == 0:
        return "N/A"
    delta = ((ours - paper) / paper) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def generate_report(
        our_results: Dict[str, Dict[str, float]],
        save_to_disk: bool = True,
) -> str:
    """
    Generate a formatted comparison report.

    Parameters
    ----------
    our_results : dict
        Our benchmark results, keyed by dataset name.
        Example: {"SQuAD-en": {"rouge_l": 0.71, "f1": 0.68}, ...}
    save_to_disk : bool
        Whether to save results to a JSON file.

    Returns
    -------
    str
        Formatted comparison table as a string.
    """

    # ── Build the comparison table ────────────────────────────
    headers = [
        "Dataset",
        "Paper ROUGE-L", "Ours ROUGE-L", "Δ ROUGE-L",
        "Paper F1", "Ours F1", "Δ F1",
    ]

    rows = []
    for dataset_name, paper_scores in PAPER_BENCHMARKS.items():
        our_scores = our_results.get(dataset_name, {"rouge_l": 0.0, "f1": 0.0})

        rows.append([
            dataset_name,
            f"{paper_scores['rouge_l']:.2f}",
            f"{our_scores['rouge_l']:.4f}",
            _compute_delta(our_scores['rouge_l'], paper_scores['rouge_l']),
            f"{paper_scores['f1']:.2f}",
            f"{our_scores['f1']:.4f}",
            _compute_delta(our_scores['f1'], paper_scores['f1']),
        ])

    # ── Format the table ──────────────────────────────────────
    table = tabulate(rows, headers=headers, tablefmt="grid")

    # ── Build the full report ─────────────────────────────────
    report_lines = [
        "=" * 80,
        "RAG BENCHMARK COMPARISON REPORT",
        "=" * 80,
        "",
        "Paper:  Oro, Granata & Ruffolo (2025)",
        "        'A Comprehensive Evaluation of Embedding Models and LLMs",
        "         for IR and QA Across English and Italian'",
        "",
        "Baseline:  Llama 3.1 8B Instruct (Paper's Table 9)",
        "Ours:      Llama 3.1 8B Instruct + GMM Semantic Chunking",
        "",
        "Embedding: intfloat/multilingual-e5-large",
        "Retrieval: ChromaDB, Top-K = 10",
        "Generation: max_new_tokens = 100, greedy decoding",
        "",
        "-" * 80,
        "",
        table,
        "",
        "-" * 80,
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ]

    report_text = "\n".join(report_lines)

    # ── Save to disk ──────────────────────────────────────────
    if save_to_disk:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save formatted report
        report_path = os.path.join(RESULTS_DIR, "comparison_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Save raw results as JSON
        json_path = os.path.join(RESULTS_DIR, "results.json")
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "paper_benchmarks": PAPER_BENCHMARKS,
            "our_results": our_results,
            "deltas": {},
        }
        for ds_name, paper_scores in PAPER_BENCHMARKS.items():
            our = our_results.get(ds_name, {"rouge_l": 0.0, "f1": 0.0})
            json_data["deltas"][ds_name] = {
                "rouge_l_delta_pct": _compute_delta(our["rouge_l"], paper_scores["rouge_l"]),
                "f1_delta_pct": _compute_delta(our["f1"], paper_scores["f1"]),
            }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"\n[Report] ✓ Saved to: {report_path}")
        print(f"[Report] ✓ Raw JSON: {json_path}")

    return report_text
