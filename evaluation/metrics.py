"""
metrics.py — ROUGE-L and Token-level F1 Score Calculators
=========================================================

Implements the two evaluation metrics used in the paper:

1. ROUGE-L: Longest Common Subsequence based metric
   (via the `rouge-score` library for correctness).

2. Token-level F1: Standard SQuAD-style F1 computed over
   whitespace-tokenized prediction vs reference text.
"""

import re
import string
from typing import Dict, List
from collections import Counter

from rouge_score import rouge_scorer


# ──────────────────────────────────────────────────────────────
# ROUGE-L
# ──────────────────────────────────────────────────────────────

# Shared scorer instance (stateless, safe to reuse)
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F-measure between prediction and reference.

    Parameters
    ----------
    prediction : str
        The model-generated answer.
    reference : str
        The ground-truth answer.

    Returns
    -------
    float
        ROUGE-L F-measure score in [0, 1].
    """
    if not prediction or not reference:
        return 0.0

    scores = _rouge_scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


# ──────────────────────────────────────────────────────────────
# Token-level F1 (SQuAD-style)
# ──────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """
    Normalize text for F1 comparison:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def compute_token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.

    This is the standard SQuAD F1 metric: tokenize both strings on
    whitespace, compute precision and recall over token overlap,
    and return the harmonic mean.

    Parameters
    ----------
    prediction : str
        The model-generated answer.
    reference : str
        The ground-truth answer.

    Returns
    -------
    float
        Token-level F1 score in [0, 1].
    """
    if not prediction or not reference:
        return 0.0

    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Count token overlap
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


# ──────────────────────────────────────────────────────────────
# Dataset-level Aggregation
# ──────────────────────────────────────────────────────────────

def compute_dataset_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate per-question scores into dataset-level averages.

    Parameters
    ----------
    results : List[dict]
        Each dict must contain 'rouge_l' and 'f1' keys with float values.

    Returns
    -------
    dict
        {"rouge_l": float, "f1": float} — macro-averaged scores.
    """
    if not results:
        return {"rouge_l": 0.0, "f1": 0.0}

    avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)

    return {
        "rouge_l": round(avg_rouge, 4),
        "f1": round(avg_f1, 4),
    }
