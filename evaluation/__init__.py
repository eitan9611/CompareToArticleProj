"""
evaluation — Metrics calculation and report generation.
"""

from evaluation.metrics import compute_rouge_l, compute_token_f1, compute_dataset_metrics
from evaluation.report import generate_report

__all__ = ["compute_rouge_l", "compute_token_f1", "compute_dataset_metrics", "generate_report"]
