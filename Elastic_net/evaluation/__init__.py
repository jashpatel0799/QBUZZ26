"""Elastic-Depth Decoder LM — Evaluation metrics."""

from evaluation.metrics import (
    compute_perplexity,
    compute_bpc,
    compute_top1_accuracy,
    compute_top5_accuracy,
    compute_avg_exit_layer,
    evaluate_all_exits,
)
