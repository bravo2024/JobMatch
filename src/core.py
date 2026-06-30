"""core.py — Learning-to-rank metrics for JobMatch (LinkedIn).

Implements ranking metrics used in recommender systems, NOT generic
classification metrics:
  * **NDCG@k** — Normalized Discounted Cumulative Gain.
  * **MAP@k** — Mean Average Precision.
  * **MRR** — Mean Reciprocal Rank.
  * **Precision@k / Recall@k** — top-k retrieval metrics.

References
----------
Burges (2010), "From RankNet to LambdaRank to LambdaMART."
Järvelin & Kekäläinen (2002), "Cumulated Gain-based Evaluation of IR."
"""
from __future__ import annotations
import numpy as np


def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain at k."""
    r = np.asarray(relevances, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(np.sum(r / discounts))


def ndcg_at_k(ranked_relevances, k):
    """NDCG@k = DCG@k / IDCG@k (ideal ranking)."""
    dcg = dcg_at_k(ranked_relevances, k)
    ideal = sorted(ranked_relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked_relevances, k):
    """Average Precision@k for a single ranked list."""
    r = np.asarray(ranked_relevances, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    hits = np.cumsum(r > 0)
    precision = hits / np.arange(1, r.size + 1)
    return float(np.sum(precision * (r > 0)) / max(hits[-1], 1))


def map_at_k(all_ranked_relevances, k):
    """Mean Average Precision@k across multiple queries."""
    if not all_ranked_relevances:
        return 0.0
    return float(np.mean([average_precision_at_k(r, k) for r in all_ranked_relevances]))


def mrr(ranked_relevances):
    """Reciprocal rank of first relevant item."""
    r = np.asarray(ranked_relevances, dtype=float)
    for i, rel in enumerate(r, 1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_relevances, k):
    """Fraction of top-k that are relevant."""
    r = np.asarray(ranked_relevances, dtype=float)[:k]
    return float(np.sum(r > 0) / k) if k > 0 else 0.0


def recall_at_k(ranked_relevances, total_relevant, k):
    """Recall@k = relevant in top-k / total relevant."""
    r = np.asarray(ranked_relevances, dtype=float)[:k]
    if total_relevant == 0:
        return 0.0
    return float(np.sum(r > 0) / total_relevant)