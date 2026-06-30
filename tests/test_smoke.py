"""Smoke tests for JobMatch — learning-to-rank candidate-job matching."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import fit_and_evaluate, train_pointwise, evaluate_ranking
from src.core import ndcg_at_k, map_at_k, mrr, precision_at_k, average_precision_at_k


def test_data():
    d = make_synthetic(n_jobs=10, candidates_per_job=10, seed=42)
    assert d["n_pairs"] == 100
    assert "relevance" in d["df"].columns


def test_ndcg():
    """NDCG is 1.0 for ideal ranking."""
    rels = [4, 3, 2, 1, 0]
    assert abs(ndcg_at_k(rels, 5) - 1.0) < 0.01


def test_mrr():
    """MRR is 1/n for first relevant at position n."""
    assert abs(mrr([0, 0, 1, 0]) - 1/3) < 0.01


def test_pointwise():
    """Pointwise LTR model trains and produces rankings."""
    d = make_synthetic(n_jobs=20, candidates_per_job=10, seed=42)
    model, features = train_pointwise(d, seed=42)
    metrics = evaluate_ranking(model, d, features, mode="pointwise")
    assert metrics["ndcg@5"] > 0.5


def test_fit_and_evaluate():
    """Full pipeline returns both pointwise and pairwise metrics."""
    d = make_synthetic(n_jobs=15, candidates_per_job=8, seed=42)
    model, metrics = fit_and_evaluate(d, seed=42)
    assert "pointwise" in metrics
    assert "pairwise" in metrics
    assert metrics["pointwise"]["ndcg@5"] > 0.5


if __name__ == "__main__":
    test_data()
    test_ndcg()
    test_mrr()
    test_pointwise()
    test_fit_and_evaluate()
    print("All JobMatch smoke tests passed!")
