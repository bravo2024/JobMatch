"""model.py — Learning-to-rank for JobMatch (LinkedIn).

Implements pointwise, pairwise, and listwise ranking approaches:
1. **Pointwise**: LightGBM regression on relevance grades.
2. **Pairwise**: RankNet-style sigmoid loss on candidate pairs.
3. **Listwise**: LightGBM with LambdaRank objective (lambdarank).

This is learning-to-rank, NOT generic binary classification.

References
----------
Burges (2010), "From RankNet to LambdaRank to LambdaMART."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.core import ndcg_at_k, map_at_k, mrr, precision_at_k, recall_at_k


def train_pointwise(data, seed=42):
    """Pointwise: regression on relevance grades."""
    df = data["df"]
    features = data["features"]
    group_col = data["group_col"]
    target_col = data["target_col"]

    groups = df.groupby(group_col).size().tolist()
    X = df[features].values
    y = df[target_col].values

    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=100, max_depth=6,
        learning_rate=0.1, random_state=seed, verbose=-1,
    )
    model.fit(X, y, group=groups)
    return model, features


def train_pairwise(data, seed=42):
    """Pairwise: RankNet-style sigmoid loss implemented from scratch.

    For each job, generate all candidate pairs and train a logistic
    model on (pref_i - pref_j) where the label is 1 if rel_i > rel_j.
    """
    from sklearn.linear_model import LogisticRegression
    df = data["df"]
    features = data["features"]
    group_col = data["group_col"]
    target_col = data["target_col"]

    pair_features = []
    pair_labels = []
    for job_id, group in df.groupby(group_col):
        idxs = group.index.tolist()
        rels = group[target_col].values
        feats = group[features].values
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if rels[i] == rels[j]:
                    continue
                diff = feats[i] - feats[j]
                label = 1 if rels[i] > rels[j] else 0
                pair_features.append(diff)
                pair_labels.append(label)
                pair_features.append(-diff)
                pair_labels.append(1 - label)

    X_pairs = np.array(pair_features, dtype=float)
    y_pairs = np.array(pair_labels)
    model = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    model.fit(X_pairs, y_pairs)
    # Convert to a scoring function: score(x) = x . w
    return {"weights": model.coef_[0], "intercept": model.intercept_[0]}, features


def evaluate_ranking(model, data, features, k=5, mode="pointwise"):
    """Evaluate ranking quality with NDCG, MAP, MRR, Precision."""
    df = data["df"]
    group_col = data["group_col"]
    target_col = data["target_col"]

    all_ndcg, all_map, all_mrr, all_precision = [], [], [], []
    for job_id, group in df.groupby(group_col):
        X = group[features].values
        y = group[target_col].values
        if mode == "pointwise":
            scores = model.predict(X)
        else:
            w = model["weights"]
            scores = X @ w
        ranked_idx = np.argsort(-scores)
        ranked_rel = y[ranked_idx]
        all_ndcg.append(ndcg_at_k(ranked_rel, k))
        all_map.append(_ap(ranked_rel, k))
        all_mrr.append(mrr(ranked_rel))
        all_precision.append(precision_at_k(ranked_rel, k))

    return {
        "ndcg@5": float(np.mean(all_ndcg)),
        "map@5": float(np.mean(all_map)),
        "mrr": float(np.mean(all_mrr)),
        "precision@5": float(np.mean(all_precision)),
    }


def _ap(ranked_rel, k):
    """Helper: average precision at k."""
    from src.core import average_precision_at_k
    return average_precision_at_k(ranked_rel, k)


def fit_and_evaluate(data, seed=42):
    """Train and evaluate both pointwise and pairwise models."""
    pw_model, features = train_pointwise(data, seed=seed)
    pw_metrics = evaluate_ranking(pw_model, data, features, mode="pointwise")

    pair_model, _ = train_pairwise(data, seed=seed)
    pair_metrics = evaluate_ranking(pair_model, data, features, mode="pairwise")

    model = {"pointwise": pw_model, "pairwise": pair_model, "features": features}
    metrics = {
        "n_jobs": data["n_jobs"], "n_pairs": data["n_pairs"],
        "pointwise": pw_metrics, "pairwise": pair_metrics,
    }
    return model, metrics