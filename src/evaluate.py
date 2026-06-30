"""evaluate.py — Metric persistence for JobMatch."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _clean(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)): return int(v)
        if isinstance(v, dict): return {k: _clean(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_clean(x) for x in v]
        return v
    with open(path, "w") as f: json.dump(_clean(metrics), f, indent=2)
    return metrics
