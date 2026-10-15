"""Retrieval metrics with bootstrap confidence intervals.

Every metric is computed per query, then averaged. With 200 queries a single query
moves a rate by 0.5 points, so averages are always reported with a 95% interval.

    recall_any@k  - at least one gold article in the top k
    recall_all@k  - every gold article in the top k (26% of ExpertWritten need 2-3)
    mrr           - 1 / rank of the first gold article (0 if not retrieved)
    ndcg@10       - binary-relevance nDCG over the top 10
"""

from __future__ import annotations

import math

import numpy as np

KS = (1, 5, 10, 20)


def per_query(ranked: list[str], gold: tuple[str, ...]) -> dict[str, float]:
    gold_set = set(gold)
    pos = {a: i for i, a in enumerate(ranked)}
    gold_ranks = sorted(pos[g] + 1 for g in gold_set if g in pos)
    out: dict[str, float] = {}
    for k in KS:
        found = sum(r <= k for r in gold_ranks)
        out[f"recall_any@{k}"] = float(found > 0)
        out[f"recall_all@{k}"] = float(found == len(gold_set))
    out["mrr"] = 1.0 / gold_ranks[0] if gold_ranks else 0.0
    dcg = sum(1 / math.log2(r + 1) for r in gold_ranks if r <= 10)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold_set), 10)))
    out["ndcg@10"] = dcg / idcg
    out["first_gold_rank"] = float(gold_ranks[0]) if gold_ranks else math.inf
    return out


def bootstrap_ci(values, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Difference b - a on the same queries: mean, 95% interval, two-sided p-value."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "p": float(min(p, 1.0)),
    }


def summarize(rows: list[dict[str, float]], names: list[str]) -> dict[str, dict]:
    out = {}
    for name in names:
        vals = [r[name] for r in rows]
        lo, hi = bootstrap_ci(vals)
        out[name] = {"mean": float(np.mean(vals)), "lo": lo, "hi": hi}
    return out
