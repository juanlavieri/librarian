"""IR metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Run = Mapping[str, Sequence[str]]
Qrels = Mapping[str, Mapping[str, int]]


def _relevant(qrels: Mapping[str, int]) -> set[str]:
    return {doc_id for doc_id, grade in qrels.items() if grade > 0}


def recall_at(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(rels)
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = _relevant(rels)
    return sum(doc_id in relevant for doc_id in ranked[:k]) / k


def reciprocal_rank(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(rels)
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(rels)
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def dcg_at(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    return sum(
        (2 ** rels.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )


def ndcg_at(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    ideal_grades = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    ideal = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    return dcg_at(ranked, rels, k) / ideal if ideal else 0.0


def per_query_metrics(run: Run, qrels: Qrels, ks: Sequence[int] = (1, 3, 5, 10, 20)) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    max_k = max(ks)
    for qid, rels in qrels.items():
        ranked = list(run.get(qid, ()))
        row: Dict[str, float] = {
            "mrr": reciprocal_rank(ranked, rels, max_k),
            "map": average_precision(ranked, rels, max_k),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at(ranked, rels, k)
            row[f"precision@{k}"] = precision_at(ranked, rels, k)
            row[f"ndcg@{k}"] = ndcg_at(ranked, rels, k)
        output[qid] = row
    return output


def aggregate(per_query: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    names = next(iter(per_query.values())).keys()
    return {name: mean(row[name] for row in per_query.values()) for name in names}


def evaluate(run: Run, qrels: Qrels, ks: Sequence[int] = (1, 3, 5, 10, 20)) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    per_query = per_query_metrics(run, qrels, ks)
    return aggregate(per_query), per_query


@dataclass(frozen=True)
class ConfidenceInterval:
    mean_delta: float
    low: float
    high: float
    samples: int


def paired_bootstrap(
    candidate: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    samples: int = 2000,
    seed: int = 20260721,
) -> ConfidenceInterval:
    """Paired query bootstrap for candidate-minus-baseline metric deltas."""
    qids = sorted(set(candidate) & set(baseline))
    if not qids:
        raise ValueError("candidate and baseline share no queries")
    deltas = [candidate[qid][metric] - baseline[qid][metric] for qid in qids]
    rng = random.Random(seed)
    boot = [
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(samples)
    ]
    boot.sort()
    low_i = max(0, int(samples * 0.025))
    high_i = min(samples - 1, int(samples * 0.975))
    return ConfidenceInterval(
        mean_delta=mean(deltas),
        low=boot[low_i],
        high=boot[high_i],
        samples=samples,
    )


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * pct
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)

