import math

from benchmarks.datasets import generate_hierarchical
from benchmarks.metrics import (
    average_precision,
    evaluate,
    ndcg_at,
    paired_bootstrap,
    precision_at,
    recall_at,
    reciprocal_rank,
)
from benchmarks.promote import BASELINES, DATASETS, evaluate_gate


def test_metrics_match_hand_calculated_fixture():
    ranked = ["d2", "d3", "d1", "d4"]
    rels = {"d1": 2, "d2": 1}

    assert recall_at(ranked, rels, 2) == 0.5
    assert precision_at(ranked, rels, 2) == 0.5
    assert reciprocal_rank(ranked, rels, 4) == 1.0
    assert average_precision(ranked, rels, 4) == (1.0 + 2.0 / 3.0) / 2.0

    actual_dcg = 1.0 + 3.0 / math.log2(4)
    ideal_dcg = 3.0 + 1.0 / math.log2(3)
    assert math.isclose(ndcg_at(ranked, rels, 4), actual_dcg / ideal_dcg)


def test_evaluate_uses_all_qrels_queries_and_empty_results():
    metrics, per_query = evaluate({"q1": ["d1"]}, {"q1": {"d1": 1}, "q2": {"d2": 1}})
    assert set(per_query) == {"q1", "q2"}
    assert metrics["recall@10"] == 0.5
    assert metrics["mrr"] == 0.5


def test_paired_bootstrap_has_positive_interval_for_uniform_win():
    baseline = {f"q{i}": {"ndcg@10": 0.2} for i in range(20)}
    candidate = {f"q{i}": {"ndcg@10": 0.4} for i in range(20)}
    ci = paired_bootstrap(candidate, baseline, "ndcg@10", samples=200)
    assert math.isclose(ci.mean_delta, 0.2)
    assert ci.low > 0
    assert ci.high > 0


def test_hierarchical_generator_is_deterministic_and_valid():
    first = generate_hierarchical(100, queries=20, seed=7)
    second = generate_hierarchical(100, queries=20, seed=7)
    assert first.corpus == second.corpus
    assert first.queries == second.queries
    assert first.qrels == second.qrels
    for qid, rels in first.qrels.items():
        relevant = first.corpus[next(iter(rels))]
        assert first.queries[qid].expected_answer in relevant.text
        assert relevant.metadata["is_current"] is True


def test_promotion_gate_requires_quality_regression_and_latency_gates():
    candidates = {}
    baselines = {name: {} for name in BASELINES}
    for dataset in DATASETS:
        candidates[dataset] = {
            "metrics": {"ndcg@10": 0.6},
            "per_query": {f"q{i}": {"ndcg@10": 0.6, "recall@10": 0.7} for i in range(20)},
            "operational": {"warm_latency_p95_ms": 10.0},
        }
        for name in BASELINES:
            baselines[name][dataset] = {
                "metrics": {"ndcg@10": 0.4},
                "per_query": {f"q{i}": {"ndcg@10": 0.4, "recall@10": 0.5} for i in range(20)},
                "operational": {"warm_latency_p95_ms": 20.0},
            }
    result = evaluate_gate(candidates, baselines, samples=200)
    assert result["passed"] is True
    candidates["scifact"]["operational"]["warm_latency_p95_ms"] = 30.0
    assert evaluate_gate(candidates, baselines, samples=200)["passed"] is False

