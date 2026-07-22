"""Reproduce the cross-dataset promotion gate and confidence intervals."""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .metrics import paired_bootstrap


DATASETS = ("scifact", "nfcorpus", "hierarchical-1000")
BASELINES = ("current-hybrid", "lexical-overlap", "hashing-cosine")
METRICS = ("ndcg@10", "recall@10")


def _load(path: Path) -> dict:
    if path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = path.open("r", encoding="utf-8")
    with fh:
        return json.load(fh)


def _latest_with_system(directory: Path, dataset: str, system: str) -> dict:
    matches = sorted(directory.glob(f"{dataset}-*.json")) + sorted(
        directory.glob(f"{dataset}-*.json.gz")
    )
    if not matches:
        raise FileNotFoundError(f"no artifact for {dataset} in {directory}")
    for path in reversed(matches):
        artifact = _load(path)
        if any(result["name"] == system for result in artifact.get("results", [])):
            return artifact
    raise ValueError(f"no artifact for {dataset} in {directory} contains {system}")


def _result(artifact: Mapping[str, object], system: str) -> Mapping[str, object]:
    for result in artifact["results"]:  # type: ignore[index]
        if result["name"] == system:
            return result
    raise ValueError(f"artifact does not contain system {system}")


def evaluate_gate(
    candidates: Mapping[str, Mapping[str, object]],
    baselines: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    samples: int = 10_000,
) -> dict:
    pooled_candidate: Dict[str, Mapping[str, float]] = {}
    pooled_baselines: Dict[str, Dict[str, Mapping[str, float]]] = {
        name: {} for name in BASELINES
    }
    per_dataset: Dict[str, object] = {}
    no_regression = True
    latency_pass = True

    for dataset in DATASETS:
        candidate = candidates[dataset]
        current = baselines["current-hybrid"][dataset]
        candidate_metrics = candidate["metrics"]  # type: ignore[index]
        current_metrics = current["metrics"]  # type: ignore[index]
        candidate_ndcg = float(candidate_metrics["ndcg@10"])  # type: ignore[index]
        current_ndcg = float(current_metrics["ndcg@10"])  # type: ignore[index]
        relative_delta = (
            (candidate_ndcg - current_ndcg) / current_ndcg if current_ndcg else 0.0
        )
        dataset_no_regression = relative_delta >= -0.02
        candidate_p95 = float(candidate["operational"]["warm_latency_p95_ms"])  # type: ignore[index]
        current_p95 = float(current["operational"]["warm_latency_p95_ms"])  # type: ignore[index]
        dataset_latency_pass = candidate_p95 <= current_p95 * 1.25
        no_regression = no_regression and dataset_no_regression
        latency_pass = latency_pass and dataset_latency_pass
        per_dataset[dataset] = {
            "candidate_ndcg@10": candidate_ndcg,
            "current_ndcg@10": current_ndcg,
            "relative_ndcg_delta": relative_delta,
            "ndcg_regression_gate": dataset_no_regression,
            "candidate_warm_p95_ms": candidate_p95,
            "current_warm_p95_ms": current_p95,
            "latency_gate": dataset_latency_pass,
        }
        for qid, row in candidate["per_query"].items():  # type: ignore[index]
            pooled_candidate[f"{dataset}:{qid}"] = row
        for baseline_name in BASELINES:
            for qid, row in baselines[baseline_name][dataset]["per_query"].items():  # type: ignore[index]
                pooled_baselines[baseline_name][f"{dataset}:{qid}"] = row

    comparisons: Dict[str, object] = {}
    quality_pass = True
    for baseline_name, baseline_rows in pooled_baselines.items():
        metric_results = {}
        for metric in METRICS:
            ci = paired_bootstrap(
                pooled_candidate,
                baseline_rows,
                metric,
                samples=samples,
            )
            passed = ci.low > 0
            quality_pass = quality_pass and passed
            metric_results[metric] = {
                "mean_delta": ci.mean_delta,
                "ci95_low": ci.low,
                "ci95_high": ci.high,
                "samples": ci.samples,
                "passed": passed,
            }
        comparisons[baseline_name] = metric_results
    return {
        "passed": quality_pass and no_regression and latency_pass,
        "quality_gate": quality_pass,
        "per_dataset_regression_gate": no_regression,
        "latency_gate": latency_pass,
        "query_count": len(pooled_candidate),
        "comparisons": comparisons,
        "per_dataset": per_dataset,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate", default="fts5-bm25")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    candidates: Dict[str, Mapping[str, object]] = {}
    baselines: Dict[str, Dict[str, Mapping[str, object]]] = {
        name: {} for name in BASELINES
    }
    inputs: Dict[str, object] = {}
    for dataset in DATASETS:
        candidate_artifact = _latest_with_system(
            Path(args.candidate_dir), dataset, args.candidate
        )
        candidates[dataset] = _result(candidate_artifact, args.candidate)
        for baseline_name in BASELINES:
            baseline_artifact = _latest_with_system(
                Path(args.baseline_dir), dataset, baseline_name
            )
            baselines[baseline_name][dataset] = _result(baseline_artifact, baseline_name)
        inputs[dataset] = {
            "candidate_created_at": candidate_artifact["created_at"],
            "baseline_created_at": baseline_artifact["created_at"],
        }
    result = evaluate_gate(candidates, baselines, samples=args.samples)
    result.update(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate": args.candidate,
            "inputs": inputs,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

