"""CLI runner for reproducible Librarian retrieval benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .datasets import BenchmarkDataset, generate_hierarchical, load_beir
from .metrics import evaluate, paired_bootstrap, percentile
from .systems import (
    BenchmarkSystem,
    CurrentHybridSystem,
    FTS5BM25System,
    ForcedQueryFailureSystem,
    LexicalFallbackSystem,
    ProductionFTS5System,
    RRFSystem,
    SentenceTransformerSystem,
)


DEFAULT_SYSTEMS = ("lexical-overlap", "hashing-cosine", "current-hybrid", "fts5-bm25")
_DEPENDENCIES = (
    "librarian-ai",
    "numpy",
    "faiss-cpu",
    "sentence-transformers",
    "torch",
    "transformers",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _source_state() -> Dict[str, object]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL
        )
        listed = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        excluded_prefixes = ("benchmarks/data/", "benchmarks/runs/", "benchmarks/results/", "dist/")
        source_files = [
            path
            for path in sorted(listed)
            if (path.startswith(("src/", "benchmarks/")) or path == "pyproject.toml")
            and not path.startswith(excluded_prefixes)
            and "__pycache__/" not in path
            and not path.endswith((".pyc", ".pyo"))
        ]
        tree = hashlib.sha256()
        for relative in source_files:
            path = Path(relative)
            if not path.is_file():
                continue
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(hashlib.sha256(path.read_bytes()).digest())
        return {
            "dirty": bool(status.strip()),
            "status": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_tree_sha256": tree.hexdigest(),
            "source_file_count": len(source_files),
        }
    except Exception:
        return {"dirty": None}


def _dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for distribution in _DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    versions["sqlite"] = __import__("sqlite3").sqlite_version
    return versions


def _systems(names: Sequence[str]) -> List[BenchmarkSystem]:
    systems: List[BenchmarkSystem] = []
    for name in names:
        if name == "lexical-overlap":
            systems.append(
                CurrentHybridSystem(
                    name=name, semantic_weight=0.0, lexical_weight=1.0, contextual=True
                )
            )
        elif name == "hashing-cosine":
            systems.append(
                CurrentHybridSystem(
                    name=name, semantic_weight=1.0, lexical_weight=0.0, contextual=True
                )
            )
        elif name == "current-hybrid":
            systems.append(CurrentHybridSystem(name=name, contextual=True))
        elif name == "current-no-rollups":
            systems.append(
                CurrentHybridSystem(name=name, contextual=True, include_rollups=False)
            )
        elif name == "current-no-context":
            systems.append(CurrentHybridSystem(name=name, contextual=False))
        elif name == "faiss-hybrid":
            from librarian.vectorstore.faiss_store import FaissVectorStore

            systems.append(
                CurrentHybridSystem(
                    name=name,
                    contextual=True,
                    store=FaissVectorStore(dim=512),
                )
            )
        elif name == "fts5-bm25":
            systems.append(FTS5BM25System())
        elif name == "production-fts5":
            systems.append(ProductionFTS5System())
        elif name == "dense-minilm":
            systems.append(SentenceTransformerSystem())
        elif name == "rrf-bm25-minilm":
            systems.append(
                RRFSystem(
                    FTS5BM25System(),
                    SentenceTransformerSystem(),
                    name=name,
                )
            )
        elif name == "rrf-bm25-hashing":
            systems.append(
                RRFSystem(
                    FTS5BM25System(),
                    CurrentHybridSystem(
                        name="hashing-cosine-inner",
                        semantic_weight=1.0,
                        lexical_weight=0.0,
                    ),
                    name=name,
                )
            )
        elif name == "forced-fallback-bm25":
            systems.append(
                LexicalFallbackSystem(ForcedQueryFailureSystem(), FTS5BM25System())
            )
            systems[-1].name = name
        else:
            raise ValueError(f"Unknown system: {name}")
    return systems


def _grounding(dataset: BenchmarkDataset, rankings: Mapping[str, Sequence[str]], k: int = 5) -> Dict[str, float]:
    answer_queries = [q for q in dataset.queries.values() if q.expected_answer]
    if not answer_queries:
        return {}
    answer_hits = 0
    citation_hits = 0
    for query in answer_queries:
        ranked = list(rankings.get(query.id, ()))[:k]
        if any(doc_id in dataset.qrels.get(query.id, {}) for doc_id in ranked):
            citation_hits += 1
        if any(
            query.expected_answer.lower() in dataset.corpus[doc_id].text.lower()
            for doc_id in ranked
            if doc_id in dataset.corpus
        ):
            answer_hits += 1
    total = len(answer_queries)
    return {
        f"evidence_contains_answer@{k}": answer_hits / total,
        f"relevant_citation@{k}": citation_hits / total,
    }


def run_system(
    system: BenchmarkSystem,
    dataset: BenchmarkDataset,
    *,
    k: int,
    repetitions: int,
) -> Dict[str, object]:
    index = system.build(dataset)
    query_items = list(dataset.queries.values())
    rankings: Dict[str, List[str]] = {}
    scored_rankings: Dict[str, List[Dict[str, object]]] = {}
    cold_ms: List[float] = []
    warm_ms: List[float] = []
    for repetition in range(repetitions):
        for index_q, query in enumerate(query_items):
            started = time.perf_counter()
            ranked = system.rank(query.text, k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if repetition == 0 and index_q == 0:
                cold_ms.append(elapsed_ms)
            else:
                warm_ms.append(elapsed_ms)
            if repetition == 0:
                rankings[query.id] = [doc_id for doc_id, _ in ranked]
                scored_rankings[query.id] = [
                    {"doc_id": doc_id, "score": score, "rank": rank}
                    for rank, (doc_id, score) in enumerate(ranked, 1)
                ]
    metrics, per_query = evaluate(rankings, dataset.qrels)
    metrics.update(_grounding(dataset, rankings))
    metadata = getattr(system, "metadata", None)
    return {
        "name": system.name,
        "system_metadata": metadata() if callable(metadata) else {},
        "metrics": metrics,
        "per_query": per_query,
        "rankings": scored_rankings,
        "operational": {
            "indexing_seconds": index.seconds,
            "index_bytes": index.bytes,
            "records": index.records,
            "cold_latency_ms": cold_ms[0] if cold_ms else 0.0,
            "warm_latency_p50_ms": percentile(warm_ms, 0.50),
            "warm_latency_p95_ms": percentile(warm_ms, 0.95),
            "timed_queries": len(cold_ms) + len(warm_ms),
            "repetitions": repetitions,
        },
    }


def _load_dataset(args: argparse.Namespace) -> BenchmarkDataset:
    if args.dataset in ("scifact", "nfcorpus"):
        return load_beir(args.dataset, Path(args.data_dir))
    if args.dataset.startswith("hierarchical-"):
        passages = int(args.dataset.split("-", 1)[1])
        return generate_hierarchical(
            passages,
            queries=min(args.queries, max(1, passages // 4)),
            seed=args.seed,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def _comparisons(results: Sequence[Mapping[str, object]], baseline_name: str) -> Dict[str, object]:
    by_name = {str(result["name"]): result for result in results}
    if baseline_name not in by_name:
        return {}
    baseline = by_name[baseline_name]
    output: Dict[str, object] = {}
    for name, result in by_name.items():
        if name == baseline_name:
            continue
        row = {}
        for metric in ("ndcg@10", "recall@10"):
            ci = paired_bootstrap(
                result["per_query"],  # type: ignore[arg-type]
                baseline["per_query"],  # type: ignore[arg-type]
                metric,
            )
            row[metric] = {
                "mean_delta": ci.mean_delta,
                "ci95_low": ci.low,
                "ci95_high": ci.high,
                "samples": ci.samples,
            }
        output[name] = row
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Librarian retrieval benchmarks")
    parser.add_argument(
        "--dataset",
        required=True,
        help="scifact, nfcorpus, or hierarchical-N (for example hierarchical-1000)",
    )
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--data-dir", default="benchmarks/data")
    parser.add_argument("--output-dir", default="benchmarks/runs")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--baseline", default="current-hybrid")
    args = parser.parse_args(argv)

    dataset = _load_dataset(args)
    systems = _systems([name.strip() for name in args.systems.split(",") if name.strip()])
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for system in systems:
        print(f"[benchmark] {dataset.name}: building/running {system.name}", flush=True)
        try:
            results.append(run_system(system, dataset, k=args.k, repetitions=args.repetitions))
        except Exception as exc:
            failures.append({"name": system.name, "type": type(exc).__name__, "message": str(exc)})
            print(f"[benchmark] {system.name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            system.close()

    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_state": _source_state(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "dependencies": _dependency_versions(),
        },
        "dataset": {
            "name": dataset.name,
            "corpus_records": len(dataset.corpus),
            "queries": len(dataset.queries),
            "qrels": sum(len(r) for r in dataset.qrels.values()),
            "metadata": dataset.metadata,
        },
        "config": {
            "k": args.k,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "systems": [system.name for system in systems],
        },
        "results": results,
        "failures": failures,
        "comparisons_to_baseline": _comparisons(results, args.baseline),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_dir / f"{dataset.name}-{stamp}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

