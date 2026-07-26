"""Create a compact CSV summary from benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


def _load(path: Path) -> Mapping[str, object]:
    if path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = path.open("r", encoding="utf-8")
    with fh:
        return json.load(fh)


def rows(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        artifact = _load(path)
        dataset = artifact["dataset"]
        assert isinstance(dataset, dict)
        for result in artifact["results"]:
            metrics = result["metrics"]
            operational = result["operational"]
            yield {
                "artifact": path.name,
                "created_at": artifact["created_at"],
                "git_commit": artifact["git_commit"],
                "dataset": dataset["name"],
                "corpus_records": dataset["corpus_records"],
                "queries": dataset["queries"],
                "system": result["name"],
                "ndcg@10": metrics.get("ndcg@10", ""),
                "recall@10": metrics.get("recall@10", ""),
                "mrr": metrics.get("mrr", ""),
                "map": metrics.get("map", ""),
                "indexing_seconds": operational["indexing_seconds"],
                "index_bytes": operational["index_bytes"],
                "cold_latency_ms": operational["cold_latency_ms"],
                "warm_latency_p50_ms": operational["warm_latency_p50_ms"],
                "warm_latency_p95_ms": operational["warm_latency_p95_ms"],
            }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths: list[Path] = []
    for pattern in args.inputs:
        paths.extend(sorted(Path().glob(pattern)))
    data = list(rows(paths))
    if not data:
        raise SystemExit("no benchmark artifacts matched")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

