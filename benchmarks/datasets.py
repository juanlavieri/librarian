"""Dataset loading and deterministic hierarchical benchmark generation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set


BEIR_DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
    },
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "md5": "a89dba18a62ef92f7d323ec890a0d38d",
    },
}


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    title: str
    text: str
    uri: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRecord:
    id: str
    text: str
    expected_answer: str = ""


@dataclass
class BenchmarkDataset:
    name: str
    corpus: Dict[str, CorpusRecord]
    queries: Dict[str, QueryRecord]
    qrels: Dict[str, Dict[str, int]]
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        missing_queries = set(self.qrels) - set(self.queries)
        if missing_queries:
            raise ValueError(f"qrels reference missing queries: {sorted(missing_queries)[:3]}")
        missing_docs: Set[str] = set()
        for rels in self.qrels.values():
            missing_docs.update(set(rels) - set(self.corpus))
        if missing_docs:
            raise ValueError(f"qrels reference missing documents: {sorted(missing_docs)[:3]}")


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - dataset integrity, not cryptography
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_beir(name: str, data_dir: Path) -> Path:
    """Download and verify a small BEIR dataset, returning its extracted path."""
    if name not in BEIR_DATASETS:
        raise ValueError(f"Unsupported BEIR dataset: {name}")
    spec = BEIR_DATASETS[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    extracted = data_dir / name
    if not archive.exists() or _md5(archive) != spec["md5"]:
        urllib.request.urlretrieve(str(spec["url"]), archive)  # noqa: S310
    actual = _md5(archive)
    if actual != spec["md5"]:
        raise ValueError(f"{name} checksum mismatch: expected {spec['md5']}, got {actual}")
    if not (extracted / "corpus.jsonl").exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    return extracted


def load_beir(name: str, data_dir: Path, split: str = "test") -> BenchmarkDataset:
    root = fetch_beir(name, data_dir)
    corpus: Dict[str, CorpusRecord] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            doc_id = str(raw["_id"])
            corpus[doc_id] = CorpusRecord(
                id=doc_id,
                title=str(raw.get("title") or ""),
                text=str(raw.get("text") or ""),
                uri=f"beir://{name}/{doc_id}",
                metadata={"dataset": name},
            )
    queries: Dict[str, QueryRecord] = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            qid = str(raw["_id"])
            queries[qid] = QueryRecord(id=qid, text=str(raw.get("text") or ""))
    qrels: Dict[str, Dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            qid, doc_id = str(row["query-id"]), str(row["corpus-id"])
            qrels.setdefault(qid, {})[doc_id] = int(row["score"])
    # BEIR query files can contain train/dev queries; retain only the evaluated split.
    queries = {qid: queries[qid] for qid in qrels if qid in queries}
    dataset = BenchmarkDataset(
        name=name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={
            "source": str(BEIR_DATASETS[name]["url"]),
            "checksum_md5": BEIR_DATASETS[name]["md5"],
            "split": split,
        },
    )
    dataset.validate()
    return dataset


_DEPARTMENTS = (
    "finance",
    "operations",
    "engineering",
    "security",
    "people",
    "sales",
    "legal",
    "support",
)
_SUBJECTS = (
    "retention policy",
    "incident escalation",
    "expense approval",
    "vendor onboarding",
    "backup recovery",
    "access review",
    "customer renewal",
    "release procedure",
    "travel reimbursement",
    "data classification",
)


def generate_hierarchical(
    passages: int,
    *,
    queries: int = 200,
    seed: int = 20260721,
    versions: int = 3,
) -> BenchmarkDataset:
    """Generate a versioned hierarchy with exact facts and lexical distractors.

    Every query has one current authoritative document. Older versions and
    cross-department documents contain closely related but incorrect values,
    exercising path, hierarchy, version, and evidence handling.
    """
    if passages < queries * (versions + 1):
        raise ValueError("passages must leave room for relevant records and distractors")
    rng = random.Random(seed)
    corpus: Dict[str, CorpusRecord] = {}
    query_map: Dict[str, QueryRecord] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    for i in range(queries):
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        code = f"{dept[:3].upper()}-{1000 + i}"
        current_value = f"{17 + (i * 37) % 900} days"
        qid = f"q-{i:05d}"
        query_map[qid] = QueryRecord(
            id=qid,
            text=f"What is the current {subject} period for {dept} policy {code}?",
            expected_answer=current_value,
        )
        current_id = f"doc-{i:05d}-v{versions}"
        qrels[qid] = {current_id: 2}
        for version in range(1, versions + 1):
            is_current = version == versions
            value = current_value if is_current else f"{11 + (i * 19 + version) % 700} days"
            doc_id = f"doc-{i:05d}-v{version}"
            corpus[doc_id] = CorpusRecord(
                id=doc_id,
                title=f"{dept.title()} {subject.title()} Policy {code} v{version}",
                text=(
                    f"Policy {code}. Department: {dept}. Subject: {subject}. "
                    f"Version {version} of {versions}. The {'current' if is_current else 'superseded'} "
                    f"{subject} period is {value}. Owner: {dept} governance. "
                    "This controlled record must be cited with its policy code and version."
                ),
                uri=f"file:///company/{dept}/policies/{subject.replace(' ', '-')}/{code}/v{version}.md",
                metadata={
                    "department": dept,
                    "subject": subject,
                    "version": version,
                    "is_current": is_current,
                    "authoritative": is_current,
                },
            )

    # Fill to target size with topically similar distractors.
    next_id = 0
    while len(corpus) < passages:
        dept = rng.choice(_DEPARTMENTS)
        subject = rng.choice(_SUBJECTS)
        token = rng.randint(100000, 999999)
        doc_id = f"distractor-{next_id:08d}"
        next_id += 1
        corpus[doc_id] = CorpusRecord(
            id=doc_id,
            title=f"{dept.title()} {subject.title()} Working Note {token}",
            text=(
                f"Working note {token} for {dept}. This discusses {subject}, review cycles, "
                f"and historical planning assumptions. Draft estimate: {rng.randint(10, 900)} days. "
                "This note is non-authoritative and does not define a current policy."
            ),
            uri=f"file:///company/{dept}/working-notes/{subject.replace(' ', '-')}/{token}.md",
            metadata={"department": dept, "subject": subject, "is_current": True, "authoritative": False},
        )

    dataset = BenchmarkDataset(
        name=f"hierarchical-{passages}",
        corpus=corpus,
        queries=query_map,
        qrels=qrels,
        metadata={
            "generator": "benchmarks.datasets.generate_hierarchical",
            "seed": seed,
            "versions": versions,
            "passages": passages,
        },
    )
    dataset.validate()
    return dataset

