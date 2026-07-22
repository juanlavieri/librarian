"""Retrieval systems and controlled ablations used by the benchmark."""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from librarian.config import LibrarianConfig
from librarian.embeddings.hashing import HashingEmbedder
from librarian.models import DOC_TYPE_FOLDER_ROLLUP, DOC_TYPE_SECTION_ROLLUP, DOC_TYPE_SUMMARY
from librarian.retrieval import Retriever
from librarian.rollups import build_folder_rollups
from librarian.vectorstore import Filters, IndexRecord, LocalVectorStore

from .datasets import BenchmarkDataset, CorpusRecord


@dataclass
class IndexStats:
    seconds: float
    bytes: int
    records: int


class BenchmarkSystem(Protocol):
    name: str

    def build(self, dataset: BenchmarkDataset) -> IndexStats: ...

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]: ...

    def close(self) -> None: ...


def _record_text(record: CorpusRecord, contextual: bool) -> str:
    if not contextual:
        return record.text
    location = record.uri or record.id
    department = str(record.metadata.get("department") or "")
    context = f"Document: {record.title} | Location: {location}"
    if department:
        context += f" | Department: {department}"
    return f"[{context}]\n{record.text}"


class CurrentHybridSystem:
    """Exact current local retrieval scoring over passage-level records.

    This uses Librarian's ``HashingEmbedder``, ``LocalVectorStore`` lexical and
    cosine implementations, and ``Retriever._hybrid`` weighted fusion. It
    bypasses parsing/summarization so all systems receive identical benchmark
    passages; this isolates retrieval behavior.
    """

    def __init__(
        self,
        *,
        name: str = "current-hybrid",
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        contextual: bool = True,
        include_rollups: bool = True,
        dim: int = 512,
        store=None,
    ) -> None:
        self.name = name
        self.contextual = contextual
        self.include_rollups = include_rollups
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.embedder = HashingEmbedder(dim=dim)
        self.store = store or LocalVectorStore()
        cfg = LibrarianConfig(
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            embedding_dim=dim,
            retrieval_strategy="hybrid",
        )
        self.retriever = Retriever(self.store, self.embedder, cfg)

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        started = time.perf_counter()
        records: List[IndexRecord] = []
        texts: List[str] = []
        for item in dataset.corpus.values():
            text = _record_text(item, self.contextual)
            texts.append(text)
            records.append(
                IndexRecord(
                    id=item.id,
                    doc_type=DOC_TYPE_SUMMARY,
                    doc_id=item.id,
                    text=text,
                    title=item.title,
                    uri=item.uri,
                    source_id=dataset.name,
                    is_current=bool(item.metadata.get("is_current", True)),
                    tags=[
                        str(item.metadata[key])
                        for key in ("department", "subject")
                        if key in item.metadata
                    ],
                )
            )
        if self.include_rollups:
            rollup_items = [
                {
                    "doc_id": item.id,
                    "uri": item.uri,
                    "title": item.title,
                    "summary": item.text[:500],
                }
                for item in dataset.corpus.values()
                if bool(item.metadata.get("is_current", True))
            ]
            for rollup in build_folder_rollups(rollup_items, min_items=2):
                texts.append(rollup.text)
                records.append(
                    IndexRecord(
                        id=rollup.rollup_id,
                        doc_type=rollup.doc_type,
                        doc_id=rollup.rollup_id,
                        text=rollup.text,
                        title=f"Folder {rollup.key}",
                        uri=f"rollup://{rollup.key}",
                        source_id=dataset.name,
                        is_current=True,
                    )
                )
        vectors = self.embedder.embed(texts)
        for rec, vector in zip(records, vectors):
            rec.vector = vector
        self.store.upsert(records)
        elapsed = time.perf_counter() - started
        # Approximate resident index bytes using encoded text + float32 vectors.
        size = sum(len(r.text.encode("utf-8")) + len(r.vector) * 4 for r in records)
        return IndexStats(elapsed, size, len(records))

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]:
        doc_types = [DOC_TYPE_SUMMARY]
        if self.include_rollups:
            doc_types.extend([DOC_TYPE_FOLDER_ROLLUP, DOC_TYPE_SECTION_ROLLUP])
        filters = Filters(doc_types=doc_types, is_current=True)
        if self.semantic_weight == 0:
            return [
                (record.id, score)
                for record, score in self.store.search_lexical(query, k, filters)
            ]
        if self.lexical_weight == 0:
            vector = self.embedder.embed_one(query)
            return [
                (record.id, score)
                for record, score in self.store.search_semantic(vector, k, filters)
            ]
        return [(record.id, score) for record, score in self.retriever._hybrid(query, k, filters)]

    def close(self) -> None:
        self.store.close()

    def metadata(self) -> Dict[str, object]:
        return {
            "retrieval_strategy": self.retriever.cfg.retrieval_strategy,
            "fusion_strategy": self.retriever.cfg.fusion_strategy,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "embedding": f"librarian.HashingEmbedder(dim={self.embedder.dim})",
            "include_rollups": self.include_rollups,
            "contextual_headers": self.contextual,
        }


class ProductionFTS5System(CurrentHybridSystem):
    """The promoted production path: LocalVectorStore + Retriever + roll-ups."""

    def __init__(self) -> None:
        super().__init__(name="production-fts5", contextual=True, include_rollups=True)
        self.retriever.cfg.retrieval_strategy = "fts5"

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        stats = super().build(dataset)
        started = time.perf_counter()
        ensure_fts = getattr(self.store, "_ensure_fts")
        with self.store._fts_lock:
            conn = ensure_fts()
            page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        return IndexStats(
            stats.seconds + (time.perf_counter() - started),
            stats.bytes + page_count * page_size,
            stats.records,
        )

    def metadata(self) -> Dict[str, object]:
        metadata = super().metadata()
        metadata["sqlite_version"] = sqlite3.sqlite_version
        return metadata


class FTS5BM25System:
    """SQLite FTS5 passage retrieval using SQLite's built-in BM25 scorer."""

    name = "fts5-bm25"

    def __init__(self, path: Optional[Path] = None) -> None:
        self._temp = path is None
        self.path = path or Path(tempfile.mkstemp(prefix="librarian-bm25-", suffix=".db")[1])
        self.conn: Optional[sqlite3.Connection] = None

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        started = time.perf_counter()
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE VIRTUAL TABLE passages USING fts5("
            "doc_id UNINDEXED, title, uri, body, tokenize='unicode61 remove_diacritics 2')"
        )
        rows = [
            (item.id, item.title, item.uri, _record_text(item, contextual=True))
            for item in dataset.corpus.values()
            if bool(item.metadata.get("is_current", True))
        ]
        self.conn.executemany("INSERT INTO passages VALUES (?,?,?,?)", rows)
        self.conn.commit()
        elapsed = time.perf_counter() - started
        size = self.path.stat().st_size if self.path.exists() else 0
        return IndexStats(elapsed, size, len(rows))

    @staticmethod
    def _query(text: str) -> str:
        terms = re.findall(r"\b[\w-]+\b", text.lower(), flags=re.UNICODE)
        # OR is robust for natural-language queries and avoids FTS syntax input.
        return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self.conn is None:
            raise RuntimeError("build must be called before rank")
        fts_query = self._query(query)
        if not fts_query:
            return []
        rows = self.conn.execute(
            "SELECT doc_id, bm25(passages, 0.0, 1.5, 0.75, 1.0) AS distance "
            "FROM passages WHERE passages MATCH ? ORDER BY distance LIMIT ?",
            (fts_query, k),
        ).fetchall()
        # SQLite bm25 is lower-is-better (usually negative); expose higher-is-better.
        return [(str(doc_id), -float(distance)) for doc_id, distance in rows]

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self._temp:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def metadata(self) -> Dict[str, object]:
        return {
            "sqlite_version": sqlite3.sqlite_version,
            "tokenizer": "unicode61 remove_diacritics 2",
            "contextual_headers": True,
            "include_rollups": False,
        }


class SentenceTransformerSystem:
    """Dense retrieval with a pinned sentence-transformer and cosine ranking."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.name = f"dense-{model_name.rsplit('/', 1)[-1]}"
        self.model = None
        self.ids: List[str] = []
        self.matrix = None

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        self.model = SentenceTransformer(self.model_name)
        current = [
            item for item in dataset.corpus.values()
            if bool(item.metadata.get("is_current", True))
        ]
        self.ids = [item.id for item in current]
        texts = [_record_text(item, contextual=True) for item in current]
        self.matrix = self.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        elapsed = time.perf_counter() - started
        return IndexStats(elapsed, int(self.matrix.nbytes), len(self.ids))

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]:
        import numpy as np

        if self.model is None or self.matrix is None:
            raise RuntimeError("build must be called before rank")
        vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0].astype(np.float32)
        scores = self.matrix @ vector
        if len(scores) <= k:
            order = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, k - 1)[:k]
            order = candidates[np.argsort(-scores[candidates])]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order[:k]]

    def close(self) -> None:
        self.model = None
        self.matrix = None

    def metadata(self) -> Dict[str, object]:
        revision = None
        if self.model is not None:
            try:
                revision = self.model[0].auto_model.config._commit_hash
            except (AttributeError, IndexError, KeyError, TypeError):
                revision = None
        return {"model": self.model_name, "model_revision": revision}


class RRFSystem:
    """Reciprocal-rank fusion over two independently ranked systems."""

    def __init__(
        self,
        first: BenchmarkSystem,
        second: BenchmarkSystem,
        *,
        rrf_k: int = 60,
        candidate_k: int = 100,
        name: str = "rrf",
    ) -> None:
        self.first = first
        self.second = second
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.name = name

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        one = self.first.build(dataset)
        two = self.second.build(dataset)
        return IndexStats(one.seconds + two.seconds, one.bytes + two.bytes, max(one.records, two.records))

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for ranking in (
            self.first.rank(query, self.candidate_k),
            self.second.rank(query, self.candidate_k),
        ):
            for rank, (doc_id, _score) in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]

    def close(self) -> None:
        self.first.close()
        self.second.close()

    def metadata(self) -> Dict[str, object]:
        def child(system):
            metadata = getattr(system, "metadata", None)
            return metadata() if callable(metadata) else {"name": system.name}

        return {
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "first": child(self.first),
            "second": child(self.second),
        }


class LexicalFallbackSystem:
    """Use dense retrieval when available and BM25 if it raises at query time."""

    def __init__(self, semantic: BenchmarkSystem, lexical: BenchmarkSystem) -> None:
        self.semantic = semantic
        self.lexical = lexical
        self.name = f"fallback-{semantic.name}-to-{lexical.name}"

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        semantic = self.semantic.build(dataset)
        lexical = self.lexical.build(dataset)
        return IndexStats(
            semantic.seconds + lexical.seconds,
            semantic.bytes + lexical.bytes,
            max(semantic.records, lexical.records),
        )

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]:
        try:
            return self.semantic.rank(query, k)
        except Exception:
            return self.lexical.rank(query, k)

    def close(self) -> None:
        self.semantic.close()
        self.lexical.close()

    def metadata(self) -> Dict[str, object]:
        return {"semantic": self.semantic.name, "lexical": self.lexical.name}


class ForcedQueryFailureSystem:
    """Test double used to verify semantic-provider query failure behavior."""

    name = "forced-semantic-failure"

    def build(self, dataset: BenchmarkDataset) -> IndexStats:
        return IndexStats(0.0, 0, len(dataset.corpus))

    def rank(self, query: str, k: int) -> List[Tuple[str, float]]:
        raise RuntimeError("forced semantic provider failure")

    def close(self) -> None:
        return None

