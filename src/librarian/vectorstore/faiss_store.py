"""FAISS-backed vector store for larger corpora (optional).

Keeps record metadata in memory (and on disk as JSON) while delegating
approximate nearest-neighbor search to a FAISS index. Lexical search reuses the
in-memory records. Install the ``faiss`` extra to use this backend
(``vector_backend="faiss"``).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .base import Filters, IndexRecord, VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, dim: int, path: Optional[str] = None) -> None:
        import faiss  # type: ignore
        import numpy as np  # type: ignore

        self._faiss = faiss
        self._np = np
        self.dim = dim
        self.path = path
        self.meta_path = f"{path}.meta.json" if path else None
        self._records: Dict[str, IndexRecord] = {}
        self._index = faiss.IndexFlatIP(dim)
        self._id_order: List[str] = []
        self._needs_rebuild = False
        if path and os.path.exists(self.meta_path):
            self._load()

    def upsert(self, records: List[IndexRecord]) -> None:
        for rec in records:
            self._records[rec.id] = rec
        self._needs_rebuild = True

    def delete_by_doc(self, doc_id: str) -> None:
        drop = [rid for rid, r in self._records.items() if r.doc_id == doc_id]
        for rid in drop:
            del self._records[rid]
        self._needs_rebuild = True

    def delete_by_doc_version(self, doc_id: str, version_id: str) -> None:
        drop = [
            rid for rid, r in self._records.items()
            if r.doc_id == doc_id and r.version_id == version_id
        ]
        for rid in drop:
            del self._records[rid]
        self._needs_rebuild = True

    def archive_doc(self, doc_id: str) -> None:
        for rec in self._records.values():
            if rec.doc_id == doc_id:
                rec.is_current = False
        self._needs_rebuild = True

    def _rebuild(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._id_order = []
        vectors = []
        for rid, rec in self._records.items():
            if not rec.vector:
                continue
            vectors.append(rec.vector)
            self._id_order.append(rid)
        if vectors:
            mat = self._np.array(vectors, dtype="float32")
            self._faiss.normalize_L2(mat)
            self._index.add(mat)
        self._needs_rebuild = False

    def search_semantic(
        self, vector: List[float], k: int, filters: Optional[Filters] = None
    ) -> List[Tuple[IndexRecord, float]]:
        if self._needs_rebuild:
            self._rebuild()
        if not self._id_order or not vector:
            return []
        q = self._np.array([vector], dtype="float32")
        self._faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, min(k * 4, len(self._id_order)))
        out: List[Tuple[IndexRecord, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            rec = self._records.get(self._id_order[idx])
            if rec is None:
                continue
            if filters is not None and not filters.matches(rec):
                continue
            out.append((rec, float(score)))
            if len(out) >= k:
                break
        return out

    def search_lexical(
        self, query: str, k: int, filters: Optional[Filters] = None
    ) -> List[Tuple[IndexRecord, float]]:
        from .local_store import LocalVectorStore

        proxy = LocalVectorStore.__new__(LocalVectorStore)
        proxy._records = self._records  # type: ignore[attr-defined]
        return LocalVectorStore.search_lexical(proxy, query, k, filters)

    def all_records(self) -> List[IndexRecord]:
        return list(self._records.values())

    def persist(self) -> None:
        if not self.meta_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.meta_path)), exist_ok=True)
        from .local_store import LocalVectorStore

        payload = {
            "dim": self.dim,
            "records": [LocalVectorStore._record_to_dict(r) for r in self._records.values()],
        }
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def _load(self) -> None:
        try:
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        self.dim = payload.get("dim", self.dim)
        for raw in payload.get("records", []):
            rec = IndexRecord(**raw)
            self._records[rec.id] = rec
        self._needs_rebuild = True

    def close(self) -> None:
        self.persist()
