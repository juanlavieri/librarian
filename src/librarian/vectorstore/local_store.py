"""Local vector store: pure-Python, persisted as JSON.

Holds records in memory and snapshots them to a JSON file. Uses numpy to
accelerate semantic search when it is installed and falls back to a pure-Python
cosine loop otherwise. Suitable for development and for knowledge bases up to
the low hundreds of thousands of records; point ``vector_backend="faiss"`` (or
a custom store) at larger corpora.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from .base import Filters, IndexRecord, VectorStore, cosine

try:  # optional acceleration
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_\-]*\b")


class LocalVectorStore(VectorStore):
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._records: Dict[str, IndexRecord] = {}
        self._np_matrix = None
        self._np_ids: List[str] = []
        self._dirty_matrix = True
        if path and os.path.exists(path):
            self._load()

    # --- writes ---
    def upsert(self, records: List[IndexRecord]) -> None:
        for rec in records:
            self._records[rec.id] = rec
        self._dirty_matrix = True

    def delete_by_doc(self, doc_id: str) -> None:
        to_drop = [rid for rid, r in self._records.items() if r.doc_id == doc_id]
        for rid in to_drop:
            del self._records[rid]
        self._dirty_matrix = True

    # --- reads ---
    def all_records(self) -> List[IndexRecord]:
        return list(self._records.values())

    def _candidates(self, filters: Optional[Filters]) -> List[IndexRecord]:
        if filters is None:
            return list(self._records.values())
        return [r for r in self._records.values() if filters.matches(r)]

    def search_semantic(
        self, vector: List[float], k: int, filters: Optional[Filters] = None
    ) -> List[Tuple[IndexRecord, float]]:
        if not vector:
            return []
        candidates = self._candidates(filters)
        if not candidates:
            return []
        if _np is not None:
            return self._search_semantic_np(vector, k, candidates)
        scored = [
            (rec, cosine(vector, rec.vector)) for rec in candidates if rec.vector
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _search_semantic_np(self, vector, k, candidates):
        vecs = [r for r in candidates if r.vector]
        if not vecs:
            return []
        mat = _np.array([r.vector for r in vecs], dtype="float32")
        q = _np.array(vector, dtype="float32")
        # L2-normalize both sides so the dot product is cosine similarity.
        mat_norm = _np.linalg.norm(mat, axis=1, keepdims=True)
        mat_norm[mat_norm == 0] = 1.0
        mat = mat / mat_norm
        qn = _np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        sims = mat @ q
        order = _np.argsort(-sims)[:k]
        return [(vecs[i], float(sims[i])) for i in order]

    def search_lexical(
        self, query: str, k: int, filters: Optional[Filters] = None
    ) -> List[Tuple[IndexRecord, float]]:
        q_tokens = set(t.lower() for t in _TOKEN_RE.findall(query or ""))
        if not q_tokens:
            return []
        candidates = self._candidates(filters)
        scored: List[Tuple[IndexRecord, float]] = []
        for rec in candidates:
            haystack = f"{rec.title}\n{rec.text}\n{' '.join(rec.tags)}".lower()
            doc_tokens = set(_TOKEN_RE.findall(haystack))
            if not doc_tokens:
                continue
            overlap = q_tokens & doc_tokens
            if not overlap:
                continue
            # Jaccard-ish recall of the query terms, with a phrase bonus.
            score = len(overlap) / len(q_tokens)
            if query.lower() in haystack:
                score += 0.25
            scored.append((rec, min(score, 1.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # --- persistence ---
    def persist(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        payload = {"records": [self._record_to_dict(r) for r in self._records.values()]}
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, self.path)

    def close(self) -> None:
        self.persist()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        for raw in payload.get("records", []):
            rec = IndexRecord(**raw)
            self._records[rec.id] = rec

    @staticmethod
    def _record_to_dict(rec: IndexRecord) -> dict:
        return {
            "id": rec.id,
            "doc_type": rec.doc_type,
            "doc_id": rec.doc_id,
            "text": rec.text,
            "title": rec.title,
            "uri": rec.uri,
            "source_id": rec.source_id,
            "version_id": rec.version_id,
            "location": rec.location,
            "is_current": rec.is_current,
            "section_ids": rec.section_ids,
            "tags": rec.tags,
            "vector": rec.vector,
        }
