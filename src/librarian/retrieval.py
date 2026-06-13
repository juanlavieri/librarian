"""Hybrid retrieval: the read path of the Librarian.

Combines four signals the way an expert navigates a repository:

1. **Direct path** -- if the query names a file/URL, surface it.
2. **Semantic** -- vector similarity over summaries (and chunks on fallback).
3. **Lexical** -- term overlap, for names, codes, and rare keywords.
4. **Structural** -- folder/section roll-ups that explain *where* things live.

It prefers current editions and summaries, then deepens into chunks only when
the question demands detail or summary confidence is low (the "summary-first,
chunk-fallback" strategy). Results come back as :class:`Evidence` with
provenance for citation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .config import LibrarianConfig
from .models import (
    DOC_TYPE_CHUNK,
    DOC_TYPE_FOLDER_ROLLUP,
    DOC_TYPE_SECTION_ROLLUP,
    DOC_TYPE_SUMMARY,
    Evidence,
)
from .vectorstore.base import Filters, IndexRecord, VectorStore

_DETAIL_TRIGGERS = (
    "according to", "quote", "exact", "page", "slide", "table", "section",
    "where does", "what does", "percent", "%", "how many", "how much",
    "how common", "what share", "market share", "breakdown", "statistic",
    "rate", "number of", "list all", "step by step", "verbatim", "figure",
)


def _blend(semantic: float, lexical: float, cfg: LibrarianConfig) -> float:
    return cfg.semantic_weight * semantic + cfg.lexical_weight * lexical


def _record_to_evidence(rec: IndexRecord, score: float, excerpt_chars: int = 800) -> Evidence:
    return Evidence(
        doc_id=rec.doc_id,
        title=rec.title,
        uri=rec.uri,
        excerpt=(rec.text or "")[:excerpt_chars],
        score=round(float(score), 4),
        doc_type=rec.doc_type,
        version_id=rec.version_id,
        source_id=rec.source_id,
        location=rec.location,
        section_ids=list(rec.section_ids),
        tags=list(rec.tags),
    )


def _wants_detail(query: str) -> bool:
    q = query.lower()
    return any(trigger in q for trigger in _DETAIL_TRIGGERS)


class Retriever:
    def __init__(self, store: VectorStore, embedder, cfg: LibrarianConfig) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def _hybrid(
        self, query: str, k: int, filters: Filters
    ) -> List[tuple]:
        qvec = self.embedder.embed_one(query)
        semantic = self.store.search_semantic(qvec, k * 3, filters)
        lexical = self.store.search_lexical(query, k * 3, filters)

        merged: Dict[str, Dict] = {}
        for rec, score in semantic:
            merged.setdefault(rec.id, {"rec": rec, "sem": 0.0, "lex": 0.0})["sem"] = score
        for rec, score in lexical:
            merged.setdefault(rec.id, {"rec": rec, "sem": 0.0, "lex": 0.0})["lex"] = score

        scored = [
            (m["rec"], _blend(m["sem"], m["lex"], self.cfg))
            for m in merged.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        # Dedupe to one record per document, keeping the best score.
        best: Dict[str, tuple] = {}
        for rec, score in scored:
            key = rec.doc_id if rec.doc_type != DOC_TYPE_CHUNK else rec.id
            if key not in best or score > best[key][1]:
                best[key] = (rec, score)
        ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        include_rollups: bool = True,
        include_chunks: Optional[bool] = None,
        source_ids: Optional[List[str]] = None,
        include_archived: Optional[bool] = None,
    ) -> List[Evidence]:
        query = (query or "").strip()
        if not query:
            return []
        k = k or self.cfg.default_k
        archived = self.cfg.include_archived if include_archived is None else include_archived

        doc_types = [DOC_TYPE_SUMMARY]
        if include_rollups:
            doc_types += [DOC_TYPE_FOLDER_ROLLUP, DOC_TYPE_SECTION_ROLLUP]

        filters = Filters(
            doc_types=doc_types,
            is_current=None if archived else True,
            source_ids=source_ids,
        )
        summary_hits = self._hybrid(query, k, filters)

        deepen = include_chunks
        if deepen is None:
            deepen = self.cfg.enable_chunk_fallback and self._should_deepen(query, summary_hits)

        if not deepen:
            return [_record_to_evidence(rec, score) for rec, score in summary_hits]

        # Deepen: pull chunk-level evidence from the top documents.
        top_doc_ids = [
            rec.doc_id for rec, _ in summary_hits if rec.doc_type == DOC_TYPE_SUMMARY
        ][:3]
        chunk_filters = Filters(
            doc_types=[DOC_TYPE_CHUNK],
            is_current=None if archived else True,
            doc_ids=top_doc_ids or None,
        )
        chunk_hits = self._hybrid(query, k, chunk_filters)
        if not chunk_hits:
            return [_record_to_evidence(rec, score) for rec, score in summary_hits]

        # Merge: chunk evidence first (it is more specific), then unique summaries.
        evidence = [_record_to_evidence(rec, score) for rec, score in chunk_hits]
        seen = {e.doc_id for e in evidence}
        for rec, score in summary_hits:
            if rec.doc_id not in seen:
                evidence.append(_record_to_evidence(rec, score))
        return evidence[:k]

    def _should_deepen(self, query: str, summary_hits: List[tuple]) -> bool:
        if not summary_hits:
            return False
        if _wants_detail(query):
            return True
        top_score = summary_hits[0][1] if summary_hits else 0.0
        return top_score < self.cfg.chunk_fallback_score_threshold
