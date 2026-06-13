"""Search index contract.

The vector store is the Librarian's *search index*: a denormalized set of
records (summaries, rollups, chunks) that each carry their own text, provenance
facets, and an embedding. This mirrors how production deployments use a
dedicated search service alongside a canonical catalog -- retrieval hits the
index, while the catalog remains the source of truth. Both semantic (vector)
and lexical (token) search run over the same records so a single store backs
hybrid retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple


@dataclass
class IndexRecord:
    id: str
    doc_type: str
    doc_id: str
    text: str
    title: str = ""
    uri: str = ""
    source_id: str = ""
    version_id: str = ""
    location: str = ""
    is_current: bool = True
    section_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    vector: List[float] = field(default_factory=list)


@dataclass
class Filters:
    doc_types: Optional[List[str]] = None
    is_current: Optional[bool] = True
    doc_ids: Optional[List[str]] = None
    source_ids: Optional[List[str]] = None

    def matches(self, rec: IndexRecord) -> bool:
        if self.doc_types is not None and rec.doc_type not in self.doc_types:
            return False
        if self.is_current is not None and rec.is_current != self.is_current:
            return False
        if self.doc_ids is not None and rec.doc_id not in self.doc_ids:
            return False
        if self.source_ids is not None and rec.source_id not in self.source_ids:
            return False
        return True


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class VectorStore(Protocol):
    def upsert(self, records: List[IndexRecord]) -> None: ...

    def delete_by_doc(self, doc_id: str) -> None: ...

    def delete_by_doc_version(self, doc_id: str, version_id: str) -> None: ...

    def archive_doc(self, doc_id: str) -> None: ...

    def search_semantic(
        self, vector: List[float], k: int, filters: Optional[Filters] = None
    ) -> List[Tuple[IndexRecord, float]]: ...

    def search_lexical(
        self, query: str, k: int, filters: Optional[Filters] = None
    ) -> List[Tuple[IndexRecord, float]]: ...

    def all_records(self) -> List[IndexRecord]: ...

    def persist(self) -> None: ...

    def close(self) -> None: ...
