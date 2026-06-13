"""Core data model for the Librarian knowledge layer.

These dataclasses are intentionally backend-agnostic. Every storage adapter
(catalog + vector store) reads and writes these shapes, so swapping SQLite for
Postgres or a numpy index for FAISS never changes the contracts the rest of the
system relies on.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


# Document type tags used across the catalog and vector store. These mirror the
# "doc_type" facet that production deployments filter on (file_summary,
# section_rollup, chunk, ...) so retrieval can prefer summaries first and fall
# back to chunks only when depth is required.
DOC_TYPE_SUMMARY = "file_summary"
DOC_TYPE_CHUNK = "chunk"
DOC_TYPE_FOLDER_ROLLUP = "folder_rollup"
DOC_TYPE_SECTION_ROLLUP = "section_rollup"


@dataclass
class Document:
    """A logical document. Identity is stable for the life of the document."""

    doc_id: str
    source_id: str
    uri: str
    title: str = ""
    media_type: str = ""
    size: int = 0
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    is_deleted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Version:
    """An immutable edition of a document. New content -> new version."""

    doc_id: str
    version_id: str
    content_hash: str = ""
    size: int = 0
    is_current: bool = True
    created_at: float = field(default_factory=_now)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    # Processing lifecycle flags so builds are incremental and idempotent.
    parsed: bool = False
    summarized: bool = False
    chunked: bool = False
    embedded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Block:
    """A located span of text produced by a reader (page, slide, heading...)."""

    type: str
    text: str
    location: str = ""


@dataclass
class ParsedDocument:
    """Structured output of a reader: ordered located blocks + signals."""

    doc_id: str
    title: str
    uri: str
    blocks: List[Block] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)


@dataclass
class Chunk:
    """A retrievable unit of a version. Carries provenance for citations."""

    chunk_id: str
    doc_id: str
    version_id: str
    index: int
    text: str
    heading: Optional[str] = None
    location: Optional[str] = None
    token_count: int = 0
    chunk_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Section:
    """A virtual "shelf": a logical grouping that never moves the source bytes."""

    section_id: str
    name: str
    description: str = ""
    parent_id: Optional[str] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Rollup:
    """A synthesized catalog entry for a folder or section."""

    rollup_id: str
    kind: str  # "folder" | "section"
    key: str  # folder path or section_id
    text: str
    doc_type: str = DOC_TYPE_FOLDER_ROLLUP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """A single retrieval result, shaped for direct injection into a prompt."""

    doc_id: str
    title: str
    uri: str
    excerpt: str
    score: float
    doc_type: str = DOC_TYPE_SUMMARY
    version_id: str = ""
    source_id: str = ""
    location: str = ""
    section_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def citation(self) -> str:
        loc = f" ({self.location})" if self.location else ""
        title = self.title or self.uri or self.doc_id
        return f"{title}{loc}"
