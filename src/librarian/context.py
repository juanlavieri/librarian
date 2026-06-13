"""Context enrichment for chunks (a.k.a. contextual retrieval).

A raw chunk pulled from the middle of a document is ambiguous: "it grew 12%
year over year" is useless without knowing *what* grew and *which* document it
came from. Before a chunk is embedded and indexed, the Librarian prepends a
compact context header derived from the document title, its folder, its
inferred subject, and the nearest heading. This dramatically improves retrieval
precision and curbs hallucination, because every retrievable unit carries the
context required to interpret it correctly.
"""

from __future__ import annotations

from typing import Optional

from .models import Chunk


def build_context_header(
    *,
    title: str,
    folder: str,
    asset_context: str = "",
    doc_summary: str = "",
    heading: Optional[str] = None,
) -> str:
    parts = []
    if title:
        parts.append(f"Document: {title}")
    if folder and folder not in (".", "/"):
        parts.append(f"Location: {folder}")
    if heading:
        parts.append(f"Section: {heading}")
    if asset_context:
        parts.append(f"About: {asset_context[:240]}")
    elif doc_summary:
        parts.append(f"About: {doc_summary[:240]}")
    return " | ".join(parts)


def contextualize_chunk(
    chunk: Chunk,
    *,
    title: str,
    folder: str,
    asset_context: str = "",
    doc_summary: str = "",
) -> str:
    """Return the chunk text augmented with a one-line context header."""
    header = build_context_header(
        title=title,
        folder=folder,
        asset_context=asset_context,
        doc_summary=doc_summary,
        heading=chunk.heading,
    )
    if not header:
        return chunk.text
    return f"[{header}]\n{chunk.text}"


def enrich_summary(
    *,
    title: str,
    folder: str,
    asset_context: str,
    summary: str,
    keywords: list,
) -> str:
    """Compose the searchable summary record from all available signals."""
    lines = []
    if title:
        lines.append(f"Title: {title}")
    if folder and folder not in (".", "/"):
        lines.append(f"Location: {folder}")
    if asset_context:
        lines.append(asset_context)
    if summary:
        lines.append(summary)
    if keywords:
        lines.append("Keywords: " + ", ".join(keywords))
    return "\n".join(lines).strip()
