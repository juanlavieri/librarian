"""Heading- and location-aware chunking.

Chunks are the deep-retrieval unit used only when summaries are not enough.
Splitting respects block boundaries (pages/slides/sheets), carries the nearest
heading, and overlaps consecutive chunks so a fact that straddles a boundary is
not lost. Token counting uses ``tiktoken`` when available and a whitespace
approximation otherwise.
"""

from __future__ import annotations

from typing import List, Optional

from .identity import chunk_id_for, text_hash
from .models import Chunk, ParsedDocument


def _encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENCODER = _encoder()


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Rough heuristic: ~0.75 words per token in English prose.
    return max(1, int(len(text.split()) / 0.75))


def _tail_tokens(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    if _ENCODER is not None:
        toks = _ENCODER.encode(text)
        return _ENCODER.decode(toks[-overlap_tokens:])
    words = text.split()
    approx_words = int(overlap_tokens * 0.75)
    return " ".join(words[-approx_words:]) if approx_words else ""


def _split_oversized(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split a single block that is larger than ``max_tokens`` into windows.

    Word-based windowing (token-approximate) with overlap, so a long page or a
    monolithic text file is still broken into retrievable units.
    """
    words = text.split()
    if not words:
        return []
    # Approximate words-per-window from the token budget.
    win = max(1, int(max_tokens * 0.75))
    step = max(1, win - int(overlap_tokens * 0.75))
    pieces: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        pieces.append(" ".join(words[i : i + win]))
        if i + win >= n:
            break
        i += step
    return pieces


def _select_heading(block_type: str, text: str) -> Optional[str]:
    bt = (block_type or "").lower()
    if bt in {"heading", "title"} and text.strip():
        return text.strip().splitlines()[0][:200]
    if bt in {"slide", "sheet", "table", "page"}:
        return bt.title()
    return None


def chunk_document(
    parsed: ParsedDocument,
    version_id: str,
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
) -> List[Chunk]:
    """Split a parsed document into overlapping, located chunks."""
    chunks: List[Chunk] = []
    parts: List[str] = []
    locations: List[str] = []
    heading: Optional[str] = None
    running_tokens = 0
    index = 0

    def flush() -> None:
        nonlocal parts, locations, heading, running_tokens, index
        if not parts:
            return
        text = "\n\n".join(p for p in parts if p).strip()
        if not text:
            parts, locations, heading, running_tokens = [], [], None, 0
            return
        first = locations[0] if locations else ""
        last = locations[-1] if locations else ""
        location = first if first == last else f"{first} - {last}".strip(" -")
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(parsed.doc_id, version_id, index),
                doc_id=parsed.doc_id,
                version_id=version_id,
                index=index,
                text=text,
                heading=heading,
                location=location or None,
                token_count=count_tokens(text),
                chunk_hash=text_hash(text),
            )
        )
        index += 1
        carry = _tail_tokens(text, overlap_tokens)
        parts = [carry] if carry else []
        locations = [last] if last else []
        running_tokens = count_tokens(carry) if carry else 0
        heading = None

    # Expand any block larger than the budget into token-bounded sub-blocks so a
    # single huge block (e.g. a whole text file) still produces multiple chunks.
    expanded: List[tuple] = []  # (type, text, location)
    for block in parsed.blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        if count_tokens(text) > max_tokens:
            for piece in _split_oversized(text, max_tokens, overlap_tokens):
                expanded.append((block.type, piece, block.location or ""))
        else:
            expanded.append((block.type, text, block.location or ""))

    for block_type, text, location in expanded:
        block_tokens = count_tokens(text)
        if parts and running_tokens + block_tokens > max_tokens:
            flush()
        h = _select_heading(block_type, text)
        if h and not heading:
            heading = h
        parts.append(text)
        locations.append(location)
        running_tokens += block_tokens

    flush()
    return chunks
