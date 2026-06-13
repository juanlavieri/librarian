"""Configuration for a Librarian instance.

Defaults are chosen so the system runs with **zero external dependencies and no
API keys**: a SQLite catalog, a local vector store, and a deterministic hashing
embedder. Every one of these is swappable for a production backend (OpenAI
embeddings, FAISS, pgvector, ...) without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LibrarianConfig:
    # Where on-disk artifacts live (catalog db + vector index).
    root: str = "./librarian_data"

    # --- Chunking ---
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 80

    # --- Summarization ---
    summarize: bool = True
    summary_max_chars: int = 1200
    # Characters of source text fed to the summarizer per document.
    summary_input_chars: int = 8000

    # --- Retrieval ---
    default_k: int = 8
    # Prefer summaries; only deepen into chunks when the question needs detail
    # or summary confidence is low.
    enable_chunk_fallback: bool = True
    chunk_fallback_score_threshold: float = 0.30
    # Blend weights for hybrid scoring (semantic vs lexical).
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    # Prefer current editions; set True to also surface archived versions.
    include_archived: bool = False

    # --- Crawl (optional) ---
    crawl_max_pages: int = 100
    crawl_max_depth: int = 6

    # --- Backends (resolved lazily by Librarian) ---
    embedding_provider: str = "hashing"  # "hashing" | "openai" | custom
    embedding_dim: int = 512  # used by the hashing embedder
    embedding_model: str = "text-embedding-3-small"  # used by openai provider
    catalog_backend: str = "sqlite"
    vector_backend: str = "local"  # "local" | "faiss"
    summarizer_provider: str = "extractive"  # "extractive" | "openai"
    summarizer_model: str = "gpt-4o-mini"

    # Free-form extras for custom backends.
    extras: dict = field(default_factory=dict)

    openai_api_key: Optional[str] = None
