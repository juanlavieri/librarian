"""Deterministic identity for documents, versions, and chunks.

Stable identity is what lets the Librarian organize a knowledge base
*virtually* -- the same source always maps to the same ``doc_id``, and a change
in content always produces a new immutable ``version_id``. Nothing is ever moved
or renamed in storage; only metadata and membership change.
"""

from __future__ import annotations

import hashlib


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_for(source_id: str, uri: str) -> str:
    """Canonical, stable identity for a document.

    Derived from the source namespace + the resource locator (path or URL) so
    re-ingesting the same source never creates a duplicate document.
    """
    return _sha256(f"{source_id}::{uri}".encode("utf-8"))


def version_id_for(content: bytes, *, etag: str = "") -> str:
    """Immutable identity for a specific edition of a document.

    Prefers a provided ``etag`` (e.g. from SharePoint/Graph or an HTTP header)
    and otherwise hashes the content bytes.
    """
    if etag:
        return _sha256(etag.encode("utf-8"))
    return _sha256(content)


def content_hash(content: bytes) -> str:
    return _sha256(content)


def chunk_id_for(doc_id: str, version_id: str, index: int) -> str:
    digest = _sha256(f"{doc_id}::{version_id}".encode("utf-8"))
    return f"chunk_{digest}_{index:04d}"


def text_hash(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def stable_key(value: str) -> str:
    """Hash an arbitrary string into a stable id (used for sections/rollups)."""
    return _sha256(value.encode("utf-8"))
