"""SQLite implementation of the catalog (the default backend).

Single-file, zero-setup, and concurrent-read friendly. The schema mirrors the
production data model (documents / versions / chunks / sections / membership)
so migrating to Postgres or Snowflake is a backend swap rather than a redesign.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import List, Optional

from ..models import Chunk, Document, Section, Version

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_id TEXT,
    uri TEXT,
    title TEXT,
    media_type TEXT,
    size INTEGER,
    created_at REAL,
    updated_at REAL,
    is_deleted INTEGER DEFAULT 0,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS versions (
    doc_id TEXT,
    version_id TEXT,
    content_hash TEXT,
    size INTEGER,
    is_current INTEGER DEFAULT 1,
    created_at REAL,
    summary TEXT,
    keywords TEXT,
    parsed INTEGER DEFAULT 0,
    summarized INTEGER DEFAULT 0,
    chunked INTEGER DEFAULT 0,
    embedded INTEGER DEFAULT 0,
    PRIMARY KEY (doc_id, version_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    version_id TEXT,
    idx INTEGER,
    text TEXT,
    heading TEXT,
    location TEXT,
    token_count INTEGER,
    chunk_hash TEXT
);
CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    name TEXT UNIQUE,
    description TEXT,
    parent_id TEXT,
    status TEXT DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS membership (
    doc_id TEXT,
    section_id TEXT,
    confidence REAL,
    rationale TEXT,
    PRIMARY KEY (doc_id, section_id)
);
CREATE INDEX IF NOT EXISTS idx_versions_doc ON versions(doc_id, is_current);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, version_id);
CREATE INDEX IF NOT EXISTS idx_membership_section ON membership(section_id);
"""


class SQLiteCatalog:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self.init()

    def init(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # --- documents ---
    def upsert_document(self, doc: Document) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO documents
                (doc_id, source_id, uri, title, media_type, size, created_at,
                 updated_at, is_deleted, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_id=excluded.source_id, uri=excluded.uri,
                    title=excluded.title, media_type=excluded.media_type,
                    size=excluded.size, updated_at=excluded.updated_at,
                    is_deleted=excluded.is_deleted, metadata=excluded.metadata
                """,
                (
                    doc.doc_id, doc.source_id, doc.uri, doc.title, doc.media_type,
                    doc.size, doc.created_at, doc.updated_at,
                    1 if doc.is_deleted else 0, json.dumps(doc.metadata),
                ),
            )
            self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def all_documents(self, include_deleted: bool = False) -> List[Document]:
        sql = "SELECT * FROM documents"
        if not include_deleted:
            sql += " WHERE is_deleted=0"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def mark_deleted(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE documents SET is_deleted=1 WHERE doc_id=?", (doc_id,)
            )
            self._conn.execute(
                "UPDATE versions SET is_current=0 WHERE doc_id=?", (doc_id,)
            )
            self._conn.commit()

    # --- versions ---
    def upsert_version(self, version: Version) -> None:
        with self._lock:
            if version.is_current:
                self._conn.execute(
                    "UPDATE versions SET is_current=0 WHERE doc_id=?",
                    (version.doc_id,),
                )
            self._conn.execute(
                """INSERT INTO versions
                (doc_id, version_id, content_hash, size, is_current, created_at,
                 summary, keywords, parsed, summarized, chunked, embedded)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id, version_id) DO UPDATE SET
                    content_hash=excluded.content_hash, size=excluded.size,
                    is_current=excluded.is_current, summary=excluded.summary,
                    keywords=excluded.keywords, parsed=excluded.parsed,
                    summarized=excluded.summarized, chunked=excluded.chunked,
                    embedded=excluded.embedded
                """,
                (
                    version.doc_id, version.version_id, version.content_hash,
                    version.size, 1 if version.is_current else 0, version.created_at,
                    version.summary, json.dumps(version.keywords),
                    int(version.parsed), int(version.summarized),
                    int(version.chunked), int(version.embedded),
                ),
            )
            self._conn.commit()

    def get_current_version(self, doc_id: str) -> Optional[Version]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM versions WHERE doc_id=? AND is_current=1 LIMIT 1",
                (doc_id,),
            ).fetchone()
        return self._row_to_version(row) if row else None

    def get_version(self, doc_id: str, version_id: str) -> Optional[Version]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM versions WHERE doc_id=? AND version_id=?",
                (doc_id, version_id),
            ).fetchone()
        return self._row_to_version(row) if row else None

    def set_version_flags(self, doc_id: str, version_id: str, **flags) -> None:
        allowed = {"parsed", "summarized", "chunked", "embedded", "summary", "keywords"}
        sets = []
        params = []
        for key, val in flags.items():
            if key not in allowed:
                continue
            if key == "keywords" and isinstance(val, list):
                val = json.dumps(val)
            elif key in {"parsed", "summarized", "chunked", "embedded"}:
                val = int(bool(val))
            sets.append(f"{key}=?")
            params.append(val)
        if not sets:
            return
        params.extend([doc_id, version_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE versions SET {', '.join(sets)} WHERE doc_id=? AND version_id=?",
                params,
            )
            self._conn.commit()

    # --- chunks ---
    def save_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            doc_id = chunks[0].doc_id
            version_id = chunks[0].version_id
            self._conn.execute(
                "DELETE FROM chunks WHERE doc_id=? AND version_id=?",
                (doc_id, version_id),
            )
            self._conn.executemany(
                """INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, version_id, idx, text, heading, location,
                 token_count, chunk_hash)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        c.chunk_id, c.doc_id, c.version_id, c.index, c.text,
                        c.heading, c.location, c.token_count, c.chunk_hash,
                    )
                    for c in chunks
                ],
            )
            self._conn.commit()

    def get_chunks(self, doc_id: str, version_id: str) -> List[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE doc_id=? AND version_id=? ORDER BY idx",
                (doc_id, version_id),
            ).fetchall()
        return [
            Chunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], version_id=r["version_id"],
                index=r["idx"], text=r["text"], heading=r["heading"],
                location=r["location"], token_count=r["token_count"] or 0,
                chunk_hash=r["chunk_hash"] or "",
            )
            for r in rows
        ]

    # --- sections ---
    def upsert_section(self, section: Section) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO sections (section_id, name, description, parent_id, status)
                VALUES (?,?,?,?,?)
                ON CONFLICT(section_id) DO UPDATE SET
                    name=excluded.name, description=excluded.description,
                    parent_id=excluded.parent_id, status=excluded.status
                """,
                (section.section_id, section.name, section.description,
                 section.parent_id, section.status),
            )
            self._conn.commit()

    def all_sections(self) -> List[Section]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sections").fetchall()
        return [
            Section(
                section_id=r["section_id"], name=r["name"],
                description=r["description"] or "", parent_id=r["parent_id"],
                status=r["status"] or "active",
            )
            for r in rows
        ]

    def assign_membership(
        self, doc_id: str, section_id: str, confidence: float = 1.0, rationale: str = ""
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO membership
                (doc_id, section_id, confidence, rationale) VALUES (?,?,?,?)""",
                (doc_id, section_id, confidence, rationale),
            )
            self._conn.commit()

    def sections_for_doc(self, doc_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT section_id FROM membership WHERE doc_id=?", (doc_id,)
            ).fetchall()
        return [r["section_id"] for r in rows]

    def docs_for_section(self, section_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id FROM membership WHERE section_id=?", (section_id,)
            ).fetchall()
        return [r["doc_id"] for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.commit()
            self._conn.close()
            self._closed = True

    # --- row mappers ---
    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"], source_id=row["source_id"] or "",
            uri=row["uri"] or "", title=row["title"] or "",
            media_type=row["media_type"] or "", size=row["size"] or 0,
            created_at=row["created_at"] or 0.0, updated_at=row["updated_at"] or 0.0,
            is_deleted=bool(row["is_deleted"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> Version:
        return Version(
            doc_id=row["doc_id"], version_id=row["version_id"],
            content_hash=row["content_hash"] or "", size=row["size"] or 0,
            is_current=bool(row["is_current"]), created_at=row["created_at"] or 0.0,
            summary=row["summary"] or "",
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            parsed=bool(row["parsed"]), summarized=bool(row["summarized"]),
            chunked=bool(row["chunked"]), embedded=bool(row["embedded"]),
        )
