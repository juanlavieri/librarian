"""The :class:`Librarian` -- the plug-and-play facade.

Two halves, one object, mirroring a real library:

* **The catalog** (write path): connectors crawl a store as deep as it goes;
  every asset is opened, profiled, summarized, chunked-with-context, and its
  metadata is rolled up recursively into its parent folders. The result is a
  logically organized, context-rich index of *what exists and what it is about*.
* **The librarian** (read path): an agentic, hybrid retriever that, given a
  question, navigates that catalog the way an expert librarian would -- by
  topic, structure, name, and meaning -- and returns highly relevant,
  citation-ready evidence fast.

Defaults run with zero dependencies and no API keys. Every backend (embedder,
summarizer, catalog, vector store) is swappable.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .catalog import SQLiteCatalog
from .catalog.base import Catalog
from .chunking import chunk_document
from .config import LibrarianConfig
from .connectors import Connector, FilesystemConnector, WebConnector
from .context import contextualize_chunk, enrich_summary
from .embeddings import HashingEmbedder
from .embeddings.base import Embedder
from .enrich import profile_document
from .identity import content_hash, doc_id_for, version_id_for
from .ingest import IntakeItem, from_text
from .keywords import extract_keywords
from .memory import ConversationMemory
from .models import (
    DOC_TYPE_CHUNK,
    DOC_TYPE_FOLDER_ROLLUP,
    DOC_TYPE_SECTION_ROLLUP,
    DOC_TYPE_SUMMARY,
    Document,
    Evidence,
    Section,
    Version,
)
from .retrieval import Retriever
from .rollups import (
    build_folder_rollups,
    build_section_rollup,
    folder_of,
)
from .summarize import ExtractiveSummarizer, Summarizer
from .vectorstore import IndexRecord, LocalVectorStore
from .vectorstore.base import VectorStore


class Librarian:
    def __init__(
        self,
        config: Optional[LibrarianConfig] = None,
        *,
        embedder: Optional[Embedder] = None,
        summarizer: Optional[Summarizer] = None,
        catalog: Optional[Catalog] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.config = config or LibrarianConfig()
        os.makedirs(self.config.root, exist_ok=True)
        self.embedder = embedder or self._default_embedder()
        self.summarizer = summarizer or self._default_summarizer()
        self.catalog = catalog or self._default_catalog()
        self.store = store or self._default_store()
        # Archival helpers are optional on the VectorStore protocol so custom
        # backends written against the original interface keep working. When a
        # store lacks them we fall back to delete-on-reindex (no archived-version
        # retrieval, but fully functional).
        self._store_archives = hasattr(self.store, "archive_doc") and hasattr(
            self.store, "delete_by_doc_version"
        )
        self.memory = ConversationMemory(summarizer=self.summarizer)
        self.retriever = Retriever(self.store, self.embedder, self.config)
        self._sources: List[Iterable[IntakeItem]] = []

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def open(cls, root: str = "./librarian_data", **overrides) -> "Librarian":
        cfg = LibrarianConfig(root=root, **overrides)
        return cls(cfg)

    def _default_embedder(self) -> Embedder:
        if self.config.embedding_provider == "openai":
            from .embeddings import OpenAIEmbedder

            return OpenAIEmbedder(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key,
            )
        return HashingEmbedder(dim=self.config.embedding_dim)

    def _default_summarizer(self) -> Summarizer:
        if self.config.summarizer_provider == "openai":
            from .summarize import OpenAISummarizer

            return OpenAISummarizer(
                model=self.config.summarizer_model,
                api_key=self.config.openai_api_key,
                max_chars=self.config.summary_max_chars,
                input_chars=self.config.summary_input_chars,
            )
        return ExtractiveSummarizer(max_chars=self.config.summary_max_chars)

    def _default_catalog(self) -> Catalog:
        return SQLiteCatalog(os.path.join(self.config.root, "catalog.db"))

    def _default_store(self) -> VectorStore:
        if self.config.vector_backend == "faiss":
            from .vectorstore.faiss_store import FaissVectorStore

            return FaissVectorStore(
                dim=self.embedder.dim,
                path=os.path.join(self.config.root, "index.faiss"),
            )
        return LocalVectorStore(os.path.join(self.config.root, "index.json"))

    # ------------------------------------------------------------------ #
    # ingestion (collect sources, processed lazily on build)
    # ------------------------------------------------------------------ #
    def add_path(self, path: str, *, source_id: str = "filesystem", recursive: bool = True) -> "Librarian":
        self._sources.append(
            FilesystemConnector(path, source_id=source_id, recursive=recursive).items()
        )
        return self

    def add_text(self, text: str, *, name: str, source_id: str = "manual", title: str = "") -> "Librarian":
        self._sources.append([from_text(text, name=name, source_id=source_id, title=title)])
        return self

    def add_url(self, url: str, *, source_id: str = "web", max_pages: Optional[int] = None) -> "Librarian":
        self._sources.append(
            WebConnector(
                url,
                source_id=source_id,
                max_pages=max_pages or self.config.crawl_max_pages,
                max_depth=self.config.crawl_max_depth,
            ).items()
        )
        return self

    def add_connector(self, connector: Connector) -> "Librarian":
        self._sources.append(connector.items())
        return self

    # ------------------------------------------------------------------ #
    # build (the write path)
    # ------------------------------------------------------------------ #
    def build(self, *, force: bool = False, rebuild_rollups: bool = True) -> dict:
        """Process all pending sources into the catalog + index.

        Incremental and idempotent: an unchanged asset (same content hash) is
        skipped. Returns counts for indexed / skipped / failed assets.
        """
        stats = {"indexed": 0, "skipped": 0, "failed": 0, "chunks": 0}
        for source in self._sources:
            # Iterate defensively: a connector itself can fail (e.g. an
            # optional dependency is missing or a remote store is unreachable).
            # One bad source/item must not abort the whole build.
            iterator = iter(source)
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                except Exception:
                    stats["failed"] += 1
                    break
                try:
                    result, n_chunks = self._index_item(item, force=force)
                    stats[result] = stats.get(result, 0) + 1
                    stats["chunks"] += n_chunks
                except Exception:
                    stats["failed"] += 1
        self._sources = []
        if rebuild_rollups:
            self._rebuild_rollups()
        self.store.persist()
        return stats

    def _index_item(self, item: IntakeItem, *, force: bool) -> tuple:
        doc_id = doc_id_for(item.source_id, item.uri)
        version_id = version_id_for(item.data)
        chash = content_hash(item.data)

        current = self.catalog.get_current_version(doc_id)
        if (
            not force
            and current is not None
            and current.content_hash == chash
            and current.embedded
        ):
            return "skipped", 0

        from .readers import parse_document

        parsed = parse_document(
            doc_id=doc_id, title=item.title, uri=item.uri,
            name=item.name, data=item.data,
        )
        text = parsed.text

        # 1. Open + profile the asset (metadata enrichment).
        profile = profile_document(parsed, media_type=item.media_type, summarizer=None)

        # 2. Summary-first understanding.
        if self.config.summarize and text:
            summary = self.summarizer.summarize(
                text[: self.config.summary_input_chars], title=item.title
            )
        else:
            summary = text[: self.config.summary_max_chars]
        keywords = extract_keywords(text)
        topics = profile.topics or keywords

        # Folder hierarchy is based on the *logical* path (relative path for
        # files, URL for web) so roll-ups follow the store's structure rather
        # than the absolute mount point.
        logical_path = item.uri if item.uri.startswith(("http://", "https://")) else item.name
        folder = folder_of(logical_path)

        # 3. Persist canonical document + version.
        metadata = dict(item.metadata)
        metadata["profile"] = profile.to_dict()
        metadata["modality"] = profile.modality
        metadata["folder"] = folder
        doc = Document(
            doc_id=doc_id, source_id=item.source_id, uri=item.uri,
            title=item.title, media_type=item.media_type, size=len(item.data),
            metadata=metadata,
        )
        self.catalog.upsert_document(doc)
        version = Version(
            doc_id=doc_id, version_id=version_id, content_hash=chash,
            size=len(item.data), is_current=True, summary=summary,
            keywords=keywords, parsed=True, summarized=True,
        )
        self.catalog.upsert_version(version)

        # Version handling: archive any prior records for this document (they
        # stay queryable via include_archived) and clear any stale records for
        # *this* version_id (e.g. a forced re-index), then add the new current
        # records below. Custom stores without the archival helpers fall back to
        # deleting the document's records (original behavior).
        if self._store_archives:
            self.store.archive_doc(doc_id)
            self.store.delete_by_doc_version(doc_id, version_id)
        else:
            self.store.delete_by_doc(doc_id)

        asset_ctx = profile.as_context()
        section_ids = self.catalog.sections_for_doc(doc_id)

        records: List[IndexRecord] = []
        # 4. Summary record (the primary retrieval unit).
        summary_text = enrich_summary(
            title=item.title, folder=folder, asset_context=asset_ctx,
            summary=summary, keywords=keywords,
        )
        records.append(
            IndexRecord(
                id=f"summary::{doc_id}::{version_id}",
                doc_type=DOC_TYPE_SUMMARY, doc_id=doc_id, text=summary_text,
                title=item.title, uri=item.uri, source_id=item.source_id,
                version_id=version_id, location="", is_current=True,
                section_ids=section_ids, tags=list(dict.fromkeys(topics + keywords))[:16],
            )
        )

        # 5. Context-enriched chunks (deep retrieval units).
        chunks = chunk_document(
            parsed, version_id,
            max_tokens=self.config.chunk_max_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
        )
        if chunks:
            self.catalog.save_chunks(chunks)
        for chunk in chunks:
            ctext = contextualize_chunk(
                chunk, title=item.title, folder=folder,
                asset_context=asset_ctx, doc_summary=summary,
            )
            records.append(
                IndexRecord(
                    id=chunk.chunk_id, doc_type=DOC_TYPE_CHUNK, doc_id=doc_id,
                    text=ctext, title=item.title, uri=item.uri,
                    source_id=item.source_id, version_id=version_id,
                    location=chunk.location or "", is_current=True,
                    section_ids=section_ids, tags=keywords[:8],
                )
            )

        # 6. Embed everything in one batch and upsert.
        vectors = self.embedder.embed([r.text for r in records])
        for rec, vec in zip(records, vectors):
            rec.vector = vec
        self.store.upsert(records)
        self.catalog.set_version_flags(doc_id, version_id, chunked=True, embedded=True)
        return "indexed", len(chunks)

    # ------------------------------------------------------------------ #
    # roll-ups + sections (virtual organization)
    # ------------------------------------------------------------------ #
    def _current_doc_items(self) -> List[dict]:
        items = []
        for doc in self.catalog.all_documents():
            version = self.catalog.get_current_version(doc.doc_id)
            items.append(
                {
                    "doc_id": doc.doc_id,
                    "uri": doc.uri,
                    "folder": doc.metadata.get("folder") or folder_of(doc.uri),
                    "title": doc.title,
                    "summary": version.summary if version else "",
                }
            )
        return items

    def _rebuild_rollups(self) -> None:
        # Drop existing roll-up records (their doc_id == rollup_id).
        for rec in self.store.all_records():
            if rec.doc_type in (DOC_TYPE_FOLDER_ROLLUP, DOC_TYPE_SECTION_ROLLUP):
                self.store.delete_by_doc(rec.doc_id)

        items = self._current_doc_items()
        rollups = build_folder_rollups(items, summarizer=self.summarizer)

        # Section roll-ups (only for sections that have members).
        by_doc = {it["doc_id"]: it for it in items}
        for section in self.catalog.all_sections():
            member_ids = self.catalog.docs_for_section(section.section_id)
            members = [by_doc[d] for d in member_ids if d in by_doc]
            roll = build_section_rollup(
                section.name, section.section_id, members, summarizer=self.summarizer
            )
            if roll is not None:
                rollups.append(roll)

        if not rollups:
            return
        records = [
            IndexRecord(
                id=r.rollup_id, doc_type=r.doc_type, doc_id=r.rollup_id,
                text=r.text, title=f"[{r.kind} roll-up] {os.path.basename(r.key) or r.key}",
                uri=r.key, location=r.key, is_current=True,
            )
            for r in rollups
        ]
        vectors = self.embedder.embed([r.text for r in records])
        for rec, vec in zip(records, vectors):
            rec.vector = vec
        self.store.upsert(records)

    def create_section(self, name: str, *, description: str = "", parent_id: Optional[str] = None) -> Section:
        from .identity import stable_key

        section = Section(
            section_id=stable_key(name), name=name,
            description=description, parent_id=parent_id,
        )
        self.catalog.upsert_section(section)
        return section

    def assign_to_section(self, doc_id: str, section_id: str, *, confidence: float = 1.0, rationale: str = "") -> None:
        self.catalog.assign_membership(doc_id, section_id, confidence, rationale)

    # ------------------------------------------------------------------ #
    # retrieval (the read path)
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        include_rollups: bool = True,
        include_chunks: Optional[bool] = None,
        source_ids: Optional[List[str]] = None,
        use_memory: bool = False,
    ) -> List[Evidence]:
        effective_query = self.memory.context_query(query) if use_memory else query
        return self.retriever.search(
            effective_query, k=k, include_rollups=include_rollups,
            include_chunks=include_chunks, source_ids=source_ids,
        )

    def context(self, query: str, *, k: Optional[int] = None, **kwargs) -> str:
        """Return a ready-to-inject context block with inline citations."""
        evidence = self.search(query, k=k, **kwargs)
        if not evidence:
            return ""
        blocks = []
        for i, ev in enumerate(evidence, start=1):
            blocks.append(f"[{i}] {ev.citation()}\n{ev.excerpt}")
        return "\n\n".join(blocks)

    def as_tool(self, *, name: str = "librarian_search", k: Optional[int] = None):
        from .tool import LibrarianTool

        return LibrarianTool(self, name=name, default_k=k or self.config.default_k)

    # ------------------------------------------------------------------ #
    # housekeeping
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        docs = self.catalog.all_documents()
        records = self.store.all_records()
        by_type: dict = {}
        for rec in records:
            by_type[rec.doc_type] = by_type.get(rec.doc_type, 0) + 1
        return {
            "documents": len(docs),
            "sections": len(self.catalog.all_sections()),
            "index_records": len(records),
            "by_type": by_type,
        }

    def close(self) -> None:
        self.store.close()
        self.catalog.close()

    def __enter__(self) -> "Librarian":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
