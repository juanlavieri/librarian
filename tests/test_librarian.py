import os
import sqlite3

import pytest

from librarian import Librarian, SQLConnector


@pytest.fixture()
def kb(tmp_path):
    docs = tmp_path / "docs"
    (docs / "billing").mkdir(parents=True)
    (docs / "eng").mkdir(parents=True)
    (docs / "billing" / "pricing.md").write_text(
        "# Pricing\nThe Pro plan costs 49 dollars per month. Invoices monthly."
    )
    (docs / "billing" / "refunds.txt").write_text(
        "Refund policy: refund within 30 days, processed in 5 business days."
    )
    (docs / "eng" / "arch.md").write_text(
        "# Architecture\nThe service uses Postgres and Redis with a FastAPI app."
    )
    lib = Librarian.open(str(tmp_path / "kb"))
    lib.add_path(str(docs), source_id="docs")
    stats = lib.build()
    yield lib, stats, tmp_path
    lib.close()


def test_build_indexes_all_docs(kb):
    lib, stats, _ = kb
    assert stats["indexed"] == 3
    assert stats["failed"] == 0
    s = lib.stats()
    assert s["documents"] == 3
    assert s["by_type"]["file_summary"] == 3
    assert s["by_type"].get("folder_rollup", 0) >= 2


def test_search_returns_relevant_evidence(kb):
    lib, _, _ = kb
    results = lib.search("postgres redis database architecture", k=5)
    assert results
    assert any("arch" in (e.uri or "") for e in results)
    assert all(e.score >= 0 for e in results)


def test_context_block_has_citations(kb):
    lib, _, _ = kb
    ctx = lib.context("refund policy", k=3)
    assert "refund" in ctx.lower()
    assert "[1]" in ctx


def test_incremental_build_skips_unchanged(kb):
    lib, _, tmp_path = kb
    lib.add_path(str(tmp_path / "docs"), source_id="docs")
    stats2 = lib.build()
    assert stats2["indexed"] == 0
    assert stats2["skipped"] == 3


def test_new_version_replaces_old(kb):
    lib, _, tmp_path = kb
    # change a file -> new content hash -> reindex, old records replaced
    (tmp_path / "docs" / "billing" / "pricing.md").write_text(
        "# Pricing\nThe Pro plan now costs 59 dollars per month."
    )
    lib.add_path(str(tmp_path / "docs"), source_id="docs")
    stats = lib.build()
    assert stats["indexed"] == 1
    assert stats["skipped"] == 2
    summaries = [r for r in lib.store.all_records() if r.doc_type == "file_summary"]
    # Exactly one CURRENT summary per doc (no stale duplicates in live results)...
    current = [r for r in summaries if r.is_current]
    assert len(current) == 3
    # ...and the superseded edition is retained (archived), not deleted.
    archived = [r for r in summaries if not r.is_current]
    assert len(archived) == 1


def test_persistence_reopen(kb):
    lib, _, tmp_path = kb
    root = str(tmp_path / "kb")
    lib.close()
    reopened = Librarian.open(root)
    assert reopened.stats()["documents"] == 3
    assert reopened.search("pricing pro plan", k=2)
    reopened.close()


def test_sections_and_membership(kb):
    lib, _, _ = kb
    section = lib.create_section("Finance", description="money stuff")
    doc = lib.catalog.all_documents()[0]
    lib.assign_to_section(doc.doc_id, section.section_id)
    assert section.section_id in lib.catalog.sections_for_doc(doc.doc_id)


def test_sql_connector(tmp_path):
    dbp = str(tmp_path / "s.db")
    conn = sqlite3.connect(dbp)
    conn.execute("CREATE TABLE customers(id INTEGER, name TEXT, email TEXT)")
    conn.executemany(
        "INSERT INTO customers VALUES(?,?,?)",
        [(i, f"n{i}", f"e{i}@x.com") for i in range(20)],
    )
    conn.commit()
    conn.close()

    lib = Librarian.open(str(tmp_path / "kb2"))
    lib.add_connector(SQLConnector(sqlite_path=dbp, source_id="db", sample_rows=10))
    stats = lib.build()
    assert stats["indexed"] == 1
    doc = lib.catalog.all_documents()[0]
    assert doc.metadata["modality"] == "tabular"
    assert doc.metadata["profile"]["row_count"] == 10  # sampled
    results = lib.search("customer email", k=2)
    assert results
    lib.close()


def test_archived_version_retrieval(kb):
    lib, _, tmp_path = kb
    # Supersede a file with new content.
    (tmp_path / "docs" / "billing" / "pricing.md").write_text(
        "# Pricing\nThe Pro plan now costs 59 dollars per month."
    )
    lib.add_path(str(tmp_path / "docs"), source_id="docs")
    lib.build()

    # Default search (current only) never surfaces the old "$49" edition.
    current_text = " ".join(e.excerpt for e in lib.search("pro plan price per month", k=10))
    assert "49" not in current_text
    assert "59" in current_text

    # Archived editions are still queryable when explicitly requested.
    archived_text = " ".join(
        e.excerpt for e in lib.retriever.search("pro plan 49 dollars", k=10, include_archived=True)
    )
    assert "49" in archived_text  # the superseded edition is retained in the index


def test_build_resilient_to_failing_connector(tmp_path):
    class BoomConnector:
        def items(self):
            raise RuntimeError("store unreachable")
            yield  # pragma: no cover

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("hello world content about widgets")

    lib = Librarian.open(str(tmp_path / "kb"))
    lib.add_connector(BoomConnector())
    lib.add_path(str(docs), source_id="docs")
    stats = lib.build()  # must not raise
    assert stats["failed"] >= 1
    assert stats["indexed"] == 1  # the good source still got indexed
    lib.close()


def test_lexical_matches_path_and_filename(kb):
    lib, _, _ = kb
    # A query naming the file/folder should surface it via lexical matching.
    results = lib.search("refunds", k=5)
    assert any("refunds" in (e.uri or "").lower() for e in results)


def test_faiss_backend(tmp_path):
    pytest.importorskip("faiss")
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.md").write_text("# Networking\nThe VPN uses SSO and a corporate gateway.")
    (docs / "b.md").write_text("# Payroll\nSalaries are paid on the last business day.")

    lib = Librarian.open(str(tmp_path / "kb"), vector_backend="faiss")
    lib.add_path(str(docs), source_id="docs")
    stats = lib.build()
    assert stats["indexed"] == 2
    results = lib.search("how does the vpn authenticate", k=3)
    assert results
    lib.close()
    # Reopen to confirm FAISS metadata persisted.
    reopened = Librarian.open(str(tmp_path / "kb"), vector_backend="faiss")
    assert reopened.stats()["documents"] == 2
    reopened.close()


def test_custom_store_without_archival_helpers(tmp_path):
    """A custom VectorStore predating archive_doc/delete_by_doc_version must
    still index successfully (falls back to delete_by_doc), not fail silently."""
    from librarian.vectorstore.base import Filters, cosine

    class LegacyStore:
        # Implements only the original VectorStore surface -- no archival helpers.
        def __init__(self):
            self._recs = {}

        def upsert(self, records):
            for r in records:
                self._recs[r.id] = r

        def delete_by_doc(self, doc_id):
            for rid in [k for k, r in self._recs.items() if r.doc_id == doc_id]:
                del self._recs[rid]

        def search_semantic(self, vector, k, filters=None):
            out = []
            for r in self._recs.values():
                if filters is not None and not filters.matches(r):
                    continue
                if r.vector:
                    out.append((r, cosine(vector, r.vector)))
            out.sort(key=lambda x: x[1], reverse=True)
            return out[:k]

        def search_lexical(self, query, k, filters=None):
            return []

        def all_records(self):
            return list(self._recs.values())

        def persist(self):
            pass

        def close(self):
            pass

    assert not hasattr(LegacyStore(), "archive_doc")

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.md").write_text("# Topic\nThe widget ships in three sizes.")

    from librarian import LibrarianConfig

    lib = Librarian(LibrarianConfig(root=str(tmp_path / "kb")), store=LegacyStore())
    lib.add_path(str(docs), source_id="docs")
    stats = lib.build()
    assert stats["failed"] == 0
    assert stats["indexed"] == 1
    assert lib.search("widget sizes", k=3)

    # Reindex a changed doc still works via the delete_by_doc fallback.
    (docs / "a.md").write_text("# Topic\nThe widget now ships in five sizes.")
    lib.add_path(str(docs), source_id="docs")
    stats2 = lib.build()
    assert stats2["failed"] == 0
    assert stats2["indexed"] == 1
    lib.close()


def test_tool_adapter(kb):
    lib, _, _ = kb
    tool = lib.as_tool()
    schema = tool.openai_schema()
    assert schema["function"]["name"] == "librarian_search"
    out = tool.run("architecture", k=2)
    assert "evidence" in out and isinstance(out["evidence"], list)


def test_tool_schemas_are_valid(kb):
    import json

    lib, _, _ = kb
    tool = lib.as_tool()

    chat = tool.openai_schema()
    assert chat["type"] == "function"
    assert chat["function"]["parameters"]["required"] == ["query"]

    resp = tool.openai_responses_schema()
    assert resp["type"] == "function" and resp["name"] == tool.name
    assert "parameters" in resp  # flat shape, no nested "function"

    anth = tool.anthropic_schema()
    assert anth["name"] == tool.name and "input_schema" in anth

    # run_json is parseable and shaped for a tool message
    payload = json.loads(tool.run_json("architecture", k=2))
    assert "evidence" in payload
