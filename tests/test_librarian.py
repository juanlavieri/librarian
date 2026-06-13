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
    # exactly one summary per doc (no stale duplicates)
    summaries = [r for r in lib.store.all_records() if r.doc_type == "file_summary"]
    assert len(summaries) == 3


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


def test_tool_adapter(kb):
    lib, _, _ = kb
    tool = lib.as_tool()
    schema = tool.openai_schema()
    assert schema["function"]["name"] == "librarian_search"
    out = tool.run("architecture", k=2)
    assert "evidence" in out and isinstance(out["evidence"], list)
