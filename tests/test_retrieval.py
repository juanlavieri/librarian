from librarian.config import LibrarianConfig
from librarian.models import DOC_TYPE_CHUNK
from librarian.retrieval import Retriever
from librarian.vectorstore import IndexRecord, LocalVectorStore


class FailingEmbedder:
    def embed_one(self, text):
        raise RuntimeError("provider unavailable")


def test_hybrid_semantic_failure_falls_back_to_lexical():
    store = LocalVectorStore()
    store.upsert(
        [
            IndexRecord(
                id="doc",
                doc_type="file_summary",
                doc_id="doc",
                title="Orchid runbook",
                text="Escalate an Orchid incident within seven minutes.",
            )
        ]
    )
    config = LibrarianConfig(retrieval_strategy="hybrid")
    retriever = Retriever(store, FailingEmbedder(), config)
    results = retriever.search("Orchid incident escalation", k=3, include_rollups=False)
    assert [result.doc_id for result in results] == ["doc"]


def test_fts_strategy_does_not_require_embeddings_for_lexical_hit():
    store = LocalVectorStore()
    store.upsert(
        [
            IndexRecord(
                id="doc",
                doc_type="file_summary",
                doc_id="doc",
                title="Zephyr policy",
                text="The Zephyr retention period is 30 days.",
            )
        ]
    )
    retriever = Retriever(store, FailingEmbedder(), LibrarianConfig())
    results = retriever.search("Zephyr retention", k=3, include_rollups=False)
    assert [result.doc_id for result in results] == ["doc"]


def test_fts_unavailable_falls_back_to_portable_token_overlap():
    store = LocalVectorStore()
    store.upsert(
        [
            IndexRecord(
                id="doc",
                doc_type="file_summary",
                doc_id="doc",
                title="Portable fallback",
                text="The kestrel identifier is searchable.",
            )
        ]
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("SQLite was built without FTS5")

    store.search_lexical_bm25 = unavailable
    retriever = Retriever(store, FailingEmbedder(), LibrarianConfig())
    results = retriever.search("kestrel identifier", k=3, include_rollups=False)
    assert [result.doc_id for result in results] == ["doc"]


def test_low_confidence_bm25_summary_deepens_to_matching_chunk():
    store = LocalVectorStore()
    store.upsert(
        [
            IndexRecord(
                id="summary",
                doc_type="file_summary",
                doc_id="doc",
                title="Orchid handbook",
                text="General Orchid guidance.",
            ),
            IndexRecord(
                id="chunk",
                doc_type=DOC_TYPE_CHUNK,
                doc_id="doc",
                title="Orchid handbook",
                text="Orchid emergency escalation takes seven minutes.",
            ),
        ]
    )
    retriever = Retriever(store, FailingEmbedder(), LibrarianConfig())
    results = retriever.search(
        "Orchid emergency escalation minutes", k=3, include_rollups=False
    )
    assert results[0].doc_type == DOC_TYPE_CHUNK
    assert "seven minutes" in results[0].excerpt

