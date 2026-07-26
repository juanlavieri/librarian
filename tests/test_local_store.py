from concurrent.futures import ThreadPoolExecutor

import pytest

from librarian.vectorstore import Filters, IndexRecord, LocalVectorStore


def _record(record_id, text, *, current=True):
    return IndexRecord(
        id=record_id,
        doc_type="file_summary",
        doc_id=record_id,
        title=record_id,
        text=text,
        is_current=current,
    )


def test_fts5_bm25_prefers_rare_specific_match():
    store = LocalVectorStore()
    store.upsert(
        [
            _record("specific", "The zephyr gateway authenticates the corporate VPN."),
            _record("generic", "A gateway provides access. Gateway access is documented."),
        ]
    )
    results = store.search_lexical_bm25(
        "zephyr gateway VPN", 2, Filters(doc_types=["file_summary"], is_current=True)
    )
    assert [record.id for record, _ in results] == ["specific", "generic"]


def test_fts5_filters_archived_records_before_limiting():
    store = LocalVectorStore()
    store.upsert(
        [
            _record("archived", "orchid orchid orchid policy", current=False),
            _record("current", "orchid policy", current=True),
        ]
    )
    results = store.search_lexical_bm25("orchid", 1, Filters(is_current=True))
    assert [record.id for record, _ in results] == ["current"]


def test_fts_index_is_refreshed_after_upsert():
    store = LocalVectorStore()
    store.upsert([_record("one", "alpha")])
    assert store.search_lexical_bm25("alpha", 3)
    store.upsert([_record("two", "beta")])
    assert store.search_lexical_bm25("beta", 3)[0][0].id == "two"


def test_fts_search_can_move_between_worker_threads():
    store = LocalVectorStore()
    store.upsert([_record("one", "thread safe orchid search")])
    assert store.search_lexical_bm25("orchid", 3)
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = executor.submit(store.search_lexical_bm25, "orchid", 3).result()
    assert results[0][0].id == "one"


@pytest.mark.parametrize("backend", ["local", "faiss"])
def test_fts_concurrent_reads_and_writes(backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        from librarian.vectorstore.faiss_store import FaissVectorStore

        store = FaissVectorStore(dim=4)
    else:
        store = LocalVectorStore()
    store.upsert([_record("seed", "orchid seed")])

    def search_many():
        for _ in range(50):
            store.search_lexical_bm25("orchid", 5)

    def write_many():
        for i in range(50):
            store.upsert([_record(f"new-{i}", f"orchid record {i}")])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(search_many), executor.submit(write_many)]
        for future in futures:
            future.result()
    assert store.search_lexical_bm25("orchid", 100)

