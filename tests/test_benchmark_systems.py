from benchmarks.datasets import BenchmarkDataset, CorpusRecord, QueryRecord
from benchmarks.systems import (
    FTS5BM25System,
    ForcedQueryFailureSystem,
    LexicalFallbackSystem,
)


def _dataset():
    return BenchmarkDataset(
        name="fixture",
        corpus={
            "relevant": CorpusRecord(
                id="relevant",
                title="Orchid incident escalation",
                text="The Orchid incident escalation window is seven minutes.",
            ),
            "other": CorpusRecord(
                id="other",
                title="Travel policy",
                text="Receipts are required for travel reimbursement.",
            ),
        },
        queries={"q": QueryRecord(id="q", text="Orchid escalation window")},
        qrels={"q": {"relevant": 1}},
    )


def test_fts5_bm25_ranks_lexical_match_first():
    system = FTS5BM25System()
    try:
        stats = system.build(_dataset())
        assert stats.records == 2
        assert system.rank("Orchid escalation window", 2)[0][0] == "relevant"
    finally:
        system.close()


def test_semantic_failure_falls_back_to_lexical_results():
    system = LexicalFallbackSystem(ForcedQueryFailureSystem(), FTS5BM25System())
    try:
        system.build(_dataset())
        assert system.rank("Orchid escalation window", 2)[0][0] == "relevant"
    finally:
        system.close()

