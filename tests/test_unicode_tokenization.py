"""Accented / non-ASCII text must be tokenized, indexed and retrieved.

Before this change every tokenizer used ``[a-zA-Z0-9]`` character classes, so
Spanish words such as "renovación", "atención" or "niño" were dropped from
query terms, keywords, tags and hashing vectors. A query consisting only of
accented words returned nothing at all.
"""

import os

import pytest

from librarian import Librarian
from librarian.config import LibrarianConfig
from librarian.embeddings.hashing import HashingEmbedder
from librarian.embeddings.hashing import _TOKEN_RE as HASH_TOKEN_RE
from librarian.keywords import _TOKEN_RE as KEYWORD_TOKEN_RE
from librarian.keywords import extract_keywords
from librarian.models import DOC_TYPE_SUMMARY
from librarian.retrieval import Retriever
from librarian.summarize import _WORD_RE as SUMMARY_WORD_RE
from librarian.summarize import ExtractiveSummarizer
from librarian.vectorstore import IndexRecord, LocalVectorStore
from librarian.vectorstore.base import Filters
from librarian.vectorstore.local_store import _TOKEN_RE as STORE_TOKEN_RE

SPANISH_WORDS = ["renovación", "atención", "niño", "pingüino", "Ñandú"]
SPANISH_PARAGRAPH = (
    "La atención médica domiciliaria y la orientación telefónica forman parte "
    "de la afiliación. La renovación es mensual, trimestral o semestral y el "
    "niño afiliado tiene atención pediátrica las 24 horas."
)

DOCS = {
    "renovaciones.md": (
        "# Política de renovación\n\nLa renovación de la afiliación puede ser "
        "mensual, trimestral o semestral. El pago se confirma por transferencia "
        "bancaria."
    ),
    "servicios.md": (
        "# Servicios de atención\n\nOrientación médica telefónica, visita médica "
        "domiciliaria y atención de emergencia las 24 horas con ambulancias "
        "propias."
    ),
    "pediatria.md": (
        "# Protocolo pediátrico\n\nCuando el paciente es un niño, el médico "
        "pregunta la edad y el peso antes de indicar la dosis. El pingüino es la "
        "mascota del área infantil."
    ),
}


@pytest.fixture
def spanish_kb(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    for name, text in DOCS.items():
        (src / name).write_text(text, encoding="utf-8")
    lib = Librarian.open(str(tmp_path / "kb"))
    lib.add_path(str(src), source_id="es")
    stats = lib.build()
    assert stats["indexed"] == len(DOCS)
    yield lib
    lib.close()


# --------------------------------------------------------------------------- #
# tokenizers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "regex", [STORE_TOKEN_RE, HASH_TOKEN_RE, KEYWORD_TOKEN_RE, SUMMARY_WORD_RE]
)
def test_token_regexes_keep_accented_letters_whole(regex):
    assert regex.findall(" ".join(SPANISH_WORDS)) == SPANISH_WORDS


def test_token_regexes_still_match_plain_ascii_identifiers():
    text = "vpn-setup customer_id 2024 x"
    assert STORE_TOKEN_RE.findall(text) == ["vpn-setup", "customer_id", "2024", "x"]
    # Keyword tokens start with a letter and are at least three characters.
    assert KEYWORD_TOKEN_RE.findall(text) == ["vpn-setup", "customer_id"]


def test_lowercasing_folds_accented_capitals():
    assert "ATENCIÓN".lower() == "atención"
    assert "NIÑO PINGÜINO".lower() == "niño pingüino"


def test_keywords_include_accented_terms():
    keywords = extract_keywords(SPANISH_PARAGRAPH, top_k=50)
    assert "atención" in keywords
    assert "renovación" in keywords
    assert "niño" in keywords


def test_extractive_summarizer_scores_accented_words():
    long_text = " ".join(
        [
            "Introducción general.",
            "La atención de emergencia es prioritaria.",
            "Otra frase sin relación.",
            "La atención pediátrica atiende al niño.",
            "Nota administrativa.",
            "Cierre.",
            "La atención telefónica complementa la atención domiciliaria.",
        ]
    )
    summary = ExtractiveSummarizer(max_chars=120, max_sentences=2).summarize(long_text)
    assert "atención" in summary


def test_hashing_embedder_accented_text_is_non_zero_and_case_insensitive():
    embedder = HashingEmbedder(dim=64)
    vec = embedder.embed_one("renovación")
    assert any(v != 0.0 for v in vec)
    assert embedder.embed_one("RENOVACIÓN") == vec


# --------------------------------------------------------------------------- #
# vector store lexical paths
# --------------------------------------------------------------------------- #
def _store_with_spanish_record():
    store = LocalVectorStore()
    store.upsert(
        [
            IndexRecord(
                id="s", doc_type=DOC_TYPE_SUMMARY, doc_id="s",
                title="Servicios de atención",
                text="Atención de emergencia para el niño afiliado.",
                tags=["atención", "niño"],
            )
        ]
    )
    return store


def test_token_overlap_lexical_search_matches_accented_terms():
    store = _store_with_spanish_record()
    for query in ("atención", "ATENCIÓN", "niño"):
        hits = store.search_lexical(query, 5)
        assert [rec.id for rec, _ in hits] == ["s"], query
        assert hits[0][1] == 1.0


def test_bm25_lexical_search_matches_accented_terms():
    store = _store_with_spanish_record()
    for query in ("atención", "ATENCIÓN", "niño"):
        hits = store.search_lexical_bm25(query, 5)
        assert [rec.id for rec, _ in hits] == ["s"], query
        # Query-term coverage confidence sees the accented token on both sides.
        assert hits[0][1] == 1.0


# --------------------------------------------------------------------------- #
# end to end: index Spanish documents, retrieve by accented terms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query, expected",
    [
        ("renovación", "renovaciones.md"),
        ("atención", "servicios.md"),
        ("ATENCIÓN", "servicios.md"),
        ("niño", "pediatria.md"),
        ("pingüino", "pediatria.md"),
    ],
)
def test_spanish_terms_index_and_retrieve_end_to_end(spanish_kb, query, expected):
    results = spanish_kb.search(query, k=3, include_rollups=False)
    assert results, query
    assert results[0].title == expected
    assert results[0].score == 1.0


def test_mixed_spanish_paragraph_query_ranks_the_right_document(spanish_kb):
    results = spanish_kb.search(
        "¿Cuándo pregunta el médico la edad del niño antes de la dosis?",
        k=3, include_rollups=False,
    )
    assert results[0].title == "pediatria.md"
    assert results[0].score > 0.5


def test_indexed_tags_keep_accented_keywords(spanish_kb):
    summaries = {
        rec.title: rec.tags
        for rec in spanish_kb.store.all_records()
        if rec.doc_type == DOC_TYPE_SUMMARY
    }
    assert "renovación" in summaries["renovaciones.md"]
    assert "atención" in summaries["servicios.md"]
    assert "niño" in summaries["pediatria.md"]


# --------------------------------------------------------------------------- #
# an FTS5 miss must not surface arbitrary nearest-neighbour noise
# --------------------------------------------------------------------------- #
def test_fts_miss_returns_empty_list_instead_of_junk(spanish_kb):
    # Sanity: the embedder fallback would rank these (non-positive) neighbours.
    qvec = spanish_kb.embedder.embed_one("xyzzy plugh")
    raw = spanish_kb.store.search_semantic(
        qvec, 10, Filters(doc_types=[DOC_TYPE_SUMMARY], is_current=True)
    )
    assert raw and all(score <= 0 for _, score in raw)

    assert spanish_kb.search("xyzzy plugh", k=5) == []


def test_hybrid_strategy_drops_non_positive_semantic_scores():
    store = _store_with_spanish_record()
    config = LibrarianConfig(retrieval_strategy="hybrid")
    retriever = Retriever(store, HashingEmbedder(dim=64), config)
    assert retriever.search("xyzzy plugh", k=3, include_rollups=False) == []
    # A genuine match still comes through the same path.
    hits = retriever.search("atención emergencia", k=3, include_rollups=False)
    assert [hit.doc_id for hit in hits] == ["s"]
