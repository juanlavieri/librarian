from librarian.enrich import (
    MODALITY_PROSE,
    MODALITY_TABULAR,
    profile_document,
)
from librarian.readers import parse_document


def _parse(name, data):
    return parse_document(doc_id="d", title=name, uri=f"file://{name}", name=name, data=data)


def test_tabular_profile_infers_columns_and_types():
    csv = b"id,name,email,active\n1,Acme,a@x.com,true\n2,Globex,g@y.com,false\n3,Initech,i@z.com,true\n"
    profile = profile_document(_parse("c.csv", csv), media_type="csv")
    assert profile.modality == MODALITY_TABULAR
    types = {c.name: c.inferred_type for c in profile.columns}
    assert types["email"] == "email"
    assert types["id"] == "number"
    assert profile.row_count == 3
    ctx = profile.as_context()
    assert "columns" in ctx.lower()


def test_prose_profile_extracts_topics():
    text = b"Payments and billing run monthly. Invoices and refunds are handled by the billing service."
    profile = profile_document(_parse("b.txt", text), media_type="txt")
    assert profile.modality == MODALITY_PROSE
    assert profile.topics
    assert profile.description


def test_empty_asset():
    profile = profile_document(_parse("e.txt", b"   "), media_type="txt")
    assert profile.modality == "empty"
