from librarian.identity import (
    chunk_id_for,
    content_hash,
    doc_id_for,
    version_id_for,
)


def test_doc_id_is_stable_and_unique():
    a = doc_id_for("src", "file://a.txt")
    assert a == doc_id_for("src", "file://a.txt")
    assert a != doc_id_for("src", "file://b.txt")
    assert a != doc_id_for("other", "file://a.txt")


def test_version_id_changes_with_content():
    v1 = version_id_for(b"hello")
    v2 = version_id_for(b"hello world")
    assert v1 != v2
    assert v1 == version_id_for(b"hello")


def test_version_id_prefers_etag():
    assert version_id_for(b"x", etag="abc") == version_id_for(b"y", etag="abc")


def test_content_hash_and_chunk_id():
    assert content_hash(b"a") != content_hash(b"b")
    cid = chunk_id_for("doc", "ver", 3)
    assert cid.startswith("chunk_") and cid.endswith("0003")
