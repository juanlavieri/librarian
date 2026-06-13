from librarian.chunking import chunk_document, count_tokens
from librarian.readers import parse_document, supported_extensions


def _parse(name, data):
    return parse_document(doc_id="d", title=name, uri=f"file://{name}", name=name, data=data)


def test_plain_text_reader():
    parsed = _parse("note.txt", b"hello world this is a note")
    assert "note" in parsed.text


def test_csv_reader_renders_pipe_table():
    parsed = _parse("t.csv", b"a,b,c\n1,2,3\n4,5,6\n")
    assert parsed.blocks[0].type == "table"
    assert " | " in parsed.blocks[0].text


def test_html_reader_strips_tags():
    parsed = _parse("p.html", b"<html><body><p>Hello</p><script>x()</script></body></html>")
    assert "Hello" in parsed.text
    assert "x()" not in parsed.text


def test_unknown_extension_falls_back_to_text():
    parsed = _parse("weird.zzz", b"plain readable content here")
    assert "readable" in parsed.text


def test_supported_extensions_includes_common():
    exts = supported_extensions()
    assert ".txt" in exts and ".csv" in exts and ".md" in exts


def test_chunking_overlaps_and_locates():
    blocks_text = "\n\n".join(f"Section {i}. " + "word " * 200 for i in range(5))
    parsed = _parse("big.txt", blocks_text.encode())
    chunks = chunk_document(parsed, "v1", max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(c.token_count > 0 for c in chunks)
    assert all(c.chunk_hash for c in chunks)


def test_count_tokens_positive():
    assert count_tokens("a b c") >= 1
