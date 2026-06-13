"""Reader registry + the ``parse`` entry point.

Maps a file extension to a reader instance. Custom readers can be registered at
runtime, which is how a deployment adds support for a proprietary format without
forking the library.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..models import Block, ParsedDocument
from .base import Reader
from .office import DocxReader, ImageOcrReader, PdfReader, PptxReader, XlsxReader
from .text import CsvReader, HtmlReader, JsonReader, PlainTextReader

_DEFAULT_READERS: List[Reader] = [
    PlainTextReader(),
    CsvReader(),
    JsonReader(),
    HtmlReader(),
    PdfReader(),
    DocxReader(),
    PptxReader(),
    XlsxReader(),
    ImageOcrReader(),
]


class ReaderRegistry:
    def __init__(self) -> None:
        self._by_ext: Dict[str, Reader] = {}
        for reader in _DEFAULT_READERS:
            self.register(reader)

    def register(self, reader: Reader) -> None:
        for ext in reader.extensions:
            self._by_ext[ext.lower()] = reader

    def reader_for(self, name: str) -> Optional[Reader]:
        ext = os.path.splitext(name)[1].lower()
        return self._by_ext.get(ext)

    def supported_extensions(self) -> frozenset:
        return frozenset(self._by_ext)


_REGISTRY = ReaderRegistry()


def register_reader(reader: Reader) -> None:
    """Register a custom reader on the process-wide default registry."""
    _REGISTRY.register(reader)


def supported_extensions() -> frozenset:
    return _REGISTRY.supported_extensions()


def parse_document(
    *,
    doc_id: str,
    title: str,
    uri: str,
    name: str,
    data: bytes,
    registry: Optional[ReaderRegistry] = None,
) -> ParsedDocument:
    """Parse raw bytes into a :class:`ParsedDocument`.

    Falls back to a best-effort UTF-8 decode when no reader matches the
    extension, so unknown text-like formats still contribute content.
    """
    reg = registry or _REGISTRY
    reader = reg.reader_for(name)
    blocks: List[Block] = []
    if reader is not None:
        blocks = reader.parse(name, data)
    if not blocks:
        # Last resort: treat as text if it decodes to something printable.
        from .base import decode_text

        text = decode_text(data).strip()
        if text and _looks_textual(text):
            blocks = [Block(type="text", text=text, location=name)]
    return ParsedDocument(doc_id=doc_id, title=title, uri=uri, blocks=blocks)


def _looks_textual(text: str, sample: int = 2000) -> bool:
    head = text[:sample]
    if not head:
        return False
    printable = sum(1 for c in head if c.isprintable() or c in "\n\r\t ")
    return printable / len(head) > 0.85
