"""Plain-text family readers (txt, md, code, csv, tsv, json, xml, html).

All of these rely only on the standard library so they are always available.
HTML is parsed with BeautifulSoup when installed and falls back to a tag
stripper otherwise.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import List

from ..models import Block
from .base import decode_text


class PlainTextReader:
    extensions = frozenset(
        {".txt", ".md", ".markdown", ".rst", ".log", ".py", ".js", ".ts",
         ".java", ".go", ".rb", ".rs", ".c", ".cpp", ".h", ".sh", ".yaml",
         ".yml", ".toml", ".ini", ".cfg"}
    )

    def parse(self, name: str, data: bytes) -> List[Block]:
        text = decode_text(data).strip()
        if not text:
            return []
        return [Block(type="text", text=text, location=name)]


class CsvReader:
    extensions = frozenset({".csv", ".tsv"})

    def parse(self, name: str, data: bytes) -> List[Block]:
        text = decode_text(data)
        delimiter = "\t" if name.lower().endswith(".tsv") else ","
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        except csv.Error:
            return [Block(type="text", text=text.strip(), location=name)]
        if not rows:
            return []
        # Keep a bounded, readable rendering of the table.
        header = rows[0]
        lines = [" | ".join(header)]
        for row in rows[1:1000]:
            lines.append(" | ".join(row))
        return [Block(type="table", text="\n".join(lines), location=name)]


class JsonReader:
    extensions = frozenset({".json", ".jsonl", ".ndjson"})

    def parse(self, name: str, data: bytes) -> List[Block]:
        text = decode_text(data).strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pretty = text
        return [Block(type="text", text=pretty[:200_000], location=name)]


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class HtmlReader:
    extensions = frozenset({".html", ".htm", ".xhtml", ".xml"})

    def parse(self, name: str, data: bytes) -> List[Block]:
        raw = decode_text(data)
        text = self._extract(raw)
        text = _WS_RE.sub(" ", text).strip()
        if not text:
            return []
        return [Block(type="text", text=text, location=name)]

    @staticmethod
    def _extract(raw: str) -> str:
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text(" ")
        except Exception:
            no_scripts = re.sub(
                r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I
            )
            return _TAG_RE.sub(" ", no_scripts)
