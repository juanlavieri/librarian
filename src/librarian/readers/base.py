"""Reader contract.

A reader turns raw bytes for one media type into an ordered list of located
``Block``s. Locations ("p.12", "slide 8", "Sheet1") are what make citations
human-checkable later. Readers must never raise on malformed input -- they
return whatever text they could recover (possibly empty).
"""

from __future__ import annotations

from typing import List, Protocol

from ..models import Block


class Reader(Protocol):
    extensions: frozenset

    def parse(self, name: str, data: bytes) -> List[Block]:  # pragma: no cover - protocol
        ...


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="ignore")
