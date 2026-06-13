"""Intake: turn heterogeneous sources into a uniform stream of raw items.

Everything the Librarian ingests -- a folder of files, an in-memory string, a
crawled web page -- becomes an :class:`IntakeItem` (bytes + locator + source
namespace). Downstream stages (parse, summarize, chunk, embed) never need to
know where an item came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .readers import supported_extensions


@dataclass
class IntakeItem:
    uri: str
    name: str
    data: bytes
    source_id: str = "default"
    title: str = ""
    media_type: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title:
            self.title = os.path.basename(self.name) or self.uri
        if not self.media_type:
            self.media_type = os.path.splitext(self.name)[1].lstrip(".").lower()


# Directories that are never useful to index.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "dist", "build",
    ".librarian_data", "librarian_data",
}


def from_text(
    text: str, *, name: str, source_id: str = "default", title: str = "",
    metadata: Optional[Dict[str, object]] = None,
) -> IntakeItem:
    return IntakeItem(
        uri=f"text://{name}",
        name=name if "." in name else f"{name}.txt",
        data=text.encode("utf-8"),
        source_id=source_id,
        title=title or name,
        metadata=metadata or {},
    )


def from_path(
    path: str,
    *,
    source_id: str = "default",
    recursive: bool = True,
    max_bytes: int = 50 * 1024 * 1024,
    extensions: Optional[frozenset] = None,
) -> Iterator[IntakeItem]:
    """Yield intake items from a file or directory.

    Only files with a supported (or explicitly allowed) extension are read.
    Files larger than ``max_bytes`` are skipped to avoid pathological inputs.
    """
    allowed = extensions if extensions is not None else supported_extensions()
    path = os.path.abspath(path)
    if os.path.isfile(path):
        item = _read_file(path, source_id, allowed, max_bytes)
        if item is not None:
            yield item
        return
    if not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            item = _read_file(fpath, source_id, allowed, max_bytes, base=path)
            if item is not None:
                yield item
        if not recursive:
            break


def _read_file(
    fpath: str,
    source_id: str,
    allowed: frozenset,
    max_bytes: int,
    base: Optional[str] = None,
) -> Optional[IntakeItem]:
    ext = os.path.splitext(fpath)[1].lower()
    if allowed and ext not in allowed:
        return None
    try:
        size = os.path.getsize(fpath)
        if size > max_bytes:
            return None
        with open(fpath, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    rel = os.path.relpath(fpath, base) if base else os.path.basename(fpath)
    return IntakeItem(
        uri=f"file://{fpath}",
        name=rel,
        data=data,
        source_id=source_id,
        title=os.path.basename(fpath),
        metadata={"path": fpath, "size": size},
    )


def from_pages(pages: List[dict], *, source_id: str = "web") -> Iterator[IntakeItem]:
    """Adapt crawled pages (see :mod:`librarian.crawl`) into intake items."""
    for page in pages:
        url = page.get("url", "")
        text = page.get("text", "")
        if not url or not text:
            continue
        yield IntakeItem(
            uri=url,
            name=(page.get("title") or url) + ".html",
            data=text.encode("utf-8"),
            source_id=source_id,
            title=page.get("title") or url,
            media_type="html",
            metadata={"url": url, "crawl_depth": page.get("depth", 0)},
        )
