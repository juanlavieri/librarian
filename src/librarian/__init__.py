"""Librarian -- a foundational knowledge layer for AI systems.

A two-part system: a context-rich, recursively organized **catalog** of any
document store, and an agentic, hybrid **retriever** that finds highly relevant,
citation-ready evidence fast. Plug-and-play with zero required dependencies.

Quickstart::

    from librarian import Librarian

    lib = Librarian.open("./kb")
    lib.add_path("./docs")
    lib.build()
    for ev in lib.search("how does billing work?"):
        print(ev.citation(), ev.score)
"""

from __future__ import annotations

from .config import LibrarianConfig
from .connectors import (
    Connector,
    FilesystemConnector,
    SQLConnector,
    WebConnector,
)
from .enrich import AssetProfile, profile_document
from .librarian import Librarian
from .memory import ConversationMemory
from .models import (
    Chunk,
    Document,
    Evidence,
    ParsedDocument,
    Rollup,
    Section,
    Version,
)
from .readers import register_reader, supported_extensions
from .tool import LibrarianTool

__version__ = "0.1.0"

__all__ = [
    "Librarian",
    "LibrarianConfig",
    "LibrarianTool",
    "ConversationMemory",
    "Connector",
    "FilesystemConnector",
    "WebConnector",
    "SQLConnector",
    "AssetProfile",
    "profile_document",
    "Document",
    "Version",
    "Chunk",
    "Section",
    "Rollup",
    "Evidence",
    "ParsedDocument",
    "register_reader",
    "supported_extensions",
    "__version__",
]
