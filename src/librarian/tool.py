"""Agent tool adapter.

Exposes the Librarian's read path as a function tool any agent runtime can call.
Provides the OpenAI tool/function JSON schema, a plain callable, and an optional
LangChain ``BaseTool`` adapter. This is the seam where the foundational
knowledge layer plugs into whatever agent framework sits on top.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .librarian import Librarian

_DESCRIPTION = (
    "Search the knowledge base for relevant, citation-ready evidence. Hybrid "
    "retrieval (semantic + lexical + structural) over a context-enriched "
    "catalog. Pass a specific natural-language query; include the metric, name, "
    "or detail you actually need. Returns JSON evidence items with title, uri, "
    "location, excerpt, and score -- cite them in your answer."
)


class LibrarianTool:
    def __init__(self, librarian: "Librarian", *, name: str = "librarian_search", default_k: int = 8) -> None:
        self._lib = librarian
        self.name = name
        self.default_k = default_k

    # --- invocation ---
    def run(self, query: str, k: Optional[int] = None, source_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        evidence = self._lib.search(query, k=k or self.default_k, source_ids=source_ids)
        return {"evidence": [e.to_dict() for e in evidence]}

    def run_json(self, query: str, k: Optional[int] = None, source_ids: Optional[List[str]] = None) -> str:
        return json.dumps(self.run(query, k=k, source_ids=source_ids), default=str)

    def __call__(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        return self.run(query, k=k)

    # --- schemas ---
    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max evidence items to return.",
                            "default": self.default_k,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def as_langchain_tool(self):
        """Return a LangChain ``BaseTool`` wrapping this tool (needs langchain)."""
        from langchain.tools import StructuredTool  # type: ignore

        return StructuredTool.from_function(
            name=self.name,
            description=_DESCRIPTION,
            func=lambda query, k=self.default_k: self.run_json(query, k=k),
        )
