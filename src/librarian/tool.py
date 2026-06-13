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
    "catalog. Pass a specific natural-language query and include the metric, "
    "name, or detail you actually need -- not just the topic. Returns JSON "
    "evidence items, each with title, uri, location, excerpt, doc_type, and "
    "score. Cite the evidence you use, and quote concrete figures verbatim. "
    "Call again with a refined query when the first results lack the specific "
    "detail the question needs, or when a comparison needs evidence for the "
    "other side. Internal knowledge only; this does not browse the public web."
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Specific natural-language search query.",
        },
        "k": {
            "type": "integer",
            "description": "Maximum number of evidence items to return.",
            "minimum": 1,
        },
    },
    "required": ["query"],
}


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
        """Tool schema for the OpenAI Chat Completions API (``tools=[...]``)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _DESCRIPTION,
                "parameters": _PARAMETERS,
            },
        }

    def openai_responses_schema(self) -> Dict[str, Any]:
        """Tool schema for the OpenAI Responses API (flat shape)."""
        return {
            "type": "function",
            "name": self.name,
            "description": _DESCRIPTION,
            "parameters": _PARAMETERS,
        }

    def anthropic_schema(self) -> Dict[str, Any]:
        """Tool schema for the Anthropic Messages API (``tools=[...]``)."""
        return {
            "name": self.name,
            "description": _DESCRIPTION,
            "input_schema": _PARAMETERS,
        }

    def as_langchain_tool(self):
        """Return a LangChain ``StructuredTool`` wrapping this tool (needs langchain)."""
        from langchain.tools import StructuredTool  # type: ignore

        default_k = self.default_k
        run_json = self.run_json

        def librarian_search(query: str, k: int = default_k) -> str:
            """Search the knowledge base and return JSON evidence."""
            return run_json(query, k=k)

        return StructuredTool.from_function(
            name=self.name,
            description=_DESCRIPTION,
            func=librarian_search,
        )
