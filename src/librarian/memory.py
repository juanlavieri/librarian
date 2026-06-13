"""Short-term conversational memory.

Retrieval is more accurate when it knows what was just discussed ("the earlier
valve document"). This keeps the last N turns verbatim and compresses older
turns into a running summary, so intent can be carried forward without unbounded
context growth. Memory is *context*, never training data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 8, summarizer=None) -> None:
        self.max_turns = max_turns
        self._turns: Deque[Turn] = deque()
        self._summary = ""
        self._summarizer = summarizer

    def add(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        self._turns.append(Turn(role=role, content=content))
        while len(self._turns) > self.max_turns:
            self._compress(self._turns.popleft())

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def _compress(self, turn: Turn) -> None:
        snippet = f"{turn.role}: {turn.content[:300]}"
        combined = f"{self._summary}\n{snippet}".strip()
        if self._summarizer is not None and len(combined) > 1500:
            try:
                combined = self._summarizer.summarize(combined, title="conversation")
            except Exception:
                combined = combined[-1500:]
        self._summary = combined[-2000:]

    def recent_turns(self) -> List[Turn]:
        return list(self._turns)

    def summary(self) -> str:
        return self._summary

    def context_query(self, query: str) -> str:
        """Augment a query with recent turns to disambiguate references."""
        recent = " ".join(t.content for t in self._turns if t.role == "user")
        parts = [p for p in (self._summary, recent, query) if p]
        return "\n".join(parts)[-4000:]

    def as_text(self) -> str:
        lines = []
        if self._summary:
            lines.append(f"[earlier conversation summary]\n{self._summary}")
        for turn in self._turns:
            lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""
