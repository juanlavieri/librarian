"""Embedding provider contract."""

from __future__ import annotations

from typing import List, Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - protocol
        ...

    def embed_one(self, text: str) -> List[float]:  # pragma: no cover - protocol
        ...
