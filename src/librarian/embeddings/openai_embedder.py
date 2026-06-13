"""OpenAI embedding provider (optional).

Requires the ``openai`` package and an API key. Install with the ``openai``
extra.
"""

from __future__ import annotations

from typing import List, Optional

_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dim: Optional[int] = None,
        batch_size: int = 128,
    ) -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.dim = dim or _DIMS.get(model, 1536)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:8000] or " " for t in texts[start : start + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
