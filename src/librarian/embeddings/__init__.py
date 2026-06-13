from .base import Embedder
from .hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "OpenAIEmbedder"]


def __getattr__(name: str):
    # Lazy import so the optional openai dependency is only required if used.
    if name == "OpenAIEmbedder":
        from .openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
