"""Summarizers: the "summary-first" content-understanding layer.

The Librarian embeds *summaries* before chunks. Summaries carry cleaner
semantic signal, cost far fewer tokens to store and search, and stay
human-readable. The default summarizer is fully offline and extractive; an
OpenAI-backed summarizer is provided for higher-quality abstractive summaries.
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol


class Summarizer(Protocol):
    def summarize(self, text: str, *, title: str = "") -> str:  # pragma: no cover - protocol
        ...


_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")


class ExtractiveSummarizer:
    """Dependency-free frequency-based extractive summarizer.

    Scores sentences by the summed frequency of their (stopword-filtered) terms
    and returns the top sentences in original order. Deterministic and offline.
    """

    def __init__(self, max_chars: int = 1200, max_sentences: int = 6) -> None:
        self.max_chars = max_chars
        self.max_sentences = max_sentences

    def summarize(self, text: str, *, title: str = "") -> str:
        text = (text or "").strip()
        if len(text) <= self.max_chars and text.count("\n") < 4:
            return text
        sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        if not sentences:
            return text[: self.max_chars]
        if len(sentences) <= self.max_sentences:
            return " ".join(sentences)[: self.max_chars]

        from .keywords import _STOPWORDS

        freq: dict = {}
        for word in (w.lower() for w in _WORD_RE.findall(text)):
            if word in _STOPWORDS or word.isdigit():
                continue
            freq[word] = freq.get(word, 0) + 1
        if not freq:
            return " ".join(sentences[: self.max_sentences])[: self.max_chars]

        scored: List[tuple] = []
        for idx, sentence in enumerate(sentences):
            words = [w.lower() for w in _WORD_RE.findall(sentence)]
            if not words:
                continue
            score = sum(freq.get(w, 0) for w in words) / (len(words) ** 0.5)
            scored.append((score, idx, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = sorted(scored[: self.max_sentences], key=lambda x: x[1])
        summary = " ".join(s for _, _, s in chosen)
        return summary[: self.max_chars]


class OpenAISummarizer:
    """Abstractive summaries via the OpenAI API (optional)."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_chars: int = 1200,
        input_chars: int = 8000,
    ) -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.max_chars = max_chars
        self.input_chars = input_chars

    def summarize(self, text: str, *, title: str = "") -> str:
        text = (text or "").strip()
        if not text:
            return ""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write crisp, factual, retrieval-friendly summaries. "
                            "Preserve concrete figures, names, and dates."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Title: {title}\n\nSummarize the document below in 4-6 "
                            f"sentences, keeping any numbers and specifics:\n\n"
                            f"{text[: self.input_chars]}"
                        ),
                    },
                ],
                max_tokens=400,
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()[: self.max_chars]
        except Exception:
            return ExtractiveSummarizer(max_chars=self.max_chars).summarize(text, title=title)
