"""Lightweight keyword extraction (no dependencies).

Frequency-based extraction over a stopword-filtered token stream. Good enough to
power lexical matching and the library "glossary"; deployments that want
KeyBERT/YAKE can replace this with their own and feed keywords in via metadata.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")

_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over under
    is are was were be been being do does did have has had this that these those it its
    as not no so such than too very can will would should could may might must shall
    about above after again against all am any because before below between both during
    each few more most other some only own same up down out off again here there when
    where why how what which who whom whose you your yours we our ours they them their
    he she his her him i me my mine us also per via etc using used use within across
    """.split()
)


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    tokens = [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_k)]
