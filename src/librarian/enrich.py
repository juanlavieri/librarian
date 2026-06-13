"""Asset profiling: open each asset and describe what is inside it.

This is the part of the Librarian that behaves like a cataloguer physically
opening every item on the shelf. For each asset it infers:

* **modality** -- is this a table, prose, structured data, or binary?
* **structure** -- for tabular assets it reads the header, samples the first
  rows, and infers column types and row count (so the catalog "knows" a CSV is
  a list of customers with an email column, not just bytes);
* **topics** -- what the asset is *about*;
* a one-line, retrieval-friendly **description** of what can be found inside.

The resulting :class:`AssetProfile` is attached to the document's metadata, woven
into its summary, and -- crucially -- rolled up into the parent folder, so the
context propagates up the tree. Profiling is offline/deterministic by default
and uses an LLM only if one is supplied.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .keywords import extract_keywords
from .models import ParsedDocument

MODALITY_TABULAR = "tabular"
MODALITY_PROSE = "prose"
MODALITY_STRUCTURED = "structured"
MODALITY_EMPTY = "empty"

_NUM_RE = re.compile(r"^-?\$?\d[\d,]*(\.\d+)?%?$")
_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
_BOOL_VALS = {"true", "false", "yes", "no", "0", "1", "y", "n"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://")


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    sample_values: List[str] = field(default_factory=list)


@dataclass
class AssetProfile:
    modality: str = MODALITY_EMPTY
    description: str = ""
    topics: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    columns: List[ColumnProfile] = field(default_factory=list)
    sample_rows: List[List[str]] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return d

    def as_context(self) -> str:
        """Compact, embeddable description of the asset for chunks/summaries."""
        parts: List[str] = []
        if self.description:
            parts.append(self.description)
        if self.modality == MODALITY_TABULAR and self.columns:
            cols = ", ".join(f"{c.name}:{c.inferred_type}" for c in self.columns[:12])
            rc = f"~{self.row_count} rows" if self.row_count is not None else "table"
            parts.append(f"Tabular dataset ({rc}); columns: {cols}.")
        if self.topics:
            parts.append("Topics: " + ", ".join(self.topics[:8]) + ".")
        return " ".join(parts).strip()


def _infer_type(values: List[str]) -> str:
    vals = [v.strip() for v in values if v is not None and str(v).strip() != ""]
    if not vals:
        return "empty"
    checks = {
        "email": lambda v: bool(_EMAIL_RE.match(v)),
        "url": lambda v: bool(_URL_RE.match(v)),
        "date": lambda v: bool(_DATE_RE.match(v)),
        "number": lambda v: bool(_NUM_RE.match(v)),
        "boolean": lambda v: v.lower() in _BOOL_VALS,
    }
    for type_name, test in checks.items():
        if sum(1 for v in vals if test(v)) / len(vals) >= 0.8:
            return type_name
    avg_len = sum(len(v) for v in vals) / len(vals)
    return "text" if avg_len > 40 else "category"


def _parse_pipe_table(text: str) -> Optional[List[List[str]]]:
    """Our CSV/XLSX readers render rows as 'a | b | c'. Recover that grid."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or " | " not in lines[0]:
        return None
    rows = [[c.strip() for c in ln.split(" | ")] for ln in lines]
    width = len(rows[0])
    rows = [r for r in rows if len(r) == width]
    return rows if len(rows) >= 2 else None


def _profile_table(rows: List[List[str]]) -> AssetProfile:
    header = rows[0]
    data = rows[1:]
    columns: List[ColumnProfile] = []
    for ci, name in enumerate(header):
        col_values = [r[ci] for r in data[:200] if ci < len(r)]
        columns.append(
            ColumnProfile(
                name=name or f"col_{ci}",
                inferred_type=_infer_type(col_values),
                sample_values=[v for v in col_values[:5] if v][:5],
            )
        )
    col_desc = ", ".join(c.name for c in columns[:10])
    description = (
        f"A table of {len(data)} records with {len(columns)} columns ({col_desc})."
    )
    return AssetProfile(
        modality=MODALITY_TABULAR,
        description=description,
        row_count=len(data),
        columns=columns,
        sample_rows=[r for r in data[:10]],
        stats={"n_columns": len(columns)},
    )


def profile_document(
    parsed: ParsedDocument,
    *,
    media_type: str = "",
    summarizer=None,
) -> AssetProfile:
    """Open a parsed asset and describe what is inside it."""
    text = parsed.text
    if not text.strip():
        return AssetProfile(modality=MODALITY_EMPTY, description="Empty or unreadable asset.")

    block_types = {b.type for b in parsed.blocks}
    is_tabularish = (
        media_type in {"csv", "tsv", "xlsx", "xlsm"}
        or "table" in block_types
        or "sheet" in block_types
    )

    if is_tabularish:
        for block in parsed.blocks:
            if block.type in {"table", "sheet"}:
                grid = _parse_pipe_table(block.text)
                if grid:
                    profile = _profile_table(grid)
                    profile.topics = extract_keywords(text, top_k=8)
                    return profile

    if media_type in {"json", "jsonl", "ndjson", "xml"}:
        modality = MODALITY_STRUCTURED
        description = "Structured data document."
    else:
        modality = MODALITY_PROSE
        description = ""

    topics = extract_keywords(text, top_k=8)
    if not description:
        head = text.strip().split("\n", 1)[0][:200]
        description = (
            f"Prose document about {', '.join(topics[:5])}." if topics
            else f"Document beginning: {head}"
        )
    profile = AssetProfile(modality=modality, description=description, topics=topics)

    if summarizer is not None:
        try:
            classified = summarizer.summarize(
                text[:4000],
                title=f"Classify the subject of '{parsed.title}' in one sentence",
            )
            if classified:
                profile.description = classified.split("\n", 1)[0][:300]
        except Exception:
            pass
    return profile
