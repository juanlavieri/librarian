"""Hierarchical roll-ups: synthesized catalogs for folders and sections.

A roll-up is a folder-/section-level summary ("what lives here, key documents,
themes, gaps"). It lets retrieval answer structural questions ("where is X?",
"what's in this area?") that chunk-level search cannot, and gives the model a
map of the knowledge base instead of a bag of fragments.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .identity import stable_key
from .models import DOC_TYPE_FOLDER_ROLLUP, DOC_TYPE_SECTION_ROLLUP, Rollup


def folder_of(uri: str) -> str:
    """Best-effort parent "folder" for a file path or URL."""
    if uri.startswith(("http://", "https://")):
        parsed = urlparse(uri)
        parts = [p for p in parsed.path.split("/") if p]
        parent = "/".join(parts[:-1])
        return f"{parsed.netloc}/{parent}".rstrip("/") or parsed.netloc
    clean = uri.replace("file://", "").replace("text://", "")
    return os.path.dirname(clean) or "/"


def _ancestors(folder: str) -> List[str]:
    """All ancestor folder keys for a folder, nearest parent first."""
    if not folder or folder in (".", "/"):
        return []
    sep = "/" if "/" in folder else os.sep
    parts = [p for p in folder.split(sep) if p]
    out = []
    for i in range(len(parts) - 1, 0, -1):
        out.append(sep.join(parts[:i]))
    return out


def build_folder_rollups(
    items: List[Dict],
    *,
    summarizer=None,
    min_items: int = 1,
    max_children: int = 30,
) -> List[Rollup]:
    """Recursively synthesize one roll-up per folder, bottom-up.

    ``items`` are dicts with keys: ``doc_id``, ``uri``, ``title``, ``summary``.

    The recursion is what makes the catalog navigable at every altitude: a
    document's profile is attributed to its folder *and* to every ancestor
    folder, then each folder is summarized from the documents and sub-folder
    roll-ups beneath it. The deepest leaf metadata therefore propagates all the
    way up to the root, so asking "what is in this whole area?" works at any
    level of the tree.
    """
    # 1. Direct membership: which documents sit directly in each folder.
    direct: Dict[str, List[Dict]] = defaultdict(list)
    # 2. Subtree membership: documents anywhere beneath each folder.
    subtree: Dict[str, List[Dict]] = defaultdict(list)
    child_folders: Dict[str, set] = defaultdict(set)

    for it in items:
        folder = it.get("folder") or folder_of(it.get("uri", ""))
        direct[folder].append(it)
        subtree[folder].append(it)
        prev = folder
        for anc in _ancestors(folder):
            subtree[anc].append(it)
            child_folders[anc].add(prev)
            prev = anc

    rollups: List[Rollup] = []
    # Process deepest folders first so summaries are stable bottom-up.
    for folder in sorted(subtree, key=lambda f: f.count("/") + f.count(os.sep), reverse=True):
        members = subtree[folder]
        if len(members) < min_items:
            continue
        n_direct = len(direct.get(folder, []))
        subfolders = sorted(child_folders.get(folder, set()))
        titles = [m.get("title") or m.get("uri", "") for m in members[:max_children]]
        body = "\n".join(
            f"- {m.get('title') or m.get('uri','')}: {(m.get('summary') or '')[:240]}"
            for m in members[:max_children]
        )
        subfolder_line = (
            f"Sub-folders: {', '.join(os.path.basename(s) or s for s in subfolders[:12])}\n"
            if subfolders else ""
        )
        base_text = (
            f"Folder: {folder}\n"
            f"Holds {len(members)} document(s) in total "
            f"({n_direct} directly, across {len(subfolders)} sub-folder(s)).\n"
            f"{subfolder_line}"
            f"Key documents: {', '.join(titles[:8])}\n\n"
            f"Contents:\n{body}"
        )
        text = base_text
        if summarizer is not None:
            try:
                synthesized = summarizer.summarize(base_text, title=f"Folder {folder}")
                if synthesized:
                    text = (
                        f"Folder: {folder} ({len(members)} documents, "
                        f"{len(subfolders)} sub-folders)\n{synthesized}"
                    )
            except Exception:
                text = base_text
        rollups.append(
            Rollup(
                rollup_id=f"folder_{stable_key(folder)}",
                kind="folder",
                key=folder,
                text=text,
                doc_type=DOC_TYPE_FOLDER_ROLLUP,
            )
        )
    return rollups


def build_section_rollup(
    section_name: str,
    section_id: str,
    members: List[Dict],
    *,
    summarizer=None,
    max_children: int = 50,
) -> Optional[Rollup]:
    if not members:
        return None
    body = "\n".join(
        f"- {m.get('title') or m.get('uri','')}: {(m.get('summary') or '')[:300]}"
        for m in members[:max_children]
    )
    base_text = (
        f"Section: {section_name}\n"
        f"{len(members)} document(s).\n\nContents:\n{body}"
    )
    text = base_text
    if summarizer is not None:
        try:
            synthesized = summarizer.summarize(base_text, title=f"Section {section_name}")
            if synthesized:
                text = f"Section: {section_name} ({len(members)} documents)\n{synthesized}"
        except Exception:
            text = base_text
    return Rollup(
        rollup_id=f"section_{section_id}",
        kind="section",
        key=section_id,
        text=text,
        doc_type=DOC_TYPE_SECTION_ROLLUP,
    )
