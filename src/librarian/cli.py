"""Command-line interface: ``librarian <command>``.

    librarian index <path-or-url> [--root DIR] [--source ID]
    librarian search "<query>" [--root DIR] [-k N] [--json]
    librarian context "<query>" [--root DIR] [-k N]
    librarian stats [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import LibrarianConfig
from .librarian import Librarian


def _make_lib(args) -> Librarian:
    cfg = LibrarianConfig(root=args.root)
    if getattr(args, "openai", False):
        cfg.embedding_provider = "openai"
        cfg.summarizer_provider = "openai"
    return Librarian(cfg)


def _cmd_index(args) -> int:
    lib = _make_lib(args)
    target = args.target
    if target.startswith(("http://", "https://")):
        lib.add_url(target, source_id=args.source or "web", max_pages=args.max_pages)
    else:
        lib.add_path(target, source_id=args.source or "filesystem")
    stats = lib.build(force=args.force)
    lib.close()
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_search(args) -> int:
    lib = _make_lib(args)
    results = lib.search(args.query, k=args.k)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
    else:
        if not results:
            print("No results.")
        for i, ev in enumerate(results, 1):
            print(f"{i}. [{ev.score:.3f}] {ev.citation()}  <{ev.doc_type}>")
            excerpt = ev.excerpt.replace("\n", " ")
            print(f"   {excerpt[:200]}")
    lib.close()
    return 0


def _cmd_context(args) -> int:
    lib = _make_lib(args)
    print(lib.context(args.query, k=args.k))
    lib.close()
    return 0


def _cmd_stats(args) -> int:
    lib = _make_lib(args)
    print(json.dumps(lib.stats(), indent=2))
    lib.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="librarian", description="The Librarian knowledge layer.")
    parser.add_argument("--root", default="./librarian_data", help="Data directory.")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI backends.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Catalog a path or URL.")
    p_index.add_argument("target")
    p_index.add_argument("--source", default=None)
    p_index.add_argument("--max-pages", type=int, default=100, dest="max_pages")
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=_cmd_index)

    p_search = sub.add_parser("search", help="Search the knowledge base.")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=_cmd_search)

    p_ctx = sub.add_parser("context", help="Print an injectable context block.")
    p_ctx.add_argument("query")
    p_ctx.add_argument("-k", type=int, default=8)
    p_ctx.set_defaults(func=_cmd_context)

    p_stats = sub.add_parser("stats", help="Show catalog stats.")
    p_stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
