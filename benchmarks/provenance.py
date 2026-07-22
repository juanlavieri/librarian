"""Write a reproducible content manifest for benchmark source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


EXCLUDED_PREFIXES = (
    "benchmarks/data/",
    "benchmarks/runs/",
    "benchmarks/results/",
    "dist/",
)


def source_manifest() -> dict:
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    files = {}
    tree = hashlib.sha256()
    for relative in sorted(listed):
        if (
            not (relative.startswith(("src/", "benchmarks/")) or relative == "pyproject.toml")
            or
            relative.startswith(EXCLUDED_PREFIXES)
            or "__pycache__/" in relative
            or relative.endswith((".pyc", ".pyo"))
        ):
            continue
        path = Path(relative)
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(bytes.fromhex(digest))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": tree.hexdigest(),
        "files": files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(source_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

