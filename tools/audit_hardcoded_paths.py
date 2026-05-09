#!/usr/bin/env python3
"""Report hard-coded private/server paths in the repository.

By default, this command fails if private/server paths are found. For the current
legacy-preserving public draft, use `--legacy-mode report` to report hits without
failing CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["/datapool", "/home/", "C:/Users/"]
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDE_FILES = {"docs/HARDCODED_PATHS.tsv"}


def scan() -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if str(rel) in EXCLUDE_FILES:
            continue
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(pattern in line for pattern in PATTERNS):
                hits.append((rel, lineno, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-mode",
        choices=["fail", "report"],
        default="fail",
        help="Use 'report' while legacy scripts intentionally preserve original paths.",
    )
    args = parser.parse_args()

    hits = scan()
    if hits:
        print("Hard-coded path hits:")
        for rel, lineno, line in hits:
            print(f"{rel}:{lineno}: {line}")
        if args.legacy_mode == "report":
            print(f"Reported {len(hits)} hits; not failing because --legacy-mode report was used.")
            return 0
        return 1

    print("No hard-coded private/server paths detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
