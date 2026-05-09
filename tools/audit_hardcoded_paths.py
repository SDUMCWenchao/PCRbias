#!/usr/bin/env python3
"""Report hard-coded private/server paths in the repository.

By default, this command fails if private/server paths are found. Use
`--legacy-mode report` only when auditing an unreleased downstream tree where
private paths are temporarily expected and should be reported without failing.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("/datapool", "/home/", "C:/Users/")
EXCLUDE_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
EXCLUDE_FILES = frozenset(
    {"docs/HARDCODED_PATHS.tsv", "tools/audit_hardcoded_paths.py", "tests/test_tools.py"}
)
Hit = tuple[Path, int, str]


def _is_excluded(path: Path, root: Path, exclude_dirs: Iterable[str], exclude_files: Iterable[str]) -> bool:
    """Return True when ``path`` should be skipped by the scanner."""
    rel = path.relative_to(root)
    excluded_dirs = set(exclude_dirs)
    excluded_files = set(exclude_files)
    return str(rel) in excluded_files or any(part in excluded_dirs for part in rel.parts)


def scan(
    root: Path = ROOT,
    patterns: Sequence[str] = PATTERNS,
    exclude_dirs: Iterable[str] = EXCLUDE_DIRS,
    exclude_files: Iterable[str] = EXCLUDE_FILES,
) -> list[Hit]:
    """Return hard-coded path hits under ``root``.

    Parameters are injectable so tests and downstream users can scan temporary
    trees without mutating module-level constants.
    """
    hits: list[Hit] = []
    root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or _is_excluded(path, root, exclude_dirs, exclude_files):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(pattern in line for pattern in patterns):
                hits.append((path.relative_to(root), lineno, line.strip()))
    return hits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-mode",
        choices=["fail", "report"],
        default="fail",
        help="Use 'report' to report hits without failing for temporary downstream audits.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to scan (defaults to the parent of tools/).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Additional private/server path pattern to flag. Can be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    patterns = (*PATTERNS, *(args.patterns or ()))
    hits = scan(root=args.root, patterns=patterns)
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
