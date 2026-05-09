#!/usr/bin/env python3
"""Regenerate docs/SCRIPT_INVENTORY.tsv."""
from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "SCRIPT_INVENTORY.tsv"
EXCLUDE_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
TYPE_BY_SUFFIX = {
    ".py": "python",
    ".sh": "shell",
    ".slurm": "slurm",
    ".R": "r",
    ".tsv": "tsv",
}
Row = tuple[Path, str, int]


def classify(path: Path) -> str:
    """Return the inventory type label for ``path``."""
    return TYPE_BY_SUFFIX.get(path.suffix, path.suffix.lstrip(".") or "file")


def iter_inventory_rows(root: Path = ROOT, exclude_dirs: Iterable[str] = EXCLUDE_DIRS) -> list[Row]:
    """Return sorted inventory rows for files under ``root``."""
    rows: list[Row] = []
    root = root.resolve()
    excluded = set(exclude_dirs)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in excluded for part in rel.parts):
            continue
        try:
            lines = len(path.read_text(errors="ignore").splitlines())
        except OSError:
            lines = 0
        rows.append((rel, classify(path), lines))
    return rows


def write_inventory(rows: Sequence[Row], output: Path = OUT) -> None:
    """Write inventory rows as tab-separated values."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("path", "type", "lines"))
        writer.writerows((str(path), typ, lines) for path, typ, lines in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to inventory (defaults to the parent of tools/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="TSV file to write (defaults to docs/SCRIPT_INVENTORY.tsv).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = iter_inventory_rows(args.root)
    write_inventory(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
