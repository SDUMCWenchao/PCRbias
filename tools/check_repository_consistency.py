#!/usr/bin/env python3
"""Check lightweight repository metadata and documentation consistency."""
from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY_URL = "https://github.com/SDUMCWenchao/PCRbias"
STALE_REPOSITORY_URLS = (
    "https://github.com/SDUMCWenchao/PCR_bias",
    "git@github.com:SDUMCWenchao/PCR_bias.git",
)
SCRIPT_SUFFIXES = {".py", ".sh", ".slurm", ".R"}
METADATA_FILES = (
    "README.md",
    "configs/config.example.yaml",
    "CITATION.cff",
    "pyproject.toml",
)
RUN_ORDER_FILES = (
    "docs/RUN_ORDER.chapter2_3.tsv",
    "docs/RUN_ORDER.chapter4.tsv",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_repository_url(root: Path = ROOT) -> list[str]:
    """Return problems for stale or missing repository URLs in public metadata."""
    problems: list[str] = []
    for rel in METADATA_FILES:
        path = root / rel
        if not path.exists():
            problems.append(f"missing metadata file: {rel}")
            continue
        text = _read_text(path)
        for stale_url in STALE_REPOSITORY_URLS:
            if stale_url in text:
                problems.append(f"stale repository URL in {rel}: {stale_url}")
        if rel in {"configs/config.example.yaml", "CITATION.cff", "README.md"}:
            if EXPECTED_REPOSITORY_URL not in text:
                problems.append(f"expected repository URL missing from {rel}")
    return problems


def _status_paths(status_file: Path) -> set[str]:
    with status_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["path", "status", "notes"]:
            raise ValueError("CHAPTER4_SCRIPT_STATUS.tsv must have columns: path, status, notes")
        return {row["path"] for row in reader if row.get("path")}


def _chapter4_script_paths(root: Path) -> set[str]:
    chapter4_dir = root / "legacy" / "chapter4"
    if not chapter4_dir.exists():
        return set()
    return {
        str(path.relative_to(root))
        for path in chapter4_dir.iterdir()
        if path.is_file() and path.suffix in SCRIPT_SUFFIXES
    }


def check_chapter4_status_coverage(root: Path = ROOT) -> list[str]:
    """Return problems when Chapter 4 scripts lack status-table entries."""
    status_file = root / "docs" / "CHAPTER4_SCRIPT_STATUS.tsv"
    if not status_file.exists():
        return ["missing docs/CHAPTER4_SCRIPT_STATUS.tsv"]
    try:
        all_documented = _status_paths(status_file)
    except ValueError as exc:
        return [str(exc)]
    documented_scripts = {path for path in all_documented if Path(path).suffix in SCRIPT_SUFFIXES}
    actual_scripts = _chapter4_script_paths(root)
    missing = sorted(actual_scripts - documented_scripts)
    stale = sorted(path for path in all_documented if not (root / path).exists())
    problems = [f"missing Chapter 4 status entry: {path}" for path in missing]
    problems.extend(f"stale Chapter 4 status entry: {path}" for path in stale)
    return problems


def check_run_order_files(root: Path = ROOT) -> list[str]:
    """Check run-order tables exist and point to existing scripts."""
    problems: list[str] = []
    required_columns = [
        "step_id",
        "stage",
        "script",
        "status",
        "required_inputs",
        "main_outputs",
        "command_example",
        "resume_safe",
    ]
    for rel in RUN_ORDER_FILES:
        path = root / rel
        if not path.exists():
            problems.append(f"missing run-order table: {rel}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != required_columns:
                problems.append(f"unexpected columns in {rel}: {reader.fieldnames}")
                continue
            for row in reader:
                script = row.get("script", "").strip()
                if script and script != "NA" and not (root / script).exists():
                    problems.append(f"run-order script not found in {rel}: {script}")
    return problems


def run_checks(root: Path = ROOT) -> list[str]:
    return [
        *check_repository_url(root),
        *check_chapter4_status_coverage(root),
        *check_run_order_files(root),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to check (defaults to the parent of tools/).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problems = run_checks(args.root.resolve())
    if problems:
        print("Repository consistency problems:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("Repository metadata, run-order tables, and Chapter 4 status table are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
