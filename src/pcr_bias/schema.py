"""Schema definitions and validation helpers for PCR_bias tables."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import csv

CHAPTER4_METADATA_COLUMNS: tuple[str, ...] = (
    "file_id",
    "sample_name",
    "species",
    "n_individuals",
    "locus",
    "pcr",
)

CHAPTER4_MANIFEST_COLUMNS: tuple[str, ...] = (
    "file_id",
    "sample_name",
    "species",
    "n_individuals",
    "locus",
    "pcr",
    "input_type",
    "fastq_path",
    "source_dir",
    "match_rule",
    "n_matches",
    "all_matches",
    "status",
)

CHAPTER23_METADATA_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "marker",
    "group_name",
    "sample_type",
    "species_scope",
    "is_core_analysis",
)

VALID_LOCI = frozenset({"12S", "16S", "srRNA", "lrRNA"})
VALID_PCR_VALUES = frozenset({"yes", "no", "true", "false", "1", "0", "y", "n", "PCR", "NoPCR"})


def validate_columns(columns: Iterable[str], required: Sequence[str], table_name: str = "table") -> None:
    """Raise ``ValueError`` if required columns are absent."""
    observed = set(columns)
    missing = [column for column in required if column not in observed]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def read_tsv_header(path: Path) -> list[str]:
    """Return the header row of a TSV file."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty TSV file: {path}") from exc


def validate_tsv_schema(path: Path, required: Sequence[str], table_name: str | None = None) -> list[str]:
    """Validate a TSV header and return the observed columns."""
    header = read_tsv_header(path)
    validate_columns(header, required, table_name or str(path))
    return header


def normalize_pcr_value(value: str) -> str:
    """Normalize common PCR yes/no values to ``yes`` or ``no``."""
    v = str(value).strip().lower()
    if v in {"yes", "true", "1", "y", "pcr"}:
        return "yes"
    if v in {"no", "false", "0", "n", "nopcr", "non-pcr", "non_pcr"}:
        return "no"
    raise ValueError(f"unrecognized PCR value: {value!r}")


def validate_chapter4_metadata_rows(rows: Sequence[dict[str, str]]) -> None:
    """Validate essential Chapter 4 metadata fields for parsed rows."""
    if not rows:
        raise ValueError("Chapter 4 metadata contains no rows")
    validate_columns(rows[0].keys(), CHAPTER4_METADATA_COLUMNS, "Chapter 4 metadata")
    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):
        file_id = str(row.get("file_id", "")).strip()
        if not file_id:
            raise ValueError(f"empty file_id at metadata line {i}")
        if file_id in seen:
            raise ValueError(f"duplicate file_id at metadata line {i}: {file_id}")
        seen.add(file_id)
        locus = str(row.get("locus", "")).strip()
        if locus and locus not in VALID_LOCI:
            raise ValueError(f"unexpected locus at metadata line {i}: {locus!r}")
        normalize_pcr_value(str(row.get("pcr", "")))
