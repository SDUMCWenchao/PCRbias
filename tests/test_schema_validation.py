from __future__ import annotations

from pathlib import Path

import pytest

from pcr_bias.schema import (
    CHAPTER4_METADATA_COLUMNS,
    normalize_pcr_value,
    validate_chapter4_metadata_rows,
    validate_columns,
    validate_tsv_schema,
)
from pcr_bias.seqio import iter_fastq_sequences, md5_seq_id


def test_validate_columns_reports_missing() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(["file_id", "sample_name"], CHAPTER4_METADATA_COLUMNS, "metadata")


def test_validate_tsv_schema_returns_header(tmp_path: Path) -> None:
    table = tmp_path / "samples.tsv"
    table.write_text(
        "file_id\tsample_name\tspecies\tn_individuals\tlocus\tpcr\n"
        "H12\tMP10_12\tmix\t10\t12S\tyes\n",
        encoding="utf-8",
    )

    header = validate_tsv_schema(table, CHAPTER4_METADATA_COLUMNS)

    assert header == list(CHAPTER4_METADATA_COLUMNS)


def test_chapter4_metadata_validation_rejects_duplicate_file_id() -> None:
    rows = [
        {"file_id": "H12", "sample_name": "a", "species": "mix", "n_individuals": "10", "locus": "12S", "pcr": "yes"},
        {"file_id": "H12", "sample_name": "b", "species": "mix", "n_individuals": "10", "locus": "16S", "pcr": "no"},
    ]

    with pytest.raises(ValueError, match="duplicate file_id"):
        validate_chapter4_metadata_rows(rows)


def test_pcr_normalization() -> None:
    assert normalize_pcr_value("YES") == "yes"
    assert normalize_pcr_value("NoPCR") == "no"


def test_iter_fastq_sequences_and_md5(tmp_path: Path) -> None:
    fastq = tmp_path / "x.fastq"
    fastq.write_text("@r1\nACGT\n+\nIIII\n@r2\nttcc\n+\nIIII\n", encoding="utf-8")

    assert list(iter_fastq_sequences(fastq, validate=True)) == ["ACGT", "TTCC"]
    assert md5_seq_id("acgt") == md5_seq_id("ACGT")
