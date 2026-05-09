"""Small sequence I/O helpers shared by refactored PCR_bias scripts."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import gzip
import hashlib
from typing import TextIO


def open_text_maybe_gz(path: str | Path, mode: str = "rt") -> TextIO:
    """Open a plain-text or gzip-compressed file in text mode."""
    path = Path(path)
    if "b" in mode:
        raise ValueError("open_text_maybe_gz is for text mode only")
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8", errors="replace")


def iter_fastq_sequences(path: str | Path, validate: bool = False) -> Iterator[str]:
    """Yield upper-case sequence strings from a single-end FASTQ file."""
    path = Path(path)
    with open_text_maybe_gz(path, "rt") as handle:
        record_no = 0
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline()
            plus = handle.readline()
            qual = handle.readline()
            record_no += 1
            if not (seq and plus and qual):
                raise ValueError(f"truncated FASTQ record {record_no} in {path}")
            seq = seq.rstrip("\n\r").upper()
            qual = qual.rstrip("\n\r")
            if validate:
                if not header.startswith("@"):
                    raise ValueError(f"FASTQ record {record_no} in {path} has invalid header")
                if not plus.startswith("+"):
                    raise ValueError(f"FASTQ record {record_no} in {path} has invalid plus line")
                if len(seq) != len(qual):
                    raise ValueError(
                        f"FASTQ record {record_no} in {path} has sequence/quality length mismatch"
                    )
            if seq:
                yield seq


def md5_seq_id(seq: str, prefix: str = "SEQ_", length: int = 16) -> str:
    """Return a deterministic MD5-based sequence identifier."""
    digest = hashlib.md5(seq.upper().encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"
