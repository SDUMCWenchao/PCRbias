"""Reusable utilities for the PCR_bias thesis analysis scripts.

The package is intentionally small at this stage. Legacy scripts remain under
``legacy/`` for provenance; new shared helpers should be added here first and
then adopted by canonical scripts gradually.
"""

from .schema import CHAPTER23_METADATA_COLUMNS, CHAPTER4_METADATA_COLUMNS, validate_columns
from .seqio import iter_fastq_sequences, md5_seq_id, open_text_maybe_gz

__all__ = [
    "CHAPTER23_METADATA_COLUMNS",
    "CHAPTER4_METADATA_COLUMNS",
    "iter_fastq_sequences",
    "md5_seq_id",
    "open_text_maybe_gz",
    "validate_columns",
]
