# Synthetic example dataset

This directory provides tiny artificial files that make the public repository self-contained without redistributing private raw sequencing data. The sequences are synthetic placeholders for parser/schema checks only; they are not biologically meaningful and must not be used for scientific inference.

Files:

- `samples_meta.synthetic.tsv`: minimal metadata in the Chapter 4 sample-manifest style.
- `raw_fastq/synthetic_pcr.fastq`: toy PCR sample reads.
- `raw_fastq/synthetic_nonpcr.fastq`: toy non-PCR/control reads.
- `refs/synthetic_12s.fasta`: toy reference amplicons.

Use these files to test path wiring, manifest validation, and lightweight parser behavior before substituting real local data configured in `configs/config.yaml`.
