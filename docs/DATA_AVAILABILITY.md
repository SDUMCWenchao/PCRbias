# Data availability and reproducibility fixtures

The thesis FASTQ/BAM/intermediate tables are not committed because they are large and may contain private project paths or unpublished sample metadata. The public repository now includes a tiny synthetic fixture under `examples/synthetic/` so users can validate path wiring, parsers, and smoke-test behavior without needing the original sequencing data.

## Included public fixture

- `examples/synthetic/samples_meta.synthetic.tsv` — minimal two-sample metadata table.
- `examples/synthetic/raw_fastq/*.fastq` — artificial reads with matching sequence/quality lengths.
- `examples/synthetic/refs/synthetic_12s.fasta` — artificial reference amplicons.

These records are synthetic and are not biologically meaningful. They are intended only for software checks. Replace every fixture path with local real-data paths in `configs/config.yaml` or command-line arguments before running scientific analyses.

## Excluded data classes

Keep the following files outside Git and point to them through local configuration:

- raw FASTQ/FASTA/FASTQ.GZ sequencing files;
- BAM/SAM/CRAM alignment files and indexes;
- large count tables, feature matrices, model binaries, and attribution arrays;
- Slurm logs, temporary chunks, and private sample sheets.
