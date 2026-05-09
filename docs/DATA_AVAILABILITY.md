# Data availability and public-release boundary

The repository is designed to make thesis analysis scripts inspectable and to support lightweight workflow checks. It does not include the original large sequencing datasets or most generated intermediate files.

## Included

- Legacy scripts needed to trace the thesis analyses.
- Public configuration templates.
- Documentation of expected schemas, run order, and output classes.
- Tiny synthetic examples and tests for parser/path-wiring validation.

## Excluded

- Original FASTQ/FASTA/BAM/SAM/CRAM files.
- Large abundance, feature, k-mer, model-input, model-output, and explanation tables.
- Trained model binaries.
- Private sample sheets and site-local `configs/config.yaml` files.
- Logs, Slurm output, caches, and temporary files.

## Reproduction guidance

To reproduce scientific results, prepare the real input data locally, copy `configs/config.example.yaml` to `configs/config.yaml`, edit local paths, and follow the run-order tables. The synthetic smoke test only confirms that selected scripts can parse tiny artificial FASTQ data and write expected output files.
