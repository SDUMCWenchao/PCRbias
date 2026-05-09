# PCR_bias

Open-source workflow draft for analyzing PCR amplification bias in mammalian mitochondrial 12S/16S amplicon sequencing.

This repository contains the thesis analysis scripts for two related workflows:

- **Chapter 2–3 workflow**: sequencing-data preprocessing, abundance table construction, sequence annotation, species-label harmonization, threshold sensitivity analysis, sequence-feature extraction, and univariate statistics.
- **Chapter 4 workflow**: PCR/non-PCR comparative analysis, sequence and k-mer feature extraction, paired statistical tests, machine-learning datasets, Random Forest/XGBoost/1D-CNN modeling, SHAP/integrated-gradient interpretation, external validation, and paper-ready table export.

The current repository is a **public-release draft**. The original scripts are preserved under `legacy/` to keep the thesis analyses traceable, but private server path defaults have been replaced with public placeholders. A later refactoring step should convert canonical scripts into a cleaner command-line package.

## Repository layout

```text
.
├── legacy/
│   ├── chapter2_3/        # Chapter 2–3 legacy workflow scripts
│   └── chapter4/          # Chapter 4 legacy workflow scripts
├── configs/               # Example YAML configuration
├── docs/                  # Audit report, pipeline notes, deployment notes, roadmap
├── examples/              # Metadata templates and tiny synthetic parser fixtures
├── tools/                 # Repository audit and smoke-test utilities
├── .github/workflows/     # GitHub Actions syntax checks
├── environment.yml        # Conda environment draft
├── requirements.txt       # Python dependency draft
├── pyproject.toml         # Project metadata
├── CITATION.cff           # Citation metadata
└── LICENSE                # AGPL-3.0 license
```

## Installation

```bash
conda env create -f environment.yml
conda activate pcr-bias-pipeline
```

For a lightweight Python-only check:

```bash
python -m pip install -r requirements.txt
```

## Sanity checks

```bash
bash tools/smoke_test.sh
python tools/audit_hardcoded_paths.py
python -m pytest
```

`smoke_test.sh` checks Python and Shell/Slurm syntax. It does not validate scientific outputs because the public draft does not include the original large FASTQ/BAM/intermediate data.

The audit and inventory utilities also support temporary or downstream repositories via explicit roots:

```bash
python tools/audit_hardcoded_paths.py --root /path/to/repo --pattern /private/path
python tools/make_inventory.py --root /path/to/repo --output /tmp/script_inventory.tsv
```

## Recommended entry points

### Chapter 2–3

```bash
cd legacy/chapter2_3
bash 27_run_chapter3_pipeline_resume.sh /path/to/chapter2_3_analysis 48 /path/to/RNAfold
```

### Chapter 4

Chapter 4 is currently a legacy collection rather than a single polished CLI. See:

```text
docs/CHAPTER4_PIPELINE.md
```

The recommended public strategy is to treat `00*`–`08*` as the canonical internal order and keep `ext*` as external-validation scripts.


## Synthetic fixture

The repository includes artificial FASTQ/FASTA records under `examples/synthetic/` for lightweight validation only:

```text
examples/synthetic/samples_meta.synthetic.tsv
examples/synthetic/raw_fastq/
examples/synthetic/refs/
```

These files are not scientifically meaningful. Use them to test parser/schema wiring, then configure real local data paths in `configs/config.yaml`. See `docs/DATA_AVAILABILITY.md` for excluded data classes and reproducibility guidance.

## Configuration

Copy the example configuration and edit paths for the target server:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Do not commit `configs/config.yaml`, private metadata, raw sequencing data, BAM files, trained models, logs, or large intermediate tables.

## Public-release improvements

- Legacy private server paths have been replaced with public placeholders such as `/path/to/PCR_bias_chapter4` and `/path/to/chapter2_3_analysis`. Override them with command-line arguments or `configs/config.yaml` before running analyses.
- Chapter 4 script status is documented in `docs/CHAPTER4_SCRIPT_STATUS.tsv`, with canonical/support scripts separated from historical variants and external-validation utilities.
- A tiny synthetic, non-biological fixture is included in `examples/synthetic/` for parser and path-wiring checks; original large sequencing data remain intentionally excluded.
- Python dependency versions are pinned in `requirements.txt`, and the Conda draft mirrors those pins where practical.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

## Citation

See `CITATION.cff`.
