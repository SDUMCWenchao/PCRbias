# PCR_bias

Open-source workflow draft for analyzing PCR amplification bias in mammalian mitochondrial 12S/16S amplicon sequencing.

This repository contains the thesis analysis scripts for two related workflows:

- **Chapter 2–3 workflow**: sequencing-data preprocessing, abundance table construction, sequence annotation, species-label harmonization, threshold sensitivity analysis, sequence-feature extraction, and univariate statistics.
- **Chapter 4 workflow**: PCR/non-PCR comparative analysis, sequence and k-mer feature extraction, paired statistical tests, machine-learning datasets, Random Forest/XGBoost/1D-CNN modeling, SHAP/integrated-gradient interpretation, external validation, and paper-ready table export.

The current repository is a **public-release draft**. The original scripts are preserved under `legacy/` to keep the thesis analyses traceable. A later refactoring step should convert canonical scripts into a cleaner command-line package.

## Repository layout

```text
.
├── legacy/
│   ├── chapter2_3/        # Chapter 2–3 legacy workflow scripts
│   └── chapter4/          # Chapter 4 legacy workflow scripts
├── configs/               # Example YAML configuration
├── docs/                  # Audit report, pipeline notes, deployment notes, roadmap
├── examples/              # Example metadata templates; no real sequencing data
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
python tools/audit_hardcoded_paths.py --legacy-mode report
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

## Configuration

Copy the example configuration and edit paths for the target server:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Do not commit `configs/config.yaml`, private metadata, raw sequencing data, BAM files, trained models, logs, or large intermediate tables.

## Current limitations

- Legacy scripts still contain private server path defaults. These are retained only for provenance and are documented in `docs/HARDCODED_PATHS.tsv`.
- Several scripts are historical alternatives (`_v2`, `_v3`, `topbias`, `resplit`, `external`). Canonical/deprecated status needs final scientific verification.
- The public repository contains metadata examples but no raw data.
- Dependency versions are inferred from script imports and should be pinned after recreating the final server environment.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

## Citation

See `CITATION.cff`.
