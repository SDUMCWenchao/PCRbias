# Main output classes

This file summarizes the expected output classes. Exact filenames may vary across historical variants; prefer the canonical paths in `docs/RUN_ORDER.chapter2_3.tsv` and `docs/RUN_ORDER.chapter4.tsv`.

## Chapter 2–3

| Output class | Typical location | Purpose |
|---|---|---|
| Filtered FASTQ | `02_filtered/` or configured filtered directory | Quality-filtered reads for abundance construction |
| Master abundance table | `03_tables/abundance/master_long_abundance.tsv` | Long-format sample-sequence abundance table |
| Sequence catalog | `03_tables/abundance/sequence_catalog.tsv` and `.fasta` | Deduplicated sequence records with stable IDs |
| Annotated threshold tables | `03_tables/annotated_tables/` | Thresholded and annotated abundance records |
| Renormalized threshold tables | `03_tables/annotated_tables/threshold_annotated_renorm/` | Tables used for recomputed bias statistics |
| Bias summaries | `05_stats/chapter2/` | Inter/intra mixture bias and target/non-target summaries |
| Chapter 3 feature tables | `04_features/chapter3/` | Global, regional, mismatch, RNAfold, and motif features |
| Chapter 3 statistics | `05_stats/chapter3/` | Univariate statistics and case-candidate tables |

## Chapter 4

| Output class | Typical location | Purpose |
|---|---|---|
| Manifest | `analysis_results/00_manifest/manifest.tsv` | Links sample metadata to local FASTQ files |
| Per-sample sequence stats | `analysis_results/01_Sequences/<file_id>_stats.tsv` | Count and relative abundance per sequence per sample |
| Global sequence database | `analysis_results/01_Sequences/global_sequences.sqlite` | Unique sequence bank |
| Global sequence FASTA | `analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta` | Sequence FASTA for feature extraction |
| Sequence features | `analysis_results/02_*` | Global sequence features and regional k-mer features |
| Pair/count database | `analysis_results/03_*` | PCR/non-PCR pair and count data for statistics/modeling |
| Feature statistics | `analysis_results/04_*` | Single-feature and joint-rank statistics |
| Model inputs | `analysis_results/05_ModelInputs/` | ML matrices and target arrays |
| Model outputs | `analysis_results/06_*` | RF/XGBoost/CNN models, metrics, predictions |
| Explainability outputs | `analysis_results/07_*` | SHAP and integrated-gradient tables |
| Paper-ready exports | `analysis_results/08_*` | Tables for thesis/manuscript writing |

## Files that should usually stay out of Git

Do not commit:

- raw FASTQ/FASTA/BAM/SAM/CRAM files from real data;
- large intermediate tables;
- model binaries such as `.joblib`, `.pkl`, `.pt`, `.pth`;
- private local configuration files;
- Slurm logs, temporary files, and caches.

Synthetic toy fixtures may be committed only if they are small and non-biological.
