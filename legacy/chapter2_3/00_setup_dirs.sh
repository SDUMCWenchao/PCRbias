#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${1:-/path/to/chapter2_3_analysis}"
mkdir -p   "${BASE_DIR}/00_meta"   "${BASE_DIR}/01_raw_fastq"   "${BASE_DIR}/02_qc_trim_filter/raw_qc"   "${BASE_DIR}/02_qc_trim_filter/trimmed"   "${BASE_DIR}/02_qc_trim_filter/filtered"   "${BASE_DIR}/02_qc_trim_filter/logs"   "${BASE_DIR}/03_tables/abundance"   "${BASE_DIR}/03_tables/qc_stats"   "${BASE_DIR}/04_features/seq_basic"   "${BASE_DIR}/04_features/mismatch"   "${BASE_DIR}/04_features/rnafold"   "${BASE_DIR}/04_features/kmer"   "${BASE_DIR}/05_stats/chapter2"   "${BASE_DIR}/05_stats/chapter3"   "${BASE_DIR}/06_figures/main"   "${BASE_DIR}/06_figures/supplementary"   "${BASE_DIR}/07_scripts"   "${BASE_DIR}/08_slurm"   "${BASE_DIR}/09_logs"   "${BASE_DIR}/10_docs"
echo "Created: ${BASE_DIR}"
