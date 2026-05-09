#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash 12_build_local_blastdb.sh <REF_DIR> <OUT_DIR>
# REF_DIR must contain:
#   12S_ref.fa
#   16S_ref.fa
#
# Recommended FASTA header format:
# >Equus_asinus|REF001
# ACGT...

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <REF_DIR> <OUT_DIR>" >&2
  exit 1
fi

REF_DIR="$1"
OUT_DIR="$2"

mkdir -p "$OUT_DIR"
command -v makeblastdb >/dev/null 2>&1 || { echo "ERROR: makeblastdb not found in PATH"; exit 1; }

for marker in 12S 16S; do
  ref_fa="${REF_DIR}/${marker}_ref.fa"
  [[ -f "$ref_fa" ]] || { echo "ERROR: missing reference FASTA: $ref_fa" >&2; exit 1; }
  makeblastdb -in "$ref_fa" -dbtype nucl -parse_seqids -out "${OUT_DIR}/${marker}_ref_db"
done

echo "[INFO] BLAST databases created in: $OUT_DIR"
