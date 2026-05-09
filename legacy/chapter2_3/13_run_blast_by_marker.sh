#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash 13_run_blast_by_marker.sh <QUERY_DIR> <BLASTDB_DIR> <OUT_DIR> [THREADS_PER_MARKER]
#
# QUERY_DIR should contain:
#   annotation_queries_12S.fasta
#   annotation_queries_16S.fasta

if [[ $# -lt 3 ]]; then
  echo "Usage: bash $0 <QUERY_DIR> <BLASTDB_DIR> <OUT_DIR> [THREADS_PER_MARKER]" >&2
  exit 1
fi

QUERY_DIR="$1"
BLASTDB_DIR="$2"
OUT_DIR="$3"
THREADS="${4:-32}"

mkdir -p "$OUT_DIR"
command -v blastn >/dev/null 2>&1 || { echo "ERROR: blastn not found in PATH"; exit 1; }

run_one() {
  local marker="$1"
  local query="${QUERY_DIR}/annotation_queries_${marker}.fasta"
  local db="${BLASTDB_DIR}/${marker}_ref_db"
  local out="${OUT_DIR}/${marker}.blast.tsv"
  [[ -f "$query" ]] || { echo "ERROR: missing query file: $query" >&2; exit 1; }

  blastn \
    -query "$query" \
    -db "$db" \
    -task megablast \
    -perc_identity 90 \
    -qcov_hsp_perc 70 \
    -max_target_seqs 10 \
    -num_threads "$THREADS" \
    -outfmt "6 qseqid sseqid pident length qcovs evalue bitscore sscinames stitle" \
    -out "$out"
}

run_one 12S &
pid1=$!
run_one 16S &
pid2=$!

wait $pid1
wait $pid2

echo "[INFO] BLAST finished. Outputs in: $OUT_DIR"
