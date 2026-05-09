#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <BASE_DIR> <METADATA_TSV> [THREADS]" >&2
  exit 1
fi
BASE_DIR="$1"
META="$2"
THREADS="${3:-16}"
RAW_QC_DIR="${BASE_DIR}/02_qc_trim_filter/raw_qc"
FILT_DIR="${BASE_DIR}/02_qc_trim_filter/filtered"
LOG_DIR="${BASE_DIR}/02_qc_trim_filter/logs"
QC_STATS_DIR="${BASE_DIR}/03_tables/qc_stats"
mkdir -p "$RAW_QC_DIR" "$FILT_DIR" "$LOG_DIR" "$QC_STATS_DIR"
command -v cutadapt >/dev/null 2>&1 || { echo "ERROR: cutadapt not found in PATH"; exit 1; }
command -v seqkit >/dev/null 2>&1 || { echo "ERROR: seqkit not found in PATH"; exit 1; }

reverse_complement() {
python3 - "$1" << 'PY'
import sys
seq = sys.argv[1].strip().upper()
table = str.maketrans("ACGTRYMKBDHVN", "TGCAYRKMVHDBN")
print(seq.translate(table)[::-1])
PY
}
count_reads() {
  local f="$1"
  if [[ "$f" == *.gz ]]; then zcat "$f" | awk 'END{print NR/4}'
  else awk 'END{print NR/4}' "$f"; fi
}
tail -n +2 "$META" | while IFS=$'\t' read -r sample_id file_path marker group_name sample_type species_scope is_core_analysis fwd rev min_len max_len notes
do
  [[ -z "${sample_id}" ]] && continue
  [[ -f "$file_path" ]] || { echo "Missing file: $file_path" >&2; exit 1; }
  rcrev="$(reverse_complement "$rev")"
  raw_qc_txt="${RAW_QC_DIR}/${sample_id}.seqkit.stats.txt"
  filtered_fq="${FILT_DIR}/${sample_id}.filtered.fq.gz"
  cutadapt_log="${LOG_DIR}/${sample_id}.cutadapt.log"
  qc_row="${QC_STATS_DIR}/${sample_id}.qc.tsv"
  echo "[INFO] Processing ${sample_id}"
  seqkit stats -T "$file_path" > "$raw_qc_txt"
  cutadapt -j "$THREADS" -g "$fwd" -a "$rcrev" --discard-untrimmed -q 20,20 --max-n 0 -m "$min_len" -M "$max_len" -o "$filtered_fq" "$file_path" > "$cutadapt_log" 2>&1
  raw_reads="$(count_reads "$file_path")"
  filt_reads="$(count_reads "$filtered_fq")"
  printf "sample_id\traw_reads\tfiltered_reads\tretention_rate\tmarker\tgroup_name\tsample_type\tspecies_scope\tis_core_analysis\n" > "$qc_row"
  python3 - << PY >> "$qc_row"
raw_reads = int(${raw_reads})
filt_reads = int(${filt_reads})
ret = 0 if raw_reads == 0 else filt_reads / raw_reads
print(f"{'${sample_id}'}\t{raw_reads}\t{filt_reads}\t{ret:.6f}\t{'${marker}'}\t{'${group_name}'}\t{'${sample_type}'}\t{'${species_scope}'}\t{'${is_core_analysis}'}")
PY
done
python3 - "$QC_STATS_DIR" << 'PY'
from pathlib import Path
import pandas as pd, sys
d = Path(sys.argv[1])
files = sorted(d.glob("*.qc.tsv"))
dfs = [pd.read_csv(f, sep="\t") for f in files]
out = d / "qc_summary.tsv"
pd.concat(dfs, ignore_index=True).to_csv(out, sep="\t", index=False)
print(f"Wrote {out}")
PY
echo "[INFO] Done."
