#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/datapool/zhangw/duwenchao/var/2511_PCR_Bias/external_test}"
MODELS_ROOT="${2:-$PROJECT_DIR/analysis_results/06_Models_external_topbias_v2_resplit_v1}"
OUT_TAR="${3:-$PROJECT_DIR/analysis_results/_deliver/external_attr_tables.tar.gz}"

mkdir -p "$(dirname "$OUT_TAR")"
tmp_list=$(mktemp)
trap 'rm -f "$tmp_list"' EXIT

cd "$PROJECT_DIR"

find "$MODELS_ROOT" -type f \( \
  -path "*/shap_tables/*.tsv" -o -path "*/shap_tables/meta.json" -o \
  -path "*/attr_tables/*.tsv" -o -path "*/attr_tables/meta.json" -o \
  -name "metrics.json" \
\) | sed "s|^$PROJECT_DIR/||" > "$tmp_list"

echo "[INFO] files_to_pack = $(wc -l < "$tmp_list")"
tar -czf "$OUT_TAR" -C "$PROJECT_DIR" -T "$tmp_list"
echo "[DONE] packed -> $OUT_TAR"
