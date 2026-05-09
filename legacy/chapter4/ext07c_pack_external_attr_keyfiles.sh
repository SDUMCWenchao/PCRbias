#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/path/to/PCR_bias_chapter4/external_test"
MODELS_ROOT="$PROJECT_DIR/analysis_results/06_Models_external_topbias_v2_resplit_v1"

OUTROOT="$PROJECT_DIR/analysis_results/_pack_attr_keyfiles"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$OUTROOT/external_attr_keyfiles_$STAMP"
mkdir -p "$OUTDIR"

echo "[INFO] scanning $MODELS_ROOT"

# Copy minimal key files
find "$MODELS_ROOT" -type f \( \
  -name "metrics.json" -o \
  -path "*/shap_tables/*.tsv" -o \
  -path "*/shap_tables/meta.json" -o \
  -path "*/attr_tables/*.tsv" -o \
  -path "*/attr_tables/meta.json" \
\) -print0 | while IFS= read -r -d '' f; do
  rel="${f#$MODELS_ROOT/}"
  mkdir -p "$OUTDIR/$(dirname "$rel")"
  cp -f "$f" "$OUTDIR/$rel"
done

tarball="$OUTROOT/external_attr_keyfiles_$STAMP.tar.gz"
tar -C "$OUTROOT" -czf "$tarball" "$(basename "$OUTDIR")"

echo "[DONE] packed -> $tarball"
echo "[HINT] upload this tar.gz to ChatGPT"

