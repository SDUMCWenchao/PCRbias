#!/usr/bin/env bash
set -euo pipefail

MODELS_ROOT="${1:-/path/to/PCR_bias_chapter4/external_test/analysis_results/06_Models_external_topbias_v2_resplit_v1}"
OUT_PREFIX="${2:-/path/to/PCR_bias_chapter4/external_test/_upload/external_attr_pack_$(date +%Y%m%d_%H%M%S)}"
SPLIT_SIZE="${3:-1800m}"  # 单文件上限附近：你自己按上传限制改

mkdir -p "$(dirname "$OUT_PREFIX")"

TMP_DIR="$(mktemp -d)"
MANIFEST="$TMP_DIR/manifest.tsv"
echo -e "path\tsize_bytes" > "$MANIFEST"

# 只收集：metrics.json + pred_test.tsv + shap/ig 表（gzip）
# 这样你上传给我我就能复核解释质量 & 与指标对齐
while IFS= read -r f; do
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo -e "${f}\t${sz}" >> "$MANIFEST"
done < <(
  find "$MODELS_ROOT" -type f \( \
    -name "metrics.json" -o \
    -name "pred_test.tsv" -o \
    -path "*/shap_tables/*.tsv.gz" -o \
    -path "*/attr_tables/*.tsv.gz" -o \
    -path "*/shap_tables/*.json" -o \
    -path "*/attr_tables/*.json" \
  \) | sort
)

# 生成 tar.gz
ARCHIVE="${OUT_PREFIX}.tar.gz"
tar -czf "$ARCHIVE" -T <(cut -f1 "$MANIFEST" | tail -n +2)

# 分卷（可选）
split -b "$SPLIT_SIZE" -d -a 2 "$ARCHIVE" "${OUT_PREFIX}.tar.gz.part"

echo "[DONE] archive: $ARCHIVE"
echo "[DONE] parts: ${OUT_PREFIX}.tar.gz.part00 ..."
echo "[DONE] manifest: $MANIFEST"
echo "Tip: 上传时把所有 part 文件一起上传即可。"
